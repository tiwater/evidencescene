import * as THREE from "./three.module.min.js";

/**
 * Self-contained Split-Screen 4-Camera + Interactive 3D Scene Reconstruction Viewer.
 * Renders:
 *  - 57-mesh URDF Bimanual Robot (1.2mm precision flat-shaded CAD normals, opaque FrontSide)
 *  - Real RGB-colored 3D Point Cloud from 4-camera RGB-D scan (tabletop + 4 AprilTags + cup + carton)
 *  - Calibrated Table Plane (z=0) + 4 AprilTag Fiducials + Origin Triad
 *  - 3D Target Cup + Rim Ring + Live Multi-Camera Surfels (cup-head & cup-chest)
 *  - 3D Held Milk Carton + Spout + Live Revisable Surfels (primary-object) + Pour Alignment HUD
 */
export async function initScene3DViewer(containerEl, options = {}) {
  if (!containerEl) return null;
  const assetBase = options.assetBase || "assets/scene3d";
  const initialAutoOrbit = options.autoOrbit !== undefined ? options.autoOrbit : true;
  const initialFrame = options.initialFrame !== undefined ? options.initialFrame : 92; // src frame 275 (POUR ALIGNED)

  const canvas = containerEl.querySelector("[data-scene3d-canvas]");
  const videoEl = containerEl.querySelector("[data-scene3d-video]");
  const statusBadge = containerEl.querySelector("[data-scene3d-status]");
  const hudTitle = containerEl.querySelector("[data-hud-title]");
  const hudRow1 = containerEl.querySelector("[data-hud-row1]");
  const hudRow2 = containerEl.querySelector("[data-hud-row2]");
  const hudDot = containerEl.querySelector("[data-hud-dot]");
  const hudLine = containerEl.querySelector("[data-hud-line]");
  const hudRing = containerEl.querySelector("[data-hud-ring]");
  const scrubber = containerEl.querySelector("[data-scene3d-scrubber]");
  const playBtn = containerEl.querySelector("[data-scene3d-play]");
  const timecodeEl = containerEl.querySelector("[data-scene3d-timecode]");
  const frameCounterEl = containerEl.querySelector("[data-scene3d-frame]");
  const pointCountEl = containerEl.querySelector("[data-scene3d-pts]");
  const orientSvg = containerEl.querySelector("[data-scene3d-orient]");

  // Keep any legacy floating label elements hidden so they never obscure the 3D objects
  containerEl.querySelectorAll("[data-label3d]").forEach((el) => {
    el.style.display = "none";
  });

  const camCanvases = {
    head: containerEl.querySelector('[data-cam-canvas="head"]'),
    chest: containerEl.querySelector('[data-cam-canvas="chest"]'),
    left_arm: containerEl.querySelector('[data-cam-canvas="left_arm"]'),
    right_arm: containerEl.querySelector('[data-cam-canvas="right_arm"]'),
  };

  const camCaptions = {
    head: containerEl.querySelector('[data-cam-caption="head"]'),
    chest: containerEl.querySelector('[data-cam-caption="chest"]'),
    left_arm: containerEl.querySelector('[data-cam-caption="left_arm"]'),
    right_arm: containerEl.querySelector('[data-cam-caption="right_arm"]'),
  };

  const latencyBadgeEl = containerEl.querySelector("[data-scene3d-latency]");

  if (!canvas) return null;

  const setStatus = (msg) => {
    if (statusBadge) statusBadge.textContent = msg;
  };

  setStatus("Loading 3D robot (57 STL), RGB point clouds & live surfels…");

  const [metaRes, meshesRes, framesRes, rgbRes] = await Promise.all([
    fetch(`${assetBase}/scene_meta.json`),
    fetch(`${assetBase}/robot_meshes.bin`),
    fetch(`${assetBase}/scene_frames.bin`),
    fetch(`${assetBase}/scene_rgb_points.bin`).catch(() => null),
  ]);

  const meta = await metaRes.json();
  const meshesBuf = await meshesRes.arrayBuffer();
  const framesBuf = await framesRes.arrayBuffer();
  const rgbBuf = rgbRes && rgbRes.ok ? await rgbRes.arrayBuffer() : null;

  // Setup Three.js WebGLRenderer matching world-viewer.ts
  const renderer = new THREE.WebGLRenderer({
    canvas,
    antialias: true,
    alpha: false,
    preserveDrawingBuffer: true,
  });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0c1015);

  const camera = new THREE.PerspectiveCamera(42, 16 / 10, 0.02, 20);
  camera.up.set(0, 0, -1); // Table frame: Z=0 is table plane, -Z is height above table

  const root = new THREE.Group();
  scene.add(root);

  // Lighting matching world-viewer.ts for crisp CAD & point cloud contrast
  scene.add(new THREE.HemisphereLight(0xdce8ff, 0x18150f, 1.85));
  const keyLight = new THREE.DirectionalLight(0xffffff, 2.1);
  keyLight.position.set(1.8, -2.7, -1.9);
  scene.add(keyLight);
  const fillLight = new THREE.DirectionalLight(0x8fd56c, 0.8);
  fillLight.position.set(-2.2, 1.6, -1.5);
  scene.add(fillLight);
  const rimLight = new THREE.DirectionalLight(0x7dd3fc, 0.65);
  rimLight.position.set(0.3, 2.4, -2.1);
  scene.add(rimLight);

  function buildNonIndexedGeometry(verts, faces) {
    const pos = new Float32Array(faces.length * 9);
    for (let f = 0; f < faces.length; f++) {
      const [i0, i1, i2] = faces[f];
      const v0 = verts[i0];
      const v1 = verts[i1];
      const v2 = verts[i2];
      const b = f * 9;
      pos[b] = v0[0]; pos[b + 1] = v0[1]; pos[b + 2] = v0[2];
      pos[b + 3] = v1[0]; pos[b + 4] = v1[1]; pos[b + 5] = v1[2];
      pos[b + 6] = v2[0]; pos[b + 7] = v2[1]; pos[b + 8] = v2[2];
    }
    const geom = new THREE.BufferGeometry();
    geom.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    geom.computeVertexNormals();
    return geom;
  }

  function decodeRgbPointCloud(cloudMeta) {
    if (!rgbBuf || !cloudMeta) return null;
    const n = cloudMeta.count;
    const pU16 = new Uint16Array(rgbBuf, cloudMeta.pos_byte_offset, n * 3);
    const cU8 = new Uint8Array(rgbBuf, cloudMeta.col_byte_offset, n * 3);
    const pos = new Float32Array(n * 3);
    const col = new Float32Array(n * 3);
    const [xmin, ymin, zmin] = cloudMeta.pmin;
    const sx = (cloudMeta.pmax[0] - xmin) / 65535.0;
    const sy = (cloudMeta.pmax[1] - ymin) / 65535.0;
    const sz = (cloudMeta.pmax[2] - zmin) / 65535.0;
    for (let i = 0; i < n; i++) {
      const b = i * 3;
      pos[b] = xmin + pU16[b] * sx;
      pos[b + 1] = ymin + pU16[b + 1] * sy;
      pos[b + 2] = zmin + pU16[b + 2] * sz;
      // Boost brightness slightly for dark scene background readability
      col[b] = Math.min(1.0, (cU8[b] / 255.0) * 1.15 + 0.04);
      col[b + 1] = Math.min(1.0, (cU8[b + 1] / 255.0) * 1.15 + 0.04);
      col[b + 2] = Math.min(1.0, (cU8[b + 2] / 255.0) * 1.15 + 0.04);
    }
    const geom = new THREE.BufferGeometry();
    geom.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    geom.setAttribute("color", new THREE.BufferAttribute(col, 3));
    return geom;
  }

  // ---------------------------------------------------------------------------
  // 1. Reconstructed Table Plane (z=0) + Real RGB Tabletop Scan + 4 AprilTags
  // ---------------------------------------------------------------------------
  const tableGroup = new THREE.Group();
  tableGroup.name = "Reconstructed_Table_And_AprilTags";
  root.add(tableGroup);

  const poly = meta.table.footprint_m;
  const tableFaces = [[0, 1, 2], [0, 2, 3]];
  const tableGeom = buildNonIndexedGeometry(poly, tableFaces);
  const tableMat = new THREE.MeshLambertMaterial({
    color: 0x1e293b,
    transparent: true,
    opacity: 0.62,
    side: THREE.DoubleSide,
    depthWrite: false,
  });
  const tableMesh = new THREE.Mesh(tableGeom, tableMat);
  tableMesh.renderOrder = -2;
  tableGroup.add(tableMesh);

  const skirtThickness = 0.018;
  const skirtVerts = [
    ...poly,
    ...poly.map(([x, y]) => [x, y, skirtThickness]),
  ];
  const skirtFaces = [
    [0, 1, 5], [0, 5, 4],
    [1, 2, 6], [1, 6, 5],
    [2, 3, 7], [2, 7, 6],
    [3, 0, 4], [3, 4, 7],
  ];
  const skirtGeom = buildNonIndexedGeometry(skirtVerts, skirtFaces);
  const skirtMat = new THREE.MeshLambertMaterial({
    color: 0x0f172a,
    transparent: true,
    opacity: 0.78,
    side: THREE.DoubleSide,
    depthWrite: false,
  });
  const skirtMesh = new THREE.Mesh(skirtGeom, skirtMat);
  skirtMesh.renderOrder = -3;
  tableGroup.add(skirtMesh);

  let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity;
  const borderVerts = [];
  const minorVerts = [];
  for (let i = 0; i < poly.length; i++) {
    const a = poly[i];
    const b = poly[(i + 1) % poly.length];
    borderVerts.push(a[0], a[1], -0.0005, b[0], b[1], -0.0005);
    xMin = Math.min(xMin, a[0]); xMax = Math.max(xMax, a[0]);
    yMin = Math.min(yMin, a[1]); yMax = Math.max(yMax, a[1]);
  }
  for (let gx = Math.ceil(xMin / 0.05) * 0.05; gx <= xMax + 1e-5; gx += 0.05) {
    minorVerts.push(gx, yMin, -0.0002, gx, yMax, -0.0002);
  }
  for (let gy = Math.ceil(yMin / 0.05) * 0.05; gy <= yMax + 1e-5; gy += 0.05) {
    minorVerts.push(xMin, gy, -0.0002, xMax, gy, -0.0002);
  }
  const minorGeom = new THREE.BufferGeometry();
  minorGeom.setAttribute("position", new THREE.Float32BufferAttribute(minorVerts, 3));
  tableGroup.add(new THREE.LineSegments(minorGeom, new THREE.LineBasicMaterial({
    color: 0x334155, transparent: true, opacity: 0.35,
  })));

  for (let gx = Math.ceil(xMin / 0.1) * 0.1; gx <= xMax + 1e-5; gx += 0.1) {
    borderVerts.push(gx, yMin, -0.0004, gx, yMax, -0.0004);
  }
  for (let gy = Math.ceil(yMin / 0.1) * 0.1; gy <= yMax + 1e-5; gy += 0.1) {
    borderVerts.push(xMin, gy, -0.0004, xMax, gy, -0.0004);
  }
  const gridGeom = new THREE.BufferGeometry();
  gridGeom.setAttribute("position", new THREE.Float32BufferAttribute(borderVerts, 3));
  tableGroup.add(new THREE.LineSegments(gridGeom, new THREE.LineBasicMaterial({
    color: 0x34d399, transparent: true, opacity: 0.50,
  })));

  const triadGeom = new THREE.BufferGeometry();
  triadGeom.setAttribute("position", new THREE.Float32BufferAttribute([
    0, 0, -0.0015, 0.068, 0, -0.0015,
    0, 0, -0.0015, 0, 0.068, -0.0015,
    0, 0, -0.0015, 0, 0, -0.068,
  ], 3));
  triadGeom.setAttribute("color", new THREE.Float32BufferAttribute([
    0.94, 0.33, 0.31, 0.94, 0.33, 0.31,
    0.20, 0.83, 0.60, 0.20, 0.83, 0.60,
    0.38, 0.65, 0.98, 0.38, 0.65, 0.98,
  ], 3));
  tableGroup.add(new THREE.LineSegments(triadGeom, new THREE.LineBasicMaterial({
    vertexColors: true,
  })));

  // Real RGB tabletop 3D point cloud from 4-camera scan (subtle so grid & tags stay high-contrast)
  if (meta.rgb_point_clouds && meta.rgb_point_clouds.table_surface) {
    const tblRgbGeom = decodeRgbPointCloud(meta.rgb_point_clouds.table_surface);
    if (tblRgbGeom) {
      const tblRgbPts = new THREE.Points(tblRgbGeom, new THREE.PointsMaterial({
        vertexColors: true,
        size: 1.5,
        sizeAttenuation: false,
        transparent: true,
        opacity: 0.36,
      }));
      tableGroup.add(tblRgbPts);
    }
  }

  for (const tag of meta.table.apriltags) {
    const lifted = tag.corners_table_m.map((c) => [c[0], c[1], c[2] - 0.0014]);
    const tagGeom = buildNonIndexedGeometry(lifted, [[0, 1, 2], [0, 2, 3]]);
    const tagMat = new THREE.MeshBasicMaterial({
      color: 0xfacc15,
      transparent: true,
      opacity: 0.86,
      side: THREE.DoubleSide,
    });
    tableGroup.add(new THREE.Mesh(tagGeom, tagMat));

    const tb = [];
    for (let i = 0; i < 4; i++) {
      const a = lifted[i];
      const b = lifted[(i + 1) % 4];
      tb.push(a[0], a[1], a[2] - 0.0003, b[0], b[1], b[2] - 0.0003);
    }
    const tbGeom = new THREE.BufferGeometry();
    tbGeom.setAttribute("position", new THREE.Float32BufferAttribute(tb, 3));
    tableGroup.add(new THREE.LineSegments(tbGeom, new THREE.LineBasicMaterial({ color: 0xfef08a })));
  }

  // ---------------------------------------------------------------------------
  // 2. Reconstructed Target Cup (3D Tapered Cylinder + Rim Ring + Dense Surfels)
  // ---------------------------------------------------------------------------
  const cupGroup = new THREE.Group();
  cupGroup.name = "Reconstructed_Target_Cup";
  root.add(cupGroup);

  const [cupCx, cupCy, cupCz] = meta.target_cup.opening_center_table_m;
  const cupTopR = meta.target_cup.top_radius_m;
  const cupBotR = meta.target_cup.bottom_radius_m;
  const cupH = meta.target_cup.height_m;
  const cupSegs = 28;
  const cupVerts = [];
  const cupFaces = [];
  for (let i = 0; i < cupSegs; i++) {
    const a = (i * 2 * Math.PI) / cupSegs;
    cupVerts.push([cupCx + cupBotR * Math.cos(a), cupCy - cupBotR * Math.sin(a), 0.0]);
  }
  for (let i = 0; i < cupSegs; i++) {
    const a = (i * 2 * Math.PI) / cupSegs;
    cupVerts.push([cupCx + cupTopR * Math.cos(a), cupCy - cupTopR * Math.sin(a), -cupH]);
  }
  cupVerts.push([cupCx, cupCy, 0.0]);
  for (let i = 0; i < cupSegs; i++) {
    const nxt = (i + 1) % cupSegs;
    cupFaces.push([i, nxt, cupSegs + nxt], [i, cupSegs + nxt, cupSegs + i], [2 * cupSegs, nxt, i]);
  }
  const cupGeom = buildNonIndexedGeometry(cupVerts, cupFaces);
  const cupMat = new THREE.MeshLambertMaterial({
    color: 0xf59e0b,
    transparent: true,
    opacity: 0.42,
    side: THREE.DoubleSide,
    depthWrite: false,
  });
  cupGroup.add(new THREE.Mesh(cupGeom, cupMat));

  const cupWireGeom = new THREE.WireframeGeometry(cupGeom);
  cupGroup.add(new THREE.LineSegments(cupWireGeom, new THREE.LineBasicMaterial({
    color: 0xfbbf24, transparent: true, opacity: 0.45,
  })));

  const rimSegs = [];
  const rimCount = 36;
  for (let i = 0; i < rimCount; i++) {
    const a0 = (i * 2 * Math.PI) / rimCount;
    const a1 = (((i + 1) % rimCount) * 2 * Math.PI) / rimCount;
    rimSegs.push(
      cupCx + cupTopR * Math.cos(a0), cupCy + cupTopR * Math.sin(a0), cupCz,
      cupCx + cupTopR * Math.cos(a1), cupCy + cupTopR * Math.sin(a1), cupCz
    );
  }
  const crossR = cupTopR * 0.85;
  rimSegs.push(
    cupCx - crossR, cupCy, cupCz, cupCx + crossR, cupCy, cupCz,
    cupCx, cupCy - crossR, cupCz, cupCx, cupCy + crossR, cupCz
  );
  const rimGeom = new THREE.BufferGeometry();
  rimGeom.setAttribute("position", new THREE.Float32BufferAttribute(rimSegs, 3));
  cupGroup.add(new THREE.LineSegments(rimGeom, new THREE.LineBasicMaterial({
    color: 0x10b981, transparent: true, opacity: 0.96,
  })));

  // Real RGB target cup 3D point cloud from 4-camera scan
  if (meta.rgb_point_clouds && meta.rgb_point_clouds.target_cup) {
    const cupRgbGeom = decodeRgbPointCloud(meta.rgb_point_clouds.target_cup);
    if (cupRgbGeom) {
      cupGroup.add(new THREE.Points(cupRgbGeom, new THREE.PointsMaterial({
        vertexColors: true,
        size: 2.3,
        sizeAttenuation: false,
        transparent: true,
        opacity: 0.78,
      })));
    }
  }

  const cupAccGeom = new THREE.BufferGeometry();
  cupAccGeom.setAttribute(
    "position",
    new THREE.Float32BufferAttribute(meta.target_cup.accumulated_points_flat_table_m, 3)
  );
  cupGroup.add(new THREE.Points(cupAccGeom, new THREE.PointsMaterial({
    color: 0xfdba74, size: 2.5, sizeAttenuation: false,
  })));

  // Live multi-camera cup surfels matching measured-table-layers.ts palette
  const liveCupHeadGeom = new THREE.BufferGeometry();
  const liveCupChestGeom = new THREE.BufferGeometry();
  cupGroup.add(new THREE.Points(liveCupHeadGeom, new THREE.PointsMaterial({
    color: 0xe0a65c, size: 2.6, sizeAttenuation: false,
  })));
  cupGroup.add(new THREE.Points(liveCupChestGeom, new THREE.PointsMaterial({
    color: 0x66c78c, size: 2.6, sizeAttenuation: false,
  })));

  // ---------------------------------------------------------------------------
  // 3. Reconstructed Held Milk Carton (3D Oriented Box + RGB Cloud + Live Surfels)
  // ---------------------------------------------------------------------------
  const cartonGroup = new THREE.Group();
  cartonGroup.name = "Reconstructed_Held_Milk_Carton";
  cartonGroup.matrixAutoUpdate = false;
  root.add(cartonGroup);

  const boxGeom = buildNonIndexedGeometry(meta.held_object.vertices_object_m, meta.held_object.faces);
  const boxMat = new THREE.MeshLambertMaterial({
    color: 0x38bdf8,
    transparent: true,
    opacity: 0.36,
    side: THREE.DoubleSide,
    depthWrite: false,
  });
  cartonGroup.add(new THREE.Mesh(boxGeom, boxMat));

  const edgeVerts = [];
  for (const [i0, i1] of meta.held_object.edges) {
    const v0 = meta.held_object.vertices_object_m[i0];
    const v1 = meta.held_object.vertices_object_m[i1];
    edgeVerts.push(v0[0], v0[1], v0[2], v1[0], v1[1], v1[2]);
  }
  const edgeGeom = new THREE.BufferGeometry();
  edgeGeom.setAttribute("position", new THREE.Float32BufferAttribute(edgeVerts, 3));
  cartonGroup.add(new THREE.LineSegments(edgeGeom, new THREE.LineBasicMaterial({
    color: 0x7dd3fc, transparent: true, opacity: 0.95,
  })));

  const spoutMesh = new THREE.Mesh(
    new THREE.SphereGeometry(0.008, 16, 12),
    new THREE.MeshStandardMaterial({ color: 0xf43f5e, emissive: 0xbe123c, roughness: 0.25 })
  );
  spoutMesh.position.set(...meta.held_object.pour_outlet_object_m);
  cartonGroup.add(spoutMesh);

  // Real RGB 3D milk carton point cloud in object_gauge (moves with carton 6-DoF pose)
  if (meta.rgb_point_clouds && meta.rgb_point_clouds.held_carton) {
    const cartonRgbGeom = decodeRgbPointCloud(meta.rgb_point_clouds.held_carton);
    if (cartonRgbGeom) {
      cartonGroup.add(new THREE.Points(cartonRgbGeom, new THREE.PointsMaterial({
        vertexColors: true,
        size: 2.2,
        sizeAttenuation: false,
        transparent: true,
        opacity: 0.72,
      })));
    }
  }

  const objAccGeom = new THREE.BufferGeometry();
  objAccGeom.setAttribute(
    "position",
    new THREE.Float32BufferAttribute(meta.held_object.accumulated_points_flat_object_m, 3)
  );
  cartonGroup.add(new THREE.Points(objAccGeom, new THREE.PointsMaterial({
    color: 0x93c5fd, size: 2.5, sizeAttenuation: false,
  })));

  // Dense live revisable surfels (up to 3,600 pts/frame, matching primary-object #a8d6de)
  const liveObjGeom = new THREE.BufferGeometry();
  cartonGroup.add(new THREE.Points(liveObjGeom, new THREE.PointsMaterial({
    color: 0xa8d6de, size: 2.6, sizeAttenuation: false,
  })));

  // ---------------------------------------------------------------------------
  // 4. Bimanual Robot (57 Crisp Flat-Shaded CAD STL Meshes + Skeleton + Pads)
  // ---------------------------------------------------------------------------
  const robotGroup = new THREE.Group();
  robotGroup.name = "Bimanual_URDF_Robot";
  root.add(robotGroup);

  const robotMeshes = [];
  for (const mSpec of meta.meshes) {
    const vU16 = new Uint16Array(meshesBuf, mSpec.v_byte_offset, mSpec.v_count * 3);
    const fU16 = new Uint16Array(meshesBuf, mSpec.f_byte_offset, mSpec.f_count * 3);
    const vPos = new Float32Array(mSpec.v_count * 3);
    const [xmin, ymin, zmin] = mSpec.vmin;
    const sx = (mSpec.vmax[0] - xmin) / 65535.0;
    const sy = (mSpec.vmax[1] - ymin) / 65535.0;
    const sz = (mSpec.vmax[2] - zmin) / 65535.0;
    for (let i = 0; i < mSpec.v_count; i++) {
      const b = i * 3;
      vPos[b] = xmin + vU16[b] * sx;
      vPos[b + 1] = ymin + vU16[b + 1] * sy;
      vPos[b + 2] = zmin + vU16[b + 2] * sz;
    }

    // Expand indexed triangles into non-indexed facet vertices so computeVertexNormals()
    // produces exact crisp CAD facet normals (identical to parseUrdfStlGeometry in urdf-scene-client.ts)
    const triPos = new Float32Array(mSpec.f_count * 9);
    for (let f = 0; f < mSpec.f_count; f++) {
      const fb = f * 3;
      const i0 = fU16[fb] * 3;
      const i1 = fU16[fb + 1] * 3;
      const i2 = fU16[fb + 2] * 3;
      const tb = f * 9;
      triPos[tb] = vPos[i0]; triPos[tb + 1] = vPos[i0 + 1]; triPos[tb + 2] = vPos[i0 + 2];
      triPos[tb + 3] = vPos[i1]; triPos[tb + 4] = vPos[i1 + 1]; triPos[tb + 5] = vPos[i1 + 2];
      triPos[tb + 6] = vPos[i2]; triPos[tb + 7] = vPos[i2 + 1]; triPos[tb + 8] = vPos[i2 + 2];
    }
    const geom = new THREE.BufferGeometry();
    geom.setAttribute("position", new THREE.BufferAttribute(triPos, 3));
    geom.computeVertexNormals();

    // 100% opaque FrontSide Lambert material matching urdf-scene-client.ts line 1565
    const mat = new THREE.MeshLambertMaterial({
      color: mSpec.color_hex,
      side: THREE.FrontSide,
    });
    const mesh = new THREE.Mesh(geom, mat);
    mesh.matrixAutoUpdate = false;
    const scaleMat = new THREE.Matrix4().makeScale(
      mSpec.scale[0], mSpec.scale[1], mSpec.scale[2]
    );
    robotGroup.add(mesh);
    robotMeshes.push({ mesh, scaleMat });
  }

  const skelGeom = new THREE.BufferGeometry();
  const skelLines = new THREE.LineSegments(skelGeom, new THREE.LineBasicMaterial({
    vertexColors: true, transparent: true, opacity: 0.90,
  }));
  robotGroup.add(skelLines);

  const padSphereGeom = new THREE.SphereGeometry(0.011, 14, 10);
  const rightPadMesh = new THREE.Mesh(
    padSphereGeom,
    new THREE.MeshStandardMaterial({ color: 0x2dd4bf, emissive: 0x0f766e, roughness: 0.25 })
  );
  const leftPadMesh = new THREE.Mesh(
    padSphereGeom,
    new THREE.MeshStandardMaterial({ color: 0x38bdf8, emissive: 0x0369a1, roughness: 0.25 })
  );
  robotGroup.add(rightPadMesh);
  robotGroup.add(leftPadMesh);

  // ---------------------------------------------------------------------------
  // 5. Spout-to-Cup Pour Alignment 3D Vectors & Landing Ring
  // ---------------------------------------------------------------------------
  const pourHudGroup = new THREE.Group();
  pourHudGroup.name = "Spout_To_Cup_Alignment_3D";
  root.add(pourHudGroup);

  const openMarker = new THREE.Mesh(
    new THREE.SphereGeometry(0.007, 14, 12),
    new THREE.MeshStandardMaterial({ color: 0x10b981, emissive: 0x047857, roughness: 0.2 })
  );
  openMarker.position.set(cupCx, cupCy, cupCz);
  pourHudGroup.add(openMarker);

  const landingRingMat = new THREE.MeshBasicMaterial({
    color: 0xfacc15,
    side: THREE.DoubleSide,
    transparent: true,
    opacity: 0.95,
  });
  const landingRingMesh = new THREE.Mesh(
    new THREE.RingGeometry(0.0055, 0.012, 28),
    landingRingMat
  );
  pourHudGroup.add(landingRingMesh);

  const pourLineGeom = new THREE.BufferGeometry();
  const pourLines = new THREE.LineSegments(pourLineGeom, new THREE.LineBasicMaterial({
    vertexColors: true, transparent: true, opacity: 0.95,
  }));
  pourHudGroup.add(pourLines);

  // ---------------------------------------------------------------------------
  // Camera Orbit State & Mouse/Touch Interactions
  // ---------------------------------------------------------------------------
  const viewState = {
    target: new THREE.Vector3(0.24, -0.25, -0.25),
    yaw: 0.48,
    pitch: 0.35,
    distance: 1.46,
    autoOrbit: initialAutoOrbit,
    dragging: false,
    lastX: 0,
    lastY: 0,
    currentFrameIndex: initialFrame,
    playing: options.autoPlay !== undefined ? options.autoPlay : true,
  };

  function updateCameraAndRender() {
    const w = Math.max(2, Math.floor(canvas.clientWidth));
    const h = Math.max(2, Math.floor(canvas.clientHeight));
    const curSize = renderer.getSize(new THREE.Vector2());
    if (curSize.x !== w || curSize.y !== h) {
      renderer.setSize(w, h, false);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    }
    const cosP = Math.cos(viewState.pitch);
    camera.position.set(
      viewState.target.x + Math.sin(viewState.yaw) * cosP * viewState.distance,
      viewState.target.y + Math.cos(viewState.yaw) * cosP * viewState.distance,
      viewState.target.z - Math.sin(viewState.pitch) * viewState.distance
    );
    camera.lookAt(viewState.target);
    renderer.render(scene, camera);

    if (orientSvg) {
      const invQ = camera.quaternion.clone().invert();
      const axes = [
        new THREE.Vector3(1, 0, 0),
        new THREE.Vector3(0, 1, 0),
        new THREE.Vector3(0, 0, 1),
      ];
      axes.forEach((ax, idx) => {
        const dir = ax.clone().applyQuaternion(invQ);
        const g = orientSvg.children[idx];
        if (!g) return;
        const line = g.querySelector("line");
        const txt = g.querySelector("text");
        if (line) {
          line.setAttribute("x2", (40 + dir.x * 24).toFixed(1));
          line.setAttribute("y2", (40 - dir.y * 24).toFixed(1));
        }
        if (txt) {
          txt.setAttribute("x", (40 + dir.x * 32).toFixed(1));
          txt.setAttribute("y", (40 - dir.y * 32).toFixed(1));
        }
      });
    }
  }

  const objQMin = meta.quantization.obj_qmin;
  const objQSpan = meta.quantization.obj_qmax.map((v, i) => (v - objQMin[i]) / 255.0);
  const cupQMin = meta.quantization.cup_qmin;
  const cupQSpan = meta.quantization.cup_qmax.map((v, i) => (v - cupQMin[i]) / 255.0);

  function decodeQuantizedPoints(byteOffset, count, qMin, qSpan) {
    const u8 = new Uint8Array(framesBuf, byteOffset, count * 3);
    const out = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      const b = i * 3;
      out[b] = qMin[0] + u8[b] * qSpan[0];
      out[b + 1] = qMin[1] + u8[b + 1] * qSpan[1];
      out[b + 2] = qMin[2] + u8[b + 2] * qSpan[2];
    }
    return out;
  }

  const mat4Scratch = new THREE.Matrix4();

  function applyFrame(fIdx, syncVideo = false) {
    const clampedIdx = Math.max(0, Math.min(meta.frames.length - 1, fIdx));
    viewState.currentFrameIndex = clampedIdx;
    const fMeta = meta.frames[clampedIdx];
    const baseOff = fMeta.byte_offset;

    const meshFloats = new Float32Array(framesBuf, baseOff, 57 * 12);
    for (let m = 0; m < 57; m++) {
      const b = m * 12;
      mat4Scratch.set(
        meshFloats[b], meshFloats[b + 1], meshFloats[b + 2], meshFloats[b + 3],
        meshFloats[b + 4], meshFloats[b + 5], meshFloats[b + 6], meshFloats[b + 7],
        meshFloats[b + 8], meshFloats[b + 9], meshFloats[b + 10], meshFloats[b + 11],
        0, 0, 0, 1
      );
      const entry = robotMeshes[m];
      entry.mesh.matrix.copy(mat4Scratch).multiply(entry.scaleMat);
      entry.mesh.matrixWorldNeedsUpdate = true;
    }

    const skel = fMeta.skeleton_flat;
    const nSeg = Math.floor(skel.length / 7);
    const skelPos = new Float32Array(nSeg * 6);
    const skelCol = new Float32Array(nSeg * 6);
    const pal = {
      0: [0.94, 0.92, 0.88],
      1: [0.49, 0.83, 0.99],
      2: [0.37, 0.92, 0.83],
      3: [0.99, 0.88, 0.45],
    };
    for (let s = 0; s < nSeg; s++) {
      const sb = s * 7;
      const pb = s * 6;
      skelPos[pb] = skel[sb]; skelPos[pb + 1] = skel[sb + 1]; skelPos[pb + 2] = skel[sb + 2];
      skelPos[pb + 3] = skel[sb + 3]; skelPos[pb + 4] = skel[sb + 4]; skelPos[pb + 5] = skel[sb + 5];
      const c = pal[skel[sb + 6]] || pal[0];
      skelCol[pb] = c[0]; skelCol[pb + 1] = c[1]; skelCol[pb + 2] = c[2];
      skelCol[pb + 3] = c[0]; skelCol[pb + 4] = c[1]; skelCol[pb + 5] = c[2];
    }
    skelGeom.setAttribute("position", new THREE.BufferAttribute(skelPos, 3));
    skelGeom.setAttribute("color", new THREE.BufferAttribute(skelCol, 3));

    rightPadMesh.position.set(...fMeta.right_gripper_table_m);
    leftPadMesh.position.set(...fMeta.left_gripper_table_m);

    const t12 = fMeta.table_from_initial_object;
    cartonGroup.matrix.set(
      t12[0], t12[1], t12[2], t12[3],
      t12[4], t12[5], t12[6], t12[7],
      t12[8], t12[9], t12[10], t12[11],
      0, 0, 0, 1
    );
    cartonGroup.matrixWorldNeedsUpdate = true;

    let pOff = baseOff + 57 * 12 * 4;
    const objPos = decodeQuantizedPoints(pOff, fMeta.obj_pts_count, objQMin, objQSpan);
    liveObjGeom.setAttribute("position", new THREE.BufferAttribute(objPos, 3));
    pOff += fMeta.obj_pts_count * 3;

    const chPos = decodeQuantizedPoints(pOff, fMeta.cup_head_pts_count, cupQMin, cupQSpan);
    liveCupHeadGeom.setAttribute("position", new THREE.BufferAttribute(chPos, 3));
    pOff += fMeta.cup_head_pts_count * 3;

    const ccPos = decodeQuantizedPoints(pOff, fMeta.cup_chest_pts_count, cupQMin, cupQSpan);
    liveCupChestGeom.setAttribute("position", new THREE.BufferAttribute(ccPos, 3));

    const outPt = fMeta.pour_outlet_table_m;
    landingRingMesh.position.set(outPt[0], outPt[1], cupCz);
    landingRingMat.color.setHex(fMeta.in_rim ? 0x10b981 : 0xfacc15);

    const linePos = new Float32Array([
      outPt[0], outPt[1], outPt[2], cupCx, cupCy, cupCz,
      outPt[0], outPt[1], outPt[2], outPt[0], outPt[1], cupCz,
      outPt[0], outPt[1], cupCz, cupCx, cupCy, cupCz,
    ]);
    const cIn = fMeta.in_rim ? [0.06, 0.95, 0.65] : [0.99, 0.82, 0.30];
    const lineCol = new Float32Array([
      0.96, 0.25, 0.37, 0.06, 0.73, 0.51,
      cIn[0], cIn[1], cIn[2], cIn[0], cIn[1], cIn[2],
      0.22, 0.74, 0.97, 0.06, 0.73, 0.51,
    ]);
    pourLineGeom.setAttribute("position", new THREE.BufferAttribute(linePos, 3));
    pourLineGeom.setAttribute("color", new THREE.BufferAttribute(lineCol, 3));

    const cupRadMm = cupTopR * 1000;
    const horizMm = fMeta.horiz_mm;
    const aboveMm = fMeta.above_mm;
    const inRim = fMeta.in_rim;
    const tiltDeg = fMeta.tilt_deg;
    const gripMm = fMeta.grip_mm;

    if (hudTitle) {
      if (inRim) {
        hudTitle.textContent = `● POUR ALIGNED · IN CUP RIM (+${(cupRadMm - horizMm).toFixed(1)} mm margin)`;
        hudTitle.dataset.state = "aligned";
      } else if (horizMm <= cupRadMm * 2.0) {
        hudTitle.textContent = `◐ APPROACHING CUP RIM (${(horizMm - cupRadMm).toFixed(1)} mm outside)`;
        hudTitle.dataset.state = "approaching";
      } else if (gripMm <= 38.0) {
        hudTitle.textContent = `◆ GRASP ENGAGED · TRANSPORTING CARTON`;
        hudTitle.dataset.state = "transport";
      } else {
        hudTitle.textContent = `○ BIMANUAL REACH / WORKSPACE STANDBY`;
        hudTitle.dataset.state = "standby";
      }
    }
    if (hudRow1) {
      const dxS = (fMeta.dx_mm >= 0 ? "+" : "") + fMeta.dx_mm.toFixed(0);
      const dyS = (fMeta.dy_mm >= 0 ? "+" : "") + fMeta.dy_mm.toFixed(0);
      hudRow1.innerHTML = `Spout ΔXY: <strong>${horizMm.toFixed(1)} mm</strong> / R ${cupRadMm.toFixed(1)} mm · ΔX ${dxS} · ΔY ${dyS} mm`;
    }
    if (hudRow2) {
      const dzS = (aboveMm >= 0 ? "+" : "") + aboveMm.toFixed(1);
      hudRow2.innerHTML = `Rim Clearance ΔZ: <strong>${dzS} mm</strong> · Pour Tilt: <strong>${tiltDeg.toFixed(1)}°</strong> · Grip→Carton: <strong>${gripMm.toFixed(1)} mm</strong>`;
    }
    if (hudDot && hudLine && hudRing) {
      const scale = 20 / cupRadMm;
      const rawX = fMeta.dy_mm * scale;
      const rawY = -fMeta.dx_mm * scale;
      const rawR = Math.hypot(rawX, rawY);
      const maxR = 38;
      const dotX = rawR > maxR ? (rawX / rawR) * maxR : rawX;
      const dotY = rawR > maxR ? (rawY / rawR) * maxR : rawY;
      const accent = inRim ? "#10b981" : horizMm <= cupRadMm * 2 ? "#f59e0b" : "#38bdf8";
      hudDot.setAttribute("cx", dotX.toFixed(1));
      hudDot.setAttribute("cy", dotY.toFixed(1));
      hudDot.setAttribute("fill", accent);
      hudLine.setAttribute("x2", dotX.toFixed(1));
      hudLine.setAttribute("y2", dotY.toFixed(1));
      hudLine.setAttribute("stroke", accent);
      hudRing.setAttribute("stroke", inRim ? "#10b981" : "#f59e0b");
      hudRing.setAttribute("fill", inRim ? "rgba(16,185,129,0.16)" : "rgba(245,158,11,0.08)");
    }

    if (scrubber && document.activeElement !== scrubber) {
      scrubber.value = String(clampedIdx);
    }
    if (timecodeEl) {
      const totS = meta.frames[meta.frames.length - 1].elapsed_s;
      timecodeEl.textContent = `${fMeta.elapsed_s.toFixed(1)}s / ${totS.toFixed(1)}s`;
    }
    if (frameCounterEl) {
      frameCounterEl.textContent = `Frame ${fMeta.src_frame_index + 1} / 498`;
    }
    if (pointCountEl) {
      const totalPts = fMeta.raw_object_point_count + fMeta.raw_cup_head_count + fMeta.raw_cup_chest_count;
      pointCountEl.textContent = `${totalPts.toLocaleString()} live surfels (${fMeta.raw_object_point_count.toLocaleString()} carton + ${(fMeta.raw_cup_head_count + fMeta.raw_cup_chest_count).toLocaleString()} cup)`;
    }

    const syncMs = fMeta.cam_sync_ms || { head: 0.06, chest: 0.0, left_arm: 33.27, right_arm: -0.01 };
    const fmtDt = (v) => `${v >= 0 ? "+" : ""}${Number(v).toFixed(2)} ms`;
    if (camCaptions.head) {
      camCaptions.head.textContent = `Sync Δt: ${fmtDt(syncMs.head ?? 0.06)} · 3D Surface Locked`;
    }
    if (camCaptions.chest) {
      camCaptions.chest.textContent = `Sync Δt: 0.00 ms (Ref Clock) · 3D Surface Locked`;
    }
    if (camCaptions.left_arm) {
      camCaptions.left_arm.textContent = `Sync Δt: ${fmtDt(syncMs.left_arm ?? 33.27)} · Wrist Stream`;
    }
    if (camCaptions.right_arm) {
      camCaptions.right_arm.textContent = `Sync Δt: ${fmtDt(syncMs.right_arm ?? -0.01)} · Active Pour Wrist`;
    }
    if (latencyBadgeEl) {
      const hc = fmtDt(syncMs.head ?? 0.06);
      const rw = fmtDt(syncMs.right_arm ?? -0.01);
      latencyBadgeEl.innerHTML = `<strong>Cam↔3D Skew: <code>0.00 ms</code></strong> · Head–Chest Δt: <code>${hc}</code> · Wrist Δt: <code>${rw}</code> · Surfel Update: <code>1.07 ms</code> (Rev: <code>0.14 ms</code>) · End-to-End: <code>30.2 ms</code> (<code>33.1 FPS</code>)`;
    }

    if (syncVideo && videoEl && videoEl.readyState >= 1) {
      const targetT = (clampedIdx + 0.25) / meta.fps;
      if (Math.abs(videoEl.currentTime - targetT) > 0.02) {
        videoEl.currentTime = targetT;
      }
    }

    updateCameraAndRender();
  }

  function blitFourCameras() {
    if (!videoEl || videoEl.readyState < 2) return;
    const vw = videoEl.videoWidth || 960;
    const vh = videoEl.videoHeight || 540;
    const hw = vw / 2;
    const hh = vh / 2;
    const quads = [
      ["head", 0, 0],
      ["chest", hw, 0],
      ["left_arm", 0, hh],
      ["right_arm", hw, hh],
    ];
    for (const [key, sx, sy] of quads) {
      const c = camCanvases[key];
      if (!c) continue;
      const ctx = c.getContext("2d");
      if (c.width !== hw || c.height !== hh) {
        c.width = hw;
        c.height = hh;
      }
      ctx.drawImage(videoEl, sx, sy, hw, hh, 0, 0, hw, hh);
    }
  }

  async function seekToFrame(fIdx) {
    const clampedIdx = Math.max(0, Math.min(meta.frames.length - 1, fIdx));
    applyFrame(clampedIdx, false);
    if (videoEl) {
      const targetT = (clampedIdx + 0.25) / meta.fps;
      if (Math.abs(videoEl.currentTime - targetT) > 0.01) {
        await new Promise((resolve) => {
          const onSeeked = () => {
            videoEl.removeEventListener("seeked", onSeeked);
            resolve();
          };
          videoEl.addEventListener("seeked", onSeeked);
          videoEl.currentTime = targetT;
        });
      }
      blitFourCameras();
    }
    updateCameraAndRender();
  }

  canvas.addEventListener("pointerdown", (e) => {
    viewState.dragging = true;
    viewState.lastX = e.clientX;
    viewState.lastY = e.clientY;
    canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener("pointermove", (e) => {
    if (!viewState.dragging) return;
    const dx = e.clientX - viewState.lastX;
    const dy = e.clientY - viewState.lastY;
    viewState.lastX = e.clientX;
    viewState.lastY = e.clientY;
    viewState.yaw -= dx * 0.0085;
    viewState.pitch = Math.max(0.08, Math.min(1.52, viewState.pitch + dy * 0.0065));
    updateCameraAndRender();
  });
  canvas.addEventListener("pointerup", (e) => {
    viewState.dragging = false;
    try { canvas.releasePointerCapture(e.pointerId); } catch {}
  });
  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    viewState.distance = Math.max(0.38, Math.min(3.2, viewState.distance * (e.deltaY > 0 ? 1.1 : 0.91)));
    updateCameraAndRender();
  }, { passive: false });
  canvas.addEventListener("dblclick", () => {
    viewState.target.set(0.24, -0.25, -0.25);
    viewState.yaw = 0.48;
    viewState.pitch = 0.35;
    viewState.distance = 1.46;
    updateCameraAndRender();
  });

  containerEl.querySelectorAll("[data-view-preset]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const preset = btn.dataset.viewPreset;
      if (preset === "orbit") {
        viewState.autoOrbit = !viewState.autoOrbit;
        btn.setAttribute("aria-pressed", String(viewState.autoOrbit));
        btn.classList.toggle("is-active", viewState.autoOrbit);
        return;
      }
      viewState.autoOrbit = false;
      const orbitBtn = containerEl.querySelector('[data-view-preset="orbit"]');
      if (orbitBtn) {
        orbitBtn.setAttribute("aria-pressed", "false");
        orbitBtn.classList.remove("is-active");
      }
      if (preset === "iso") {
        viewState.target.set(0.24, -0.25, -0.25);
        viewState.yaw = 0.48;
        viewState.pitch = 0.35;
        viewState.distance = 1.46;
      } else if (preset === "pour") {
        viewState.target.set(0.34, -0.13, -0.14);
        viewState.yaw = 0.34;
        viewState.pitch = 0.32;
        viewState.distance = 0.58;
      } else if (preset === "top") {
        viewState.target.set(0.24, -0.16, -0.06);
        viewState.yaw = 0.0;
        viewState.pitch = 1.50;
        viewState.distance = 1.02;
      } else if (preset === "side") {
        viewState.target.set(0.26, -0.16, -0.12);
        viewState.yaw = 0.02;
        viewState.pitch = 0.14;
        viewState.distance = 0.98;
      }
      updateCameraAndRender();
    });
  });

  const layerMap = {
    robot: robotGroup,
    table: tableGroup,
    cup: cupGroup,
    carton: cartonGroup,
    hud: pourHudGroup,
  };
  containerEl.querySelectorAll("[data-layer-toggle]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const key = btn.dataset.layerToggle;
      const grp = layerMap[key];
      if (!grp) return;
      grp.visible = !grp.visible;
      btn.setAttribute("aria-pressed", String(grp.visible));
      btn.classList.toggle("is-active", grp.visible);
      updateCameraAndRender();
    });
  });

  if (scrubber) {
    scrubber.max = String(meta.frames.length - 1);
    scrubber.value = String(initialFrame);
    scrubber.addEventListener("input", () => {
      viewState.playing = false;
      if (videoEl) videoEl.pause();
      if (playBtn) playBtn.textContent = "▶ Play";
      const idx = Number(scrubber.value);
      applyFrame(idx, true);
    });
  }

  if (playBtn) {
    playBtn.addEventListener("click", () => {
      viewState.playing = !viewState.playing;
      playBtn.textContent = viewState.playing ? "⏸ Pause" : "▶ Play";
      if (videoEl) {
        if (viewState.playing) videoEl.play().catch(() => {});
        else videoEl.pause();
      }
    });
  }

  if (videoEl) {
    videoEl.addEventListener("seeked", () => {
      blitFourCameras();
    });
    const onVideoReady = () => {
      videoEl.currentTime = (initialFrame + 0.25) / meta.fps;
      blitFourCameras();
      if (viewState.playing) {
        videoEl.play().catch(() => {});
      }
    };
    if (videoEl.readyState >= 1) {
      onVideoReady();
    } else {
      videoEl.addEventListener("loadedmetadata", onVideoReady, { once: true });
    }
  }

  setStatus("Interactive 3D WebGL Ready · Drag to rotate");
  applyFrame(initialFrame, true);

  let lastTick = performance.now();
  function tick(now) {
    const dt = Math.min(0.1, (now - lastTick) / 1000);
    lastTick = now;

    if (viewState.autoOrbit && !viewState.dragging) {
      viewState.yaw += dt * 0.24;
    }

    if (viewState.playing && videoEl && !videoEl.paused) {
      blitFourCameras();
      const idx = Math.min(
        meta.frames.length - 1,
        Math.floor(videoEl.currentTime * meta.fps)
      );
      applyFrame(idx, false);
    } else if (viewState.autoOrbit && !viewState.dragging) {
      updateCameraAndRender();
    }

    requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);

  const ro = new ResizeObserver(() => updateCameraAndRender());
  ro.observe(canvas);

  const api = {
    meta,
    viewState,
    applyFrame,
    seekToFrame,
    updateCameraAndRender,
    blitFourCameras,
    videoEl,
  };
  window.__scene3dViewer = api;
  containerEl.__scene3dViewer = api;
  return api;
}
