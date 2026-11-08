// 1. Recorded Paper Figures Switcher
const views = [
  {
    image: '04-urdf-four-camera-measured-scene.png',
    alt: 'Measured multi-camera robot workspace reconstruction',
    kicker: '01 / SCENE',
    title: 'A measured workspace',
    description: 'A cross-calibrated 4-camera view of the robot and its manipulation workspace. Registered RGB-D observations provide metric geometry without CAD priors.',
    source: 'Paper figure · measured scene'
  },
  {
    image: '06-object-surface-completed-carton-frame245.png',
    alt: 'Object surface and completed carton geometry at frame 245',
    kicker: '02 / SURFACE',
    title: 'An object over time',
    description: 'Object-centric observations accumulate into a lineage-indexed RevisableSurfelMap as the object moves. Two-stage support10/coarse12mm selection retains multi-view corroborated surfels across the full envelope.',
    source: 'Paper figure · frame 245'
  },
  {
    image: '08-teleop-hud-top-xy-bullseye-frame275.png',
    alt: 'Top-down teleoperation visualization at frame 275',
    kicker: '03 / TOP VIEW',
    title: 'Alignment from above',
    description: 'An orthographic top XY bullseye view presents planar landing alignment and the 95% uncertainty ellipse relative to the measured container aperture.',
    source: 'Paper figure · frame 275'
  },
  {
    image: '09-teleop-hud-side-xz-clearance-frame275.png',
    alt: 'Side elevation teleoperation visualization at frame 275',
    kicker: '04 / SIDE VIEW',
    title: 'Clearance from the side',
    description: 'A side XZ elevation complements the top view with vertical rim clearance and the parabolic ballistic liquid stream locus.',
    source: 'Paper figure · frame 275'
  }
];

document.querySelectorAll('[data-view]').forEach(button => {
  button.addEventListener('click', () => {
    const index = Number(button.dataset.view);
    const view = views[index];
    const image = document.querySelector('#view-image');
    image.src = `../assets/figures/${view.image}`;
    image.alt = view.alt;
    const updates = {
      'view-index': `0${index + 1} — 04`,
      'view-kicker': view.kicker,
      'view-title': view.title,
      'view-description': view.description,
      'view-source': view.source
    };
    for (const [id, text] of Object.entries(updates)) {
      const el = document.getElementById(id);
      if (el) el.textContent = text;
    }
    document.querySelectorAll('[data-view]').forEach(item => {
      const selected = item === button;
      item.classList.toggle('active', selected);
      item.setAttribute('aria-pressed', String(selected));
    });
  });
});

// 2. Interactive 3D Algorithm Sandbox (RevisableSurfelMap + Schur Observability Gate + L2 Ballistic Pouring)
(() => {
  const canvas = document.getElementById('scene-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  const state = {
    yaw: -0.62,
    pitch: 0.44,
    distance: 0.68,
    dragging: false,
    lastX: 0,
    lastY: 0,
    policy: 'evidencescene', // 'evidencescene' | 'stride' | 'naive'
    observability: 'full',   // 'full' | 'collinear' | 'disconnected'
    offsetMm: 0.0,
    exitSpeedMps: 0.35,
    sigmaMm: 3.0
  };

  // Generate deterministic 3D carton surfels in object frame
  const baseSurfels = [];
  for (let ix = -3; ix <= 3; ix++) {
    for (let iy = -3; iy <= 3; iy++) {
      for (let iz = -5; iz <= 5; iz++) {
        if (Math.abs(ix) === 3 || Math.abs(iy) === 3 || Math.abs(iz) === 5) {
          const x = ix * 0.009;
          const y = iy * 0.009;
          const z = iz * 0.011;
          const support = ((ix + iy + iz + 15) % 5 === 0) ? 4 : 12;
          baseSurfels.push({ x, y, z, support });
        }
      }
    }
  }

  // Landmarks on the carton for factor-graph observability visualization
  const landmarksObj = [
    { id: 0, x: 0.027, y: -0.024, z: 0.055, collinear: true },
    { id: 1, x: 0.027, y: 0.000, z: 0.055, collinear: true },
    { id: 2, x: 0.027, y: 0.024, z: 0.055, collinear: true },
    { id: 3, x: -0.027, y: -0.024, z: 0.055, collinear: false },
    { id: 4, x: -0.027, y: 0.024, z: 0.055, collinear: false },
    { id: 5, x: 0.027, y: -0.020, z: -0.045, collinear: false },
    { id: 6, x: -0.027, y: 0.015, z: -0.045, collinear: false }
  ];

  // Target cup aperture on table: center (0.06, 0.0, -0.09), radius 0.045m
  const cupCenter = { x: 0.06, y: 0.0, z: -0.09 };
  const cupRadius = 0.045;
  const cupHeight = 0.075;

  function project3D(p, width, height) {
    const cosY = Math.cos(state.yaw), sinY = Math.sin(state.yaw);
    const cosP = Math.cos(state.pitch), sinP = Math.sin(state.pitch);
    const x1 = p.x * cosY - p.y * sinY;
    const y1 = p.x * sinY + p.y * cosY;
    const z1 = p.z;
    const z2 = y1 * cosP + z1 * sinP + state.distance;
    const y2 = z1 * cosP - y1 * sinP;
    const focal = Math.min(width, height) * 1.15;
    const scale = focal / Math.max(0.15, z2);
    return {
      u: width * 0.5 + x1 * scale,
      v: height * 0.52 - y2 * scale,
      depth: z2,
      scale
    };
  }

  function updateTelemetryAndRender() {
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    if (canvas.width !== Math.round(rect.width * dpr) || canvas.height !== Math.round(rect.height * dpr)) {
      canvas.width = Math.round(rect.width * dpr);
      canvas.height = Math.round(rect.height * dpr);
    }
    ctx.save();
    ctx.scale(dpr, dpr);
    const W = rect.width, H = rect.height;

    // Background gradient
    const grad = ctx.createLinearGradient(0, 0, 0, H);
    grad.addColorStop(0, '#0f1e1e');
    grad.addColorStop(1, '#152928');
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, W, H);

    // Draw reference table grid at z = -0.165
    ctx.strokeStyle = 'rgba(150, 185, 165, 0.13)';
    ctx.lineWidth = 1;
    for (let g = -0.20; g <= 0.201; g += 0.04) {
      const a = project3D({ x: g, y: -0.20, z: -0.165 }, W, H);
      const b = project3D({ x: g, y: 0.20, z: -0.165 }, W, H);
      ctx.beginPath(); ctx.moveTo(a.u, a.v); ctx.lineTo(b.u, b.v); ctx.stroke();
      const c = project3D({ x: -0.20, y: g, z: -0.165 }, W, H);
      const d = project3D({ x: 0.20, y: g, z: -0.165 }, W, H);
      ctx.beginPath(); ctx.moveTo(c.u, c.v); ctx.lineTo(d.u, d.v); ctx.stroke();
    }

    // Carton center pose (shifted by user slider Δx)
    const shiftX = state.offsetMm * 0.001;
    const cartonCenter = { x: -0.055 + shiftX, y: 0.0, z: 0.045 };
    const tiltRad = 0.28; // 16 deg pour tilt toward +x
    const cosT = Math.cos(tiltRad), sinT = Math.sin(tiltRad);

    function toWorld(localPt, extraDrift = null) {
      const rx = localPt.x * cosT + localPt.z * sinT;
      const rz = -localPt.x * sinT + localPt.z * cosT;
      return {
        x: cartonCenter.x + rx + (extraDrift ? extraDrift.x : 0),
        y: cartonCenter.y + localPt.y + (extraDrift ? extraDrift.y : 0),
        z: cartonCenter.z + rz + (extraDrift ? extraDrift.z : 0)
      };
    }

    // Draw target cup cylinder & rim aperture
    const rimPts = [];
    const basePts = [];
    for (let i = 0; i <= 48; i++) {
      const ang = (i / 48) * Math.PI * 2;
      rimPts.push(project3D({
        x: cupCenter.x + cupRadius * Math.cos(ang),
        y: cupCenter.y + cupRadius * Math.sin(ang),
        z: cupCenter.z
      }, W, H));
      basePts.push(project3D({
        x: cupCenter.x + cupRadius * 0.88 * Math.cos(ang),
        y: cupCenter.y + cupRadius * 0.88 * Math.sin(ang),
        z: cupCenter.z - cupHeight
      }, W, H));
    }

    // Cup vertical ribs
    ctx.strokeStyle = 'rgba(120, 195, 220, 0.22)';
    for (let i = 0; i < 48; i += 6) {
      ctx.beginPath();
      ctx.moveTo(rimPts[i].u, rimPts[i].v);
      ctx.lineTo(basePts[i].u, basePts[i].v);
      ctx.stroke();
    }
    ctx.strokeStyle = 'rgba(120, 195, 220, 0.35)';
    ctx.beginPath();
    basePts.forEach((p, idx) => idx === 0 ? ctx.moveTo(p.u, p.v) : ctx.lineTo(p.u, p.v));
    ctx.stroke();

    // Ballistic Pouring Stream & Landing Uncertainty Envelope
    const spoutLocal = { x: 0.028, y: 0.0, z: 0.055 };
    const spoutWorld = toWorld(spoutLocal);
    const dirX = 0.26, dirY = 0.0, dirZ = -0.9656;
    const v0 = state.exitSpeedMps;
    const g = 9.80665;
    const dz = spoutWorld.z - cupCenter.z;
    const vz0 = v0 * dirZ;
    const disc = vz0 * vz0 + 2 * g * dz;
    const tof = (-vz0 + Math.sqrt(Math.max(0, disc))) / g;

    const landingX = spoutWorld.x + v0 * dirX * tof;
    const landingY = spoutWorld.y + v0 * dirY * tof;
    const radialOffsetM = Math.hypot(landingX - cupCenter.x, landingY - cupCenter.y);

    // Propagated 95% landing uncertainty radius (chi2_2(0.95) = 2.4477)
    const sigmaPosM = state.sigmaMm * 0.001;
    const streamSpreadM = 0.004 * tof * 2.5;
    const r95M = 2.4477 * Math.hypot(sigmaPosM, streamSpreadM);
    const clearance95M = cupRadius - (radialOffsetM + r95M);
    const clearance95Mm = clearance95M * 1000.0;

    // Observability status (G + R)
    let rankText = '6 / 6 (Admitted)';
    let rankShort = 'r_i = 6/6 (Full Rank)';
    let obsAdmitted = true;
    if (state.observability === 'collinear') {
      rankText = '5 / 6 (R-Gate Rejected)';
      rankShort = 'r_i = 5/6 (Collinear Blocked)';
      obsAdmitted = false;
    } else if (state.observability === 'disconnected') {
      rankText = '0 / 6 (G-Gate Rejected)';
      rankShort = 'C\' Disconnected (Blocked)';
      obsAdmitted = false;
    }

    const safeToPour = obsAdmitted && clearance95M > 0.0;

    // Draw cup rim aperture (colored by safety)
    ctx.lineWidth = 2.4;
    ctx.strokeStyle = safeToPour ? '#8bc641' : '#f97316';
    ctx.beginPath();
    rimPts.forEach((p, idx) => idx === 0 ? ctx.moveTo(p.u, p.v) : ctx.lineTo(p.u, p.v));
    ctx.stroke();

    // Draw 95% landing uncertainty ellipse on aperture plane
    ctx.fillStyle = safeToPour ? 'rgba(139, 198, 65, 0.22)' : 'rgba(239, 68, 68, 0.26)';
    ctx.strokeStyle = safeToPour ? '#b6eb6d' : '#fca5a5';
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    for (let i = 0; i <= 40; i++) {
      const ang = (i / 40) * Math.PI * 2;
      const pt = project3D({
        x: landingX + r95M * Math.cos(ang),
        y: landingY + r95M * Math.sin(ang),
        z: cupCenter.z
      }, W, H);
      if (i === 0) ctx.moveTo(pt.u, pt.v);
      else ctx.lineTo(pt.u, pt.v);
    }
    ctx.closePath();
    ctx.fill();
    ctx.stroke();

    // Draw carton surfels according to selected policy
    const drawList = [];
    baseSurfels.forEach((s, idx) => {
      if (state.policy === 'stride' && idx % 3 !== 0) return;
      const wPt = toWorld(s);
      const proj = project3D(wPt, W, H);
      drawList.push({
        u: proj.u,
        v: proj.v,
        depth: proj.depth,
        color: s.support >= 10 ? '#b6eb6d' : '#5ea89c',
        radius: s.support >= 10 ? 3.1 : 2.3
      });
      // In 'naive' unrevised mode, also render un-purged ghost surfels from drifted keyframes 4-5 & frame 6
      if (state.policy === 'naive' && idx % 2 === 0) {
        const ghostPt = toWorld(s, { x: 0.024, y: -0.016, z: 0.014 });
        const gProj = project3D(ghostPt, W, H);
        drawList.push({
          u: gProj.u,
          v: gProj.v,
          depth: gProj.depth,
          color: 'rgba(248, 113, 113, 0.72)',
          radius: 2.5
        });
      }
    });

    drawList.sort((a, b) => b.depth - a.depth);
    for (const item of drawList) {
      ctx.fillStyle = item.color;
      ctx.beginPath();
      ctx.arc(item.u, item.v, item.radius, 0, Math.PI * 2);
      ctx.fill();
    }

    // Draw Factor-Graph Observability Rays from Prior Anchor X_0
    const anchorProj = project3D({ x: -0.17, y: -0.12, z: 0.14 }, W, H);
    ctx.fillStyle = '#38bdf8';
    ctx.beginPath();
    ctx.arc(anchorProj.u, anchorProj.v, 5, 0, Math.PI * 2);
    ctx.fill();
    ctx.font = '700 10px DM Sans, sans-serif';
    ctx.fillText('Prior X₀ (C₀)', anchorProj.u + 8, anchorProj.v + 3);

    landmarksObj.forEach(lm => {
      if (state.observability === 'collinear' && !lm.collinear) return;
      const wLm = toWorld(lm);
      const lmProj = project3D(wLm, W, H);
      if (state.observability !== 'disconnected') {
        ctx.strokeStyle = state.observability === 'collinear'
          ? 'rgba(250, 204, 21, 0.45)'
          : 'rgba(56, 189, 248, 0.35)';
        ctx.lineWidth = 1.2;
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(anchorProj.u, anchorProj.v);
        ctx.lineTo(lmProj.u, lmProj.v);
        ctx.stroke();
        ctx.setLineDash([]);
      }
      ctx.fillStyle = state.observability === 'collinear' ? '#facc15' : '#38bdf8';
      ctx.beginPath();
      ctx.arc(lmProj.u, lmProj.v, 3.8, 0, Math.PI * 2);
      ctx.fill();
    });

    // Draw parabolic ballistic stream centerline
    ctx.strokeStyle = safeToPour ? '#c3ef68' : '#fb7185';
    ctx.lineWidth = 2.8;
    ctx.beginPath();
    for (let step = 0; step <= 28; step++) {
      const t = (step / 28) * tof;
      const sx = spoutWorld.x + v0 * dirX * t;
      const sy = spoutWorld.y + v0 * dirY * t;
      const sz = spoutWorld.z + v0 * dirZ * t - 0.5 * g * t * t;
      const sp = project3D({ x: sx, y: sy, z: sz }, W, H);
      if (step === 0) ctx.moveTo(sp.u, sp.v);
      else ctx.lineTo(sp.u, sp.v);
    }
    ctx.stroke();

    // If unsafe due to offset, draw minimum-norm shared-control assist vector Δp_assist
    let assistDxMm = 0.0;
    if (clearance95M < 0 && radialOffsetM > 1e-6) {
      const neededShiftM = -clearance95M * ((cupCenter.x - landingX) / radialOffsetM);
      assistDxMm = neededShiftM * 1000.0;
      const pStart = project3D(spoutWorld, W, H);
      const pEnd = project3D({
        x: spoutWorld.x + neededShiftM,
        y: spoutWorld.y,
        z: spoutWorld.z
      }, W, H);
      ctx.strokeStyle = '#38bdf8';
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.moveTo(pStart.u, pStart.v);
      ctx.lineTo(pEnd.u, pEnd.v);
      ctx.stroke();
    }

    ctx.restore();

    // Update DOM telemetry
    let surfStats = '100.0% / 85.1%';
    let c10Short = '85.1%';
    let kernelText = '1.05 ms (14.77x)';
    if (state.policy === 'stride') {
      surfStats = '98.2% / 40.6%';
      c10Short = '40.6%';
      kernelText = '0.35 ms (Coverage Loss)';
    } else if (state.policy === 'naive') {
      surfStats = '61.4% / 76.8% (Drifted)';
      c10Short = '76.8% (Ghosting)';
      kernelText = '15.52 ms (Unoptimized 2D)';
    }

    const badge = document.getElementById('hud-badge');
    if (!obsAdmitted) {
      badge.textContent = 'L2 STATUS: BLOCKED (OBSERVABILITY GATE)';
      badge.className = 'hud-badge danger';
    } else if (safeToPour) {
      badge.textContent = 'L2 STATUS: SAFE TO POUR';
      badge.className = 'hud-badge safe';
    } else {
      badge.textContent = 'L2 STATUS: HOLD / REPOSITION';
      badge.className = 'hud-badge warn';
    }

    const sign = clearance95Mm >= 0 ? '+' : '';
    document.getElementById('hud-metrics').textContent =
      `${rankShort} · C₁₀ = ${c10Short} · Clearance₉₅ = ${sign}${clearance95Mm.toFixed(1)} mm · Kernel 8.65x`;
    document.getElementById('tel-rank').textContent = rankText;
    document.getElementById('tel-surface').textContent = surfStats;
    document.getElementById('tel-clearance').textContent = `${sign}${clearance95Mm.toFixed(1)} mm`;
    document.getElementById('tel-assist').textContent =
      `[${assistDxMm >= 0 ? '+' : ''}${assistDxMm.toFixed(1)}, 0.0, 0.0] mm`;
    const telKernel = document.getElementById('tel-kernel');
    if (telKernel) telKernel.textContent = kernelText;
  }

  // Orbit interaction
  canvas.addEventListener('pointerdown', e => {
    state.dragging = true;
    state.lastX = e.clientX;
    state.lastY = e.clientY;
    canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener('pointermove', e => {
    if (!state.dragging) return;
    const dx = e.clientX - state.lastX;
    const dy = e.clientY - state.lastY;
    state.lastX = e.clientX;
    state.lastY = e.clientY;
    state.yaw += dx * 0.008;
    state.pitch = Math.max(-0.2, Math.min(1.25, state.pitch + dy * 0.008));
    updateTelemetryAndRender();
  });
  canvas.addEventListener('pointerup', () => { state.dragging = false; });
  canvas.addEventListener('wheel', e => {
    e.preventDefault();
    state.distance = Math.max(0.42, Math.min(1.15, state.distance + e.deltaY * 0.0005));
    updateTelemetryAndRender();
  }, { passive: false });

  // Controls binding
  document.querySelectorAll('#policy-buttons .pill').forEach(btn => {
    btn.addEventListener('click', () => {
      state.policy = btn.dataset.policy;
      document.querySelectorAll('#policy-buttons .pill').forEach(b => b.classList.toggle('active', b === btn));
      updateTelemetryAndRender();
    });
  });

  document.querySelectorAll('#observability-buttons .pill').forEach(btn => {
    btn.addEventListener('click', () => {
      state.observability = btn.dataset.obs;
      document.querySelectorAll('#observability-buttons .pill').forEach(b => b.classList.toggle('active', b === btn));
      updateTelemetryAndRender();
    });
  });

  const ctrlOffset = document.getElementById('ctrl-offset');
  const ctrlSpeed = document.getElementById('ctrl-speed');
  const ctrlSigma = document.getElementById('ctrl-sigma');

  ctrlOffset.addEventListener('input', () => {
    state.offsetMm = Number(ctrlOffset.value);
    document.getElementById('val-offset').textContent = `${state.offsetMm >= 0 ? '+' : ''}${state.offsetMm} mm`;
    updateTelemetryAndRender();
  });
  ctrlSpeed.addEventListener('input', () => {
    state.exitSpeedMps = Number(ctrlSpeed.value);
    document.getElementById('val-speed').textContent = `${state.exitSpeedMps.toFixed(2)} m/s`;
    updateTelemetryAndRender();
  });
  ctrlSigma.addEventListener('input', () => {
    state.sigmaMm = Number(ctrlSigma.value);
    document.getElementById('val-sigma').textContent = `${state.sigmaMm.toFixed(1)} mm`;
    updateTelemetryAndRender();
  });

  window.addEventListener('resize', updateTelemetryAndRender);
  updateTelemetryAndRender();
})();
