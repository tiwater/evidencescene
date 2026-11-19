"""Validate meshes on an existing owner's exported registered-sample NPZ files.

NPZ fields: points_m, colors_rgb_u8, pixels_uv (original pixels), camera_slot,
image_shapes (one [height,width] per camera), plane_normal, plane_offset_m,
plane_support_count, plane_rmse_m, timestamp_ns. Samples must already be registered
and calibrated by their existing owner; this script performs no backprojection.
Raw sensor recordings and robot/world ground truth are never included in output.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from evidencescene import __version__
from evidencescene.surface import reconstruct_planar_surface, reconstruct_organized_surface


def verify(mesh, points, colors):
    if not mesh.ready:
        return {"ready": False, "reason": mesh.reason}
    np.testing.assert_array_equal(mesh.vertices_m, points[mesh.source_indices])
    np.testing.assert_array_equal(mesh.colors_rgb_u8, colors[mesh.source_indices])
    assert mesh.maximum_edge_m <= .024
    return dict(ready=True, vertices=len(mesh.vertices_m), triangles=len(mesh.triangles),
                area_m2=mesh.surface_area_m2, maximum_edge_m=mesh.maximum_edge_m)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('samples',type=Path); parser.add_argument('output',type=Path)
    args=parser.parse_args(); rows=[]
    for path in sorted(args.samples.glob('*.npz')):
        with np.load(path,allow_pickle=False) as f:
            p,c=f['points_m'],f['colors_rgb_u8']
            meta=dict(coordinate_frame='existing_registered_owner_frame',timestamp_ns=int(f['timestamp_ns']),source_id=path.name)
            start=time.perf_counter()
            mesh=reconstruct_planar_surface(p,c,plane_normal=f['plane_normal'],plane_offset_m=float(f['plane_offset_m']),
                plane_support_count=int(f['plane_support_count']),plane_rmse_m=float(f['plane_rmse_m']),**meta)
            duration=(time.perf_counter()-start)*1000
            planar=verify(mesh,p,c)
            assert mesh.ready, mesh.reason
            residual=np.abs(mesh.vertices_m @ f['plane_normal'] + f['plane_offset_m'])
            planar.update(vertex_plane_rmse_m=float(np.sqrt(np.mean(residual**2))),
                          vertex_plane_max_error_m=float(residual.max()),processing_ms=duration)
            organized=[]
            for cid,shape in enumerate(f['image_shapes']):
                selected=f['camera_slot']==cid
                if not selected.any(): continue
                camera_meta={**meta,'source_id':f'{path.name}/camera-{cid}'}
                result=reconstruct_organized_surface(p[selected],c[selected],pixels_uv=f['pixels_uv'][selected],
                    image_shape=tuple(int(x) for x in shape),pixel_step=4,**camera_meta)
                organized.append({'camera_slot':cid,**verify(result,p[selected],c[selected])})
            rows.append(dict(sample=path.name,input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                             timestamp_ns=str(meta['timestamp_ns']),planar=planar,organized=organized))
    assert rows, 'No registered samples supplied'
    assert all(any(c['ready'] for c in r['organized']) for r in rows)
    summary=dict(schema='evidencescene.measured-surface-validation.v1',version=__version__,frames=len(rows),
        exact_vertex_and_rgb_lineage=True,rows=rows,
        planar_p50_ms=float(np.median([r['planar']['processing_ms'] for r in rows])),
        planar_p95_ms=float(np.percentile([r['planar']['processing_ms'] for r in rows],95)),
        limits=['Recorded simulation RGB-D, not physical-store data.',
                'Plane residual is consistency with the supplied measured plane, not an independent ground-truth accuracy score.',
                'Topology interiors are interpolation; missing-data holes retained. No pose or watertightness claims.',
                'Timing is a short local replay, not a cross-machine benchmark.'])
    args.output.write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='rows'}))

if __name__=='__main__':main()
