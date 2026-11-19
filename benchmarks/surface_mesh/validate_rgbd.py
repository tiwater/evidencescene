"""Validate standalone RGB-D meshing on NPZ exports (depth, rgb, K, timestamp_ns).

Inputs must already be registered color/depth. No robot, extrinsics, ground
truth, host application, or network is required. Source indices are pixel indices.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from evidencescene import __version__
from evidencescene.surface import reconstruct_rgbd_surface


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('samples',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();rows=[]
    for path in sorted(a.samples.glob('*.npz')):
        with np.load(path,allow_pickle=False) as f:
            start=time.perf_counter()
            mesh=reconstruct_rgbd_surface(f['depth'],f['rgb'],f['K'],coordinate_frame='camera/local',
                timestamp_ns=int(f['timestamp_ns']),source_id=path.stem,pixel_step=4)
            elapsed=(time.perf_counter()-start)*1000
            assert mesh.ready, (path.name,mesh.reason)
            v,u=np.divmod(mesh.source_indices,f['depth'].shape[1]);depth=f['depth'][v,u]
            np.testing.assert_array_equal(mesh.colors_rgb_u8,f['rgb'][v,u])
            np.testing.assert_allclose(mesh.vertices_m[:,2],depth,rtol=0,atol=0)
            np.testing.assert_allclose(mesh.vertices_m[:,0],(u-f['K'][0,2])*depth/f['K'][0,0],atol=1e-12)
            assert mesh.maximum_edge_m<=.024
            rows.append(dict(sample=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                timestamp_ns=str(mesh.timestamp_ns),vertices=len(mesh.vertices_m),triangles=len(mesh.triangles),
                maximum_edge_m=mesh.maximum_edge_m,surface_area_m2=mesh.surface_area_m2,processing_ms=elapsed))
    assert rows
    result=dict(schema='evidencescene.rgbd-mesh-validation.v1',version=__version__,camera_frames=len(rows),
        exact_registered_pixel_and_rgb_lineage=True,rows=rows,
        p50_ms=float(np.median([r['processing_ms'] for r in rows])),
        p95_ms=float(np.percentile([r['processing_ms'] for r in rows],95)),
        limits=['Recorded simulation RGB-D, not a physical-store benchmark.',
                'Per-camera surfaces only; no multi-camera fusion or pose ownership change.',
                'No ground-truth geometry was used; source consistency is not an absolute accuracy score.'])
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}))

if __name__=='__main__': main()
