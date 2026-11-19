import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from evidencescene.surface import (
    SurfaceMeshConfig, reconstruct_planar_surface, reconstruct_organized_surface,
)

META = dict(coordinate_frame='camera/local', timestamp_ns=1234, source_id='recording/head/42')


def samples(n=16):
    u, v = np.meshgrid(np.arange(n), np.arange(n))
    points = np.column_stack((u.ravel() * .008, v.ravel() * .008, np.ones(n*n)))
    pixels = np.column_stack((u.ravel(), v.ravel()))
    colors = np.random.default_rng(8).integers(0, 256, (n*n, 3), dtype=np.uint8)
    return points, pixels, colors


def planar(p, c, **kw):
    args = dict(plane_normal=[0, 0, 1], plane_offset_m=-1., plane_support_count=100,
                plane_rmse_m=.001, **META)
    args.update(kw)
    return reconstruct_planar_surface(p, c, **args)


def test_planar_mesh_retains_exact_samples_and_rgb():
    p, _, c = samples()
    out = planar(p, c)
    assert out.ready
    np.testing.assert_array_equal(out.vertices_m, p[out.source_indices])
    np.testing.assert_array_equal(out.colors_rgb_u8, c[out.source_indices])
    assert len(out.triangles) == 450
    assert out.surface_area_m2 == pytest.approx(.12**2)
    assert out.maximum_edge_m <= .024
    np.testing.assert_allclose(out.face_normals, np.tile([0,0,1], (len(out.triangles),1)))
    assert out.timestamp_ns == 1234 and out.coordinate_frame == 'camera/local'
    with pytest.raises(ValueError):
        out.vertices_m[0] = 3


@pytest.mark.parametrize('angles', [(0,90,0), (25,45,85), (180,0,0)])
def test_planes_are_not_restricted_to_horizontal_tables(angles):
    p, _, c = samples()
    R = Rotation.from_euler('xyz', angles, degrees=True).as_matrix()
    t = np.array([.3,-.7,.9]); p = p @ R.T + t
    normal = R @ np.array([0.,0.,1.])
    out = planar(p, c, plane_normal=normal, plane_offset_m=-1.-normal@t)
    assert out.ready and out.surface_area_m2 > .011
    np.testing.assert_array_equal(out.vertices_m, p[out.source_indices])
    np.testing.assert_allclose(out.face_normals @ normal, 1., atol=1e-12)


def test_missing_strip_and_object_outliers_are_not_filled():
    p, _, c = samples()
    p[80:96, 2] += .1
    p[160:176] = np.nan
    out = planar(p,c)
    assert out.ready
    assert not set(range(80,96)) & set(out.source_indices)
    assert not set(range(160,176)) & set(out.source_indices)
    tri = out.vertices_m[out.triangles]
    assert not np.any((tri[:,:,1].min(axis=1) < .04) & (tri[:,:,1].max(axis=1) > .04))
    assert not np.any((tri[:,:,1].min(axis=1) < .08) & (tri[:,:,1].max(axis=1) > .08))


def test_planar_normal_discontinuity_is_rejected():
    p, _, c = samples()
    p[:128,2] -= .004; p[128:,2] += .004
    out = planar(p,c)
    assert out.ready
    tri = out.vertices_m[out.triangles]
    assert np.max(np.ptp(tri[:,:,2],axis=1)) < .005


@pytest.mark.parametrize('evidence', [dict(plane_support_count=0),dict(plane_rmse_m=.1),
    dict(plane_support_count=float("nan")),dict(plane_support_count=True),
    dict(plane_rmse_m=-1),dict(plane_normal=[0,0,2]),dict(plane_offset_m=np.nan)])
def test_invalid_plane_fails_closed(evidence):
    p, _, c = samples(); out=planar(p,c,**evidence)
    assert not out.ready and not len(out.triangles)


def test_empty_input_and_large_extent_fail_closed():
    p,_,c=samples()
    assert not planar(p[:0],c[:0]).ready
    p[-1,0]=1e30
    out=planar(p,c)
    assert not out.ready and out.reason=='surface_grid_budget_exceeded'
    assert not planar(p[:-1],c[:-1],config=SurfaceMeshConfig(maximum_vertices=20)).ready


def test_organized_mesh_reconstructs_curved_surface_without_flattening():
    p,uv,c=samples()
    p[:,2] += .02 * np.sin(p[:,0] * 18)
    out=reconstruct_organized_surface(p,c,pixels_uv=uv,image_shape=(16,16),**META)
    assert out.ready and len(out.triangles)==450
    np.testing.assert_array_equal(out.vertices_m, p[out.source_indices])
    np.testing.assert_array_equal(out.colors_rgb_u8, c[out.source_indices])
    assert np.ptp(out.vertices_m[:,2]) > .015
    assert np.ptp(out.face_normals[:,0]) > .1


def test_organized_missing_pixels_and_depth_edges_are_not_bridged():
    p,uv,c=samples(); p[:128,2] += .1
    valid = uv[:,0] != 8
    out=reconstruct_organized_surface(p[valid],c[valid],pixels_uv=uv[valid],image_shape=(16,16),**META)
    assert out.ready
    tri = out.vertices_m[out.triangles]
    assert np.max(np.ptp(tri[:,:,2],axis=1)) == 0
    assert not np.any((tri[:,:,0].min(axis=1)<.064) & (tri[:,:,0].max(axis=1)>.064))


def test_organized_sampling_preserves_original_indices():
    p,uv,c=samples()
    out=reconstruct_organized_surface(p,c,pixels_uv=uv,image_shape=(16,16),pixel_step=2,
        config=SurfaceMeshConfig(minimum_support=4),**META)
    assert out.ready and len(out.triangles)==98
    assert np.all(uv[out.source_indices] % 2 == 0)
    np.testing.assert_array_equal(out.vertices_m,p[out.source_indices])


def test_organized_rejects_duplicate_pixels_and_invalid_metadata():
    p,uv,c=samples(); uv[-1]=uv[0]
    with pytest.raises(ValueError,match='duplicate'):
        reconstruct_organized_surface(p,c,pixels_uv=uv,image_shape=(16,16),**META)
    with pytest.raises(ValueError,match='coordinate_frame'):
        planar(p,c,coordinate_frame='')
    with pytest.raises(ValueError,match='uint8'):
        planar(p,c.astype(float))
    with pytest.raises(ValueError,match='positive'):
        SurfaceMeshConfig(maximum_edge_m=float('nan'))
    with pytest.raises(ValueError,match='positive integer'):
        SurfaceMeshConfig(maximum_vertices=True)


def test_organized_grid_budget_precedes_allocation():
    p,uv,c=samples()
    out=reconstruct_organized_surface(p,c,pixels_uv=uv,image_shape=(1000000,1000000),**META)
    assert not out.ready and out.reason=='surface_grid_budget_exceeded'


def test_invalid_input_does_not_corrupt_previous_mesh():
    p,uv,c=samples(); good=planar(p,c)
    copy=good.vertices_m.copy()
    bad=planar(p,c,plane_support_count=0)
    assert not bad.ready
    np.testing.assert_array_equal(good.vertices_m,copy)
    assert not np.shares_memory(good.vertices_m,p)


def test_rgbd_reuses_registered_depth_geometry_and_pixel_lineage():
    from evidencescene.surface import reconstruct_rgbd_surface
    depth = np.ones((24,32),dtype=np.float32)
    depth[8:12, :] = np.nan
    depth[:,16:] = .8
    color = np.random.default_rng(42).integers(0,256,(24,32,3),dtype=np.uint8)
    K = np.array([[200.,0,16],[0,200,12],[0,0,1]])
    out = reconstruct_rgbd_surface(depth,color,K,pixel_step=1,**META)
    assert out.ready
    v,u = np.divmod(out.source_indices,32)
    np.testing.assert_array_equal(out.colors_rgb_u8,color[v,u])
    np.testing.assert_allclose(out.vertices_m[:,2],depth[v,u])
    np.testing.assert_allclose(out.vertices_m[:,0],(u-16)*depth[v,u]/200)
    tri = out.vertices_m[out.triangles]
    assert np.max(np.ptp(tri[:,:,2],axis=1)) == 0
    assert not np.any((v[out.triangles].min(axis=1)<8)&(v[out.triangles].max(axis=1)>=12))


def test_rgbd_requires_registered_shapes_and_valid_calibration():
    from evidencescene.surface import reconstruct_rgbd_surface
    d=np.ones((12,12));c=np.zeros((12,12,3),dtype=np.uint8);K=np.eye(3)
    with pytest.raises(ValueError,match='intrinsics'):
        reconstruct_rgbd_surface(d,c,K*0,**META)
    with pytest.raises(ValueError,match='match'):
        reconstruct_rgbd_surface(d,c[:8],K,**META)
    with pytest.raises(ValueError,match='mask'):
        reconstruct_rgbd_surface(d,c,K,mask=np.ones(d.shape),**META)
    result=reconstruct_rgbd_surface(d*0,c,K,**META)
    assert not result.ready and not len(result.vertices_m)
