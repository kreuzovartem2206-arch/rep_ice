import pytest
from ice_monitor.object_pipeline import tile_grid


def test_grid_plan_preserves_native_pixel_alignment_and_rejects_wrong_crs():
    region={'crs':'EPSG:3576','projected_center':[123.4,-456.7],'size_m':10000}
    bounds,affine,size=tile_grid(region)
    assert size==(1000,1000)
    assert bounds==[-4880,-5460,5120,4540]
    assert affine*(0,0)==(bounds[0],bounds[3])
    assert affine*(1000,1000)==(bounds[2],bounds[1])
    with pytest.raises(ValueError):tile_grid(dict(region,crs='EPSG:4326'))
    with pytest.raises(ValueError):tile_grid(dict(region,size_m=10001))
    with pytest.raises(ValueError):tile_grid(dict(region,size_m=0))
