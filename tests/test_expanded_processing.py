import numpy as np
import rasterio
import pytest
from affine import Affine
from pyproj import Transformer
from shapely.geometry import box,mapping
from shapely.ops import transform

from ice_monitor.expanded_processing import windows,compose_sar,source_properties,profile


@pytest.mark.parametrize('change',[{'crs':'EPSG:4326'},{'size_m':10000}])
def test_grid_rejects_false_crs_or_area(change):
    plan=dict(crs='EPSG:3576',size_m=50000,bounds=[0,0,50000,50000])
    plan.update(change)
    with pytest.raises(ValueError):profile(plan,1)


def test_windows_cover_nonmultiple_extent_once():
    count=np.zeros((53,53),'uint8')
    for w in windows(53,20):
        count[int(w.row_off):int(w.row_off+w.height),int(w.col_off):int(w.col_off+w.width)]+=1
    assert np.all(count==1)


def test_sar_mosaic_preserves_independent_channel_validity():
    original=np.ones((4,5,5),'float32');original[2,1:4,1:4]=np.nan
    bands=np.full_like(original,np.nan);index=np.zeros((5,5),'uint16')
    compose_sar(bands,index,original,1)
    np.testing.assert_equal(bands,original)
    compose_sar(bands,index,original*2,2)
    np.testing.assert_equal(bands,original)
    assert np.all(index==1)


def test_object_source_is_pixel_majority_not_mosaic_primary(tmp_path):
    path=tmp_path/'index.tif';affine=Affine(10,0,-500000,0,-10,-1700000)
    index=np.full((10,10),2,'uint16');index[:,:3]=1
    with rasterio.open(path,'w',driver='GTiff',width=10,height=10,count=1,dtype='uint16',
                       crs='EPSG:3576',transform=affine,nodata=0) as dst:dst.write(index,1)
    inverse=Transformer.from_crs(3576,4326,always_xy=True).transform
    polygon=box(-500000,-1700100,-499900,-1700000)
    feature={'type':'Feature','geometry':mapping(transform(inverse,polygon)),'properties':{}}
    items=[{'id':'first','datetime':'2026-02-20T07:00:00Z','sun':15},
           {'id':'second','datetime':'2026-02-20T07:00:02Z','sun':16}]
    p=source_properties([feature],path,items,'first','optical')[0]['properties']
    assert p['source_id']=='second'
    assert p['frame_id']=='first'
    assert p['source_ids']==['second','first']
    assert p['observed_at']==items[1]['datetime']
    assert p['multi_scene_boundary'] is True
