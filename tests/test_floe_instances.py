import json
import math
import numpy as np
import rasterio
from affine import Affine
from shapely.geometry import box,shape
from scipy import ndimage
from ice_monitor.floe_instances import split_instances,segment
from ice_monitor.floe_instances import label_iou,all_label_ious


def test_bulk_iou_retains_global_variant_area_and_matches_reference():
    reference=np.zeros((20,20),dtype='int32');variant=np.zeros_like(reference)
    reference[2:5,2:5]=1;reference[12:16,12:16]=2
    variant[:10,:10]=1;variant[12:15,12:15]=3
    result=all_label_ious(reference,variant)
    assert result[1]==.09
    assert result[2]==9/16
    assert all(result[i]==label_iou(reference,variant,i) for i in [1,2])


def test_watershed_separates_neck_but_keeps_small_isolated_component():
    y,x=np.mgrid[:120,:120]
    ice=((x-35)**2+(y-60)**2<20**2)|((x-80)**2+(y-60)**2<20**2)
    ice[59:62,35:81]=True
    ice[10:13,10:13]=True
    labels=split_instances(ice)
    assert labels[60,35] != labels[60,80]
    assert labels[11,11]>0
    assert not labels[~ice].any()
    for n in np.unique(labels)[1:]:
        assert ndimage.label(labels==n)[1]==1


def test_export_has_no_subthreshold_islands_and_flags_invalid_edges(tmp_path):
    src_folder=tmp_path/'input';src_folder.mkdir()
    item={'id':'synthetic','collection':'sentinel-2-c1-l2a','properties':{'datetime':'2026-02-20T00:00:00Z','view:sun_elevation':8}}
    (src_folder/'source.json').write_text(json.dumps(item))
    (src_folder/'status.json').write_text(json.dumps({'bounds':[0,-2000000,1000,-1999000]}))
    green=np.full((100,100),.05,dtype='float32')
    swir=np.full_like(green,.04);scl=np.full_like(green,6)
    for sl in [np.s_[45:48,45:48],np.s_[10:12,10:12],np.s_[:10,60:70]]:
        green[sl]=.5;swir[sl]=.05;scl[sl]=11
    affine=Affine(10,0,0,0,-10,-1999000)
    with rasterio.open(src_folder/'reflectance.tif','w',driver='GTiff',width=100,height=100,count=3,dtype='float32',crs='EPSG:3576',transform=affine) as dst:
        dst.write(np.stack([green,swir,scl]))
    features,status=segment(src_folder,'test',box(0,-2000000,1000,-1999000),tmp_path/'output')
    assert len(features)==2
    assert all(f['properties']['area_m2']>=math.pi*15**2 for f in features)
    assert sum(f['properties']['truncated'] for f in features)==1
    assert all(not f['properties']['confirmed_stamukha'] for f in features)
    assert all(shape(f['geometry']).is_valid for f in features)
