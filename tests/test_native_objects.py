import copy
import xml.etree.ElementTree as ET
import numpy as np
import pytest
from shapely.geometry import box, mapping
from shapely.ops import transform
from pyproj import Transformer

from ice_monitor.sar_native import calibrate,noise_power,interpolate_vectors
from ice_monitor.object_map import candidate_mask,repeated_locations


def lut(vector,value,values):
    return ET.fromstring('<root>'+''.join(f'<{vector}><line>{line}</line><pixel>0 10</pixel><{value}>{left} {right}</{value}></{vector}>' for line,left,right in values)+'</root>')


def test_calibration_uses_power_minus_noise_and_squared_sigma_lut():
    cal=lut('calibrationVector','sigmaNought',[(0,2,2),(10,2,2)])
    noise=lut('noiseVector','noiseLut',[(0,9,9),(10,9,9)])
    sigma,snr=calibrate(np.array([[5,0],[3,5]],dtype='uint16'),cal,noise,np.array([0,10]),np.array([0,10]))
    assert sigma[0,0]==pytest.approx(4)
    assert snr[0,0]==pytest.approx(10*np.log10(16/9))
    assert np.isnan(sigma[0,1]) and np.isnan(sigma[1,0])
    corrected,_=calibrate(np.full((2,2),5),cal,noise,np.array([0,10]),np.array([0,10]),True)
    assert np.allclose(corrected,6.25) # Never subtract noise twice.


def test_lut_interpolates_both_dimensions_at_native_coordinates():
    root=lut('calibrationVector','sigmaNought',[(0,2,4),(10,4,6)])
    assert interpolate_vectors(root,'calibrationVector','sigmaNought',np.array([5]),np.array([5]))[0,0]==4
    with pytest.raises(ValueError,match='source rows'):
        interpolate_vectors(root,'calibrationVector','sigmaNought',np.array([11]),np.array([5]))


def test_modern_noise_multiplies_azimuth_and_masks_unannotated_pixels():
    root=lut('noiseRangeVector','noiseRangeLut',[(0,2,2),(10,2,2)])
    root.append(ET.fromstring('<noiseAzimuthVector><firstAzimuthLine>0</firstAzimuthLine><lastAzimuthLine>10</lastAzimuthLine><firstRangeSample>0</firstRangeSample><lastRangeSample>5</lastRangeSample><line>0 10</line><noiseAzimuthLut>1 3</noiseAzimuthLut></noiseAzimuthVector>'))
    values=noise_power(root,np.array([5]),np.array([0,10]))
    assert values[0,0]==4 and np.isnan(values[0,1])


def test_dual_pol_screen_rejects_single_channel_speckle_and_below_30m():
    co=np.ones((150,150),dtype='float32');cross=co.copy();snr=np.full_like(co,20)
    co[50:53,50:53]=10;cross[50:53,50:53]=10 # 900 m2 retained
    co[80:82,80:82]=10;cross[80:82,80:82]=10 # 400 m2 excluded
    co[100:103,100:103]=10 # Not present in cross-pol
    labels,valid,*_=candidate_mask(co,cross,snr,snr,np.ones_like(co,dtype=bool))
    assert np.count_nonzero(labels)==9
    assert np.all(labels[50:53,50:53]>0)
    assert not labels[80:82,80:82].any() and not labels[100:103,100:103].any()
    low=snr.copy();low[50:53,50:53]=0
    labels,*_=candidate_mask(co,cross,snr,low,np.ones_like(co,dtype=bool))
    assert not labels.any()


def frame(identifier,day,x=0):
    inverse=Transformer.from_crs(3576,4326,always_xy=True).transform
    return {'type':'Feature','geometry':mapping(transform(inverse,box(x,-2000000,x+50,-1999950))),
            'properties':{'id':identifier,'stream':'same-orbit','observed_at':f'2026-01-{day:02d}T00:00:00Z','grounding':'unverified'}}


def test_repeated_positions_never_confirm_grounding():
    features=[frame('a',1),frame('b',13),frame('c',25)]
    result=repeated_locations(features)
    assert len(result)==1
    assert result[0]['properties']['grounding']=='unverified'
    assert not result[0]['properties']['confirmed_stamukha']


def test_ambiguous_and_moving_targets_do_not_become_persistent():
    assert not repeated_locations([frame('a',1),frame('b',13,200),frame('c',25,400)])
    features=[frame('a',1),frame('b',13),frame('b2',13,10),frame('c',25)]
    assert not repeated_locations(features)
