import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine

from ice_monitor.processing import optical_mask, read_asset, vectorize, process_sar, process_s2


def test_negative_reflectance_does_not_make_ndsi_above_one_ice(config):
    green=np.full((30,30),.2);swir=np.full((30,30),-.06)
    ice,valid=optical_mask(green,swir,np.full((30,30),11),np.ones((30,30),bool),config)
    assert not valid.any() and not ice.any()


def test_conflicting_legacy_offset_is_rejected_before_read(config):
    asset={'raster:bands':[{'scale':.0001,'offset':-.1}]}
    item={'collection':'sentinel-2-l2a','properties':{'earthsearch:boa_offset_applied':True},
          'assets':{'green':asset,'swir16':asset}}
    with pytest.raises(ValueError,match='Conflicting legacy BOA'):
        process_s2(item,config,Affine.identity(),np.ones((30,30),bool))


@pytest.fixture
def config():
    return json.loads((Path(__file__).parents[1] / 'config/kara.json').read_text())


def test_ice_scl_is_not_cloud(config):
    green = np.full((30, 30), 0.6)
    swir = np.full((30, 30), 0.1)
    scl = np.full((30, 30), 11.)
    ice, valid = optical_mask(green, swir, scl, np.ones((30, 30), bool), config)
    assert valid.all() and ice.all()
    scl[:] = 9
    ice, valid = optical_mask(green, swir, scl, np.ones((30, 30), bool), config)
    assert not valid.any() and not ice.any()


def test_cloud_buffer_and_ocean_mask(config):
    scl = np.full((30, 30), 11.)
    scl[15, 15] = 9
    ocean = np.ones((30, 30), bool)
    ocean[:5] = False
    ice, valid = optical_mask(np.full((30, 30), .6), np.full((30, 30), .1), scl, ocean, config)
    assert not valid[15, 16]
    assert not ice[:5].any()
    assert valid[10, 10]


def test_30m_candidate_and_truncation(config):
    ice = np.zeros((30, 30), bool)
    ice[5:8, 5:8] = True
    ice[15:17, 15:17] = True
    transform = Affine(10, 0, 480000, 0, -10, 8100000)
    features = vectorize(ice, np.ones_like(ice), transform, config)
    assert len(features) == 1
    assert features[0]['properties']['area_m2'] == 900
    assert features[0]['properties']['confidence'] == 'low'
    assert features[0]['properties']['equivalent_diameter_m'] > 30
    ice[:4, :4] = True
    features = vectorize(ice, np.ones_like(ice), transform, config)
    assert any(f['properties']['truncated'] for f in features)


def test_scale_offset_nodata_and_nearest(config, tmp_path):
    affine = Affine(10, 0, 480000, 0, -10, 8100000)
    path = tmp_path / 'input.tif'
    values = np.full((10, 10), 6000, dtype='uint16')
    values[0, 0] = 0
    with rasterio.open(path, 'w', driver='GTiff', width=10, height=10, count=1,
                       dtype='uint16', nodata=0, transform=affine, crs=config['crs']) as dst:
        dst.write(values, 1)
    asset = {'href':str(path), 'raster:bands':[{'scale':.0001, 'offset':-.1}]}
    result = read_asset(asset, config, affine, (10, 10))
    assert np.isnan(result[0, 0])
    assert result[5, 5] == pytest.approx(.5)
    with pytest.raises(ValueError, match='scale'):
        read_asset({'href':str(path)}, config, affine, (10, 10))


def test_sar_db_input_is_rejected(config, tmp_path):
    affine = Affine(10, 0, 480000, 0, -10, 8100000)
    path = tmp_path / 'sar.tif'
    with rasterio.open(path, 'w', driver='GTiff', width=20, height=20, count=1,
                       dtype='float32', transform=affine, crs=config['crs']) as dst:
        dst.write(np.full((20, 20), -20, dtype='float32'), 1)
    with pytest.raises(ValueError, match='linear'):
        process_sar(path, config, affine, np.ones((20, 20), bool), 'IW', 'HH')


def test_positive_visual_dn_cannot_enter_sigma0_detector(config, tmp_path):
    affine = Affine(2000, 0, 480000, 0, -2000, 8100000)
    path = tmp_path / 'overview.tif'
    with rasterio.open(path, 'w', driver='GTiff', width=20, height=20, count=1,
                       dtype='uint8', transform=affine, crs=config['crs']) as dst:
        dst.write(np.full((20, 20), 128, dtype='uint8'), 1)
        dst.update_tags(product='uncalibrated_L1_visual_overview')
    with pytest.raises(ValueError, match='Visual overview'):
        process_sar(path, config, affine, np.ones((20, 20), bool), 'IW', 'HH')
