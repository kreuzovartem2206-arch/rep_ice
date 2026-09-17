import json
from pathlib import Path
from unittest.mock import patch

import pytest

from ice_monitor.sar_worker import product_parameters, execute


def product():
    attrs = {'processingLevel':'LEVEL1','operationalMode':'IW','polarisationChannels':'VV&VH',
             'relativeOrbitNumber':93,'orbitDirection':'DESCENDING'}
    return {'Name':'S1A_IW_GRDH_1SDV_TEST.SAFE', 'ContentDate':{'Start':'2025-06-01T00:00:00Z'},
            'Attributes':[{'Name':k,'Value':v} for k,v in attrs.items()]}


def test_metadata_polarization_and_level():
    assert product_parameters(product())['polarization'] == 'VH'
    bad = product()
    bad['Attributes'][0]['Value'] = 'LEVEL2'
    with pytest.raises(ValueError, match='L1 GRD'):
        product_parameters(bad)


def test_missing_snap_produces_explicit_error(tmp_path):
    with patch('ice_monitor.sar_worker.shutil.which', return_value=None):
        assert execute({'fingerprint':'test'}, None, tmp_path, 'a', 'b', 'missing-gpt', 'missing.xml') == 1
    report = json.loads((tmp_path/'sar-status.json').read_text())
    assert report['status'] == 'error'
    assert 'gpt not found' in report['error']
