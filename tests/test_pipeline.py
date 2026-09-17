import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from ice_monitor.cli import configuration, main

CONFIG = str(Path(__file__).parents[1] / 'config/kara.json')


def test_idempotence_and_catalogue_failure(tmp_path):
    item = {'id':'scene-a', 'collection':'sentinel-2-l2a', 'properties':{'datetime':'2025-06-01T12:00:00Z'}}
    args = ['--config', CONFIG, '--output',str(tmp_path),'run',
            '--start','2025-06-01T00:00:00Z','--end','2025-06-02T00:00:00Z']
    def process(item, cfg, affine, mask):
        return [], mask.copy(), np.zeros_like(mask)
    with patch('ice_monitor.cli.s2_search', return_value=[item]), patch('ice_monitor.cli.process_s2', side_effect=process) as mock:
        assert main(args) == 0
        assert main(args) == 0
        assert mock.call_count == 1
    status = json.loads((tmp_path/'status.json').read_text())
    assert status['cached'] == 1 and status['stale']
    with patch('ice_monitor.cli.s2_search', side_effect=RuntimeError('network failure')):
        assert main(args) == 1
    status = json.loads((tmp_path/'status.json').read_text())
    assert status['status'] == 'error' and status['total_scenes'] == 1


def test_new_configuration_requires_new_state(tmp_path):
    args = ['--config',CONFIG,'--output',str(tmp_path/'run'),'run']
    with patch('ice_monitor.cli.s2_search', return_value=[]):
        assert main(args) == 0
    cfg = json.loads(Path(CONFIG).read_text())
    cfg['min_diameter_m'] = 60
    cfg['ocean_aoi'] = str(Path(CONFIG).parent/'kara_ocean.geojson')
    edited = tmp_path/'edited.json'
    edited.write_text(json.dumps(cfg))
    args[1] = str(edited)
    with pytest.raises(ValueError, match='Configuration changed'):
        main(args)
