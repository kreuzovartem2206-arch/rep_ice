import json
from pathlib import Path

from pyproj import Transformer
from shapely.geometry import box, mapping
from shapely.ops import transform

from ice_monitor.tracking import associate, track

CONFIG = json.loads((Path(__file__).parents[1] / 'config/kara.json').read_text())


def frame(day, x=480000, confidence='medium', stream='S2'):
    inverse = Transformer.from_crs(CONFIG['crs'], 4326, always_xy=True).transform
    geometry = mapping(transform(inverse, box(x, 8100000, x+150, 8100150)))
    return {'id':f'{day}-{x}', 'datetime':f'2025-06-{day:02d}T12:00:00Z', 'stream':stream,
            'valid_fraction':1, 'features':[{'type':'Feature','geometry':geometry,
            'properties':{'truncated':False,'confidence':confidence,'class':'ice_candidate'}}]}


def test_stationarity_is_not_grounding():
    features, _ = track([frame(1), frame(2), frame(3)], CONFIG)
    assert features[-1]['properties']['motion'] == 'stationary_candidate'
    assert features[-1]['properties']['grounding'] == 'unverified'


def test_small_unreliable_detections_do_not_become_stationary():
    features, _ = track([frame(n, confidence='low') for n in [1,2,3]], CONFIG)
    assert features[-1]['properties']['motion'] != 'stationary_candidate'


def test_ambiguity_does_not_invent_a_track():
    left = [box(0, 0, 100, 100)]
    right = [box(10, 0, 110, 100), box(200, 0, 300, 100)]
    matches, ambiguous, _ = associate(left, right, 3600, CONFIG)
    assert matches == {} and ambiguous == {0,1}


def test_large_gap_and_independent_streams():
    features, _ = track([frame(1), frame(10), frame(11, stream='S1:IW:HH:11:ascending')], CONFIG)
    assert len({f['properties']['track_id'] for f in features}) == 3


def test_clouds_do_not_report_loss_or_no_ice():
    cloudy = frame(2)
    cloudy['valid_fraction'] = 0
    cloudy['features'] = []
    features, events = track([frame(1), cloudy, frame(3)], CONFIG)
    assert len({f['properties']['track_id'] for f in features}) == 1
    assert all(e['type'] != 'lost' for e in events)


def test_motion_has_units_and_uncertainty():
    features, _ = track([frame(1), frame(2, x=480864)], CONFIG)
    props = features[-1]['properties']
    assert abs(props['speed_m_s'] - .01) < 1e-5
    assert props['motion'] == 'moving_candidate'
    assert props['speed_uncertainty_m_s'] > 0
