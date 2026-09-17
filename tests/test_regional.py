import pytest
from shapely.geometry import box, mapping, Point, shape
from pyproj import Transformer

from ice_monitor.regional import acquisition_key, project_footprint, domain_geometry, geographic_geometry, in_interval


def test_exclusive_end_handles_fractional_seconds():
    start, end = "2026-01-01T00:00:00Z", "2026-02-01T00:00:00Z"
    assert in_interval("2026-01-01T00:00:00.000Z", start, end)
    assert not in_interval("2026-02-01T00:00:00.123Z", start, end)
    assert not in_interval(end, start, end)


def test_geojson_dateline_is_cut_without_world_spanning_ring():
    geom=shape({'type':'Polygon','coordinates':[[[179,70],[-179,70],[-179,72],[179,72],[179,70]]]})
    result=shape(geographic_geometry(geom))
    assert result.is_valid
    assert result.area == pytest.approx(4)
    assert len(result.geoms) == 2
    assert all(p.bounds[2]-p.bounds[0] <= 1 for p in result.geoms)


def test_dateline_scene_stays_small_in_polar_projection():
    geometry={'type':'Polygon','coordinates':[[[179,70],[-179,70],[-179,72],[179,72],[179,70]]]}
    projected=project_footprint(geometry,'EPSG:3576')
    assert projected.is_valid
    assert 10000 < projected.area/1e6 < 30000
    transform=Transformer.from_crs(4326,3576,always_xy=True)
    assert projected.contains(Point(*transform.transform(180,71)))
    assert not projected.contains(Point(*transform.transform(0,71)))


def test_packaging_does_not_duplicate_acquisition():
    a={'id':'S1A_EW_GRDM_1SDH_20260101T000000_20260101T000100_123456_ABCDEF_1234_COG'}
    b={'id':'S1A_EW_GRDM_1SDH_20260101T000000_20260101T000100_123456_ABCDEF_9876'}
    assert acquisition_key(a)==acquisition_key(b)


def test_overview_domain_excludes_source_land():
    cfg={'crs':'EPSG:3576','query_bboxes':[[40,70,50,75]]}
    land={'type':'FeatureCollection','features':[{'type':'Feature','geometry':mapping(box(40,70,45,75))}]}
    ocean,projected_land=domain_geometry(cfg,land)
    transform=Transformer.from_crs(4326,3576,always_xy=True)
    assert not ocean.contains(Point(*transform.transform(42,72)))
    assert ocean.contains(Point(*transform.transform(48,72)))
    assert ocean.intersection(projected_land).area < 1
