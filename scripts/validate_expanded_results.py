"""Check exported scientific data, coverage arithmetic and all local map tiles."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from pyproj import Transformer
from shapely.geometry import shape
from shapely.ops import transform


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def validate(root):
    root=Path(root);summary=read(root/'summary.json');plans=read(root/'processing-plan.json')['regions']
    forward=Transformer.from_crs(4326,3576,always_xy=True).transform
    candidates=read(root/'floe-candidates.geojson')['features'];targets=read(root/'radar-targets.geojson')['features']
    valid_sources={};bounds={}
    for p in plans:
        bounds[p['id']]=p['bounds']
        for sensor in ['optical','sar']:
            for group in p[sensor]:
                ids=[i['id'] if sensor=='optical' else '_'.join(i['id'].split('_')[:8]) for i in group['items']]
                valid_sources[(p['id'],ids[0])]=set(ids)
    identifiers=set()
    for f in candidates+targets:
        p=f['properties'];g=shape(f['geometry']);projected=transform(forward,g)
        assert g.is_valid and not g.is_empty,(p['id'],'geometry')
        assert p['id'] not in identifiers,(p['id'],'duplicate ID')
        identifiers.add(p['id'])
        assert p['grounding']=='unverified'
        assert '2025-12-01'<=p['observed_at']<'2026-03-01'
        allowed=valid_sources[(p['region'],p['frame_id'])]
        assert p['source_id'] in p['source_ids'] and set(p['source_ids'])<=allowed
        area=p.get('area_m2',p.get('radar_area_m2'))
        diameter=p.get('equivalent_diameter_m',p.get('radar_equivalent_diameter_m'))
        assert area>=math.pi*15**2 and diameter>=30
        assert abs(area-projected.area)<max(.2,area*1e-7),(p['id'],'area')
        assert abs(diameter-2*math.sqrt(area/math.pi))<.011
    assert len(candidates)==summary['candidate_observations']
    assert len(targets)==summary['radar_target_observations']
    assert all(not f['properties']['truncated'] for f in candidates)
    for name,key in [('threshold-stable-candidates','threshold_stable'),('isolated-stable-candidates','isolated_stable_candidate')]:
        expected={f['properties']['id'] for f in candidates if f['properties'][key]}
        assert {f['properties']['id'] for f in read(root/(name+'.geojson'))['features']}==expected
    assert not read(root/'confirmed-stamukhi.geojson')['features']
    assert summary['confirmed_stamukhi']==0 and not summary['full_arctic_complete']
    assert not summary['completeness_at_30m_validated']
    for p in plans:
        stats=next(s for s in summary['regions'] if s['region']==p['id'])
        for sensor,key in [('optical','optical_usable_union_km2'),('sar','radar_qualified_union_km2')]:
            with rasterio.open(root/'masks/union'/f'{p["id"]}-{sensor}.tif') as src:
                assert src.res==(10,10) and str(src.crs)=='EPSG:3576'
                assert list(src.bounds)==p['bounds']
                area=float(np.count_nonzero(src.read(1)))*.0001
            assert abs(area-stats[key])<.00011
    assets=read(root/'map-assets.json');tile_count=0
    for key,asset in assets.items():
        rid=key.split(':',1)[0];area=0
        assert (root/asset['image']).is_file()
        for tile in asset['tiles']:
            path=(root/tile['image']).resolve();assert path.is_relative_to(root.resolve())
            with Image.open(path) as image:
                image.verify();assert image.width*10==tile['rect'][2] and image.height*10==tile['rect'][3]
            area+=tile['rect'][2]*tile['rect'][3];tile_count+=1
        b=bounds[rid];assert area==(b[2]-b[0])*(b[3]-b[1])
    map_check=read(root/'map.validation.json')
    assert map_check['optical_objects']==len(candidates) and map_check['radar_objects']==len(targets)
    assert map_check['optical_frames']==summary['optical_frames'] and map_check['radar_frames']==summary['sar_frames']
    report={'validated_observations':len(candidates)+len(targets),'native_display_tiles':tile_count,
            'nominal_area_km2':summary['nominal_processed_area_km2'],'data_integrity_checks':'passed',
            'scientific_accuracy_independently_validated':False}
    (root/'export-checks.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root');validate(p.parse_args().root)
