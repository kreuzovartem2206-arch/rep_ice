"""Export completed wide-area frames, provenance, coverage and a tiled review map."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window
from PIL import Image
from pyproj import Transformer
from shapely.geometry import box,mapping
from shapely.ops import transform

from ice_monitor.cli import save
from ice_monitor.object_map import repeated_locations
from ice_monitor.expanded_processing import signature
from build_arctic_map import build,rect

NAMES={'barents':'Баренцево море','kara':'Карское море','laptev':'Море Лаптевых',
       'east_siberian':'Восточно-Сибирское море','chukchi':'Чукотское море'}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def fc(fs):return {'type':'FeatureCollection','features':fs}
def copy(source,dest):
    dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)


def display(values,sensor):
    if sensor=='optical':
        valid=np.isfinite(values).all(axis=0)
        rgb=np.uint8(np.nan_to_num(np.clip(values/.65,0,1),nan=0)**(1/1.6)*255)
        return np.dstack([np.moveaxis(rgb,0,-1),np.uint8(valid)*255])
    signal=values[0];valid=np.isfinite(signal)&(signal>0)
    db=np.full_like(signal,-40);np.log10(signal,out=db,where=valid);db[valid]*=10
    gray=np.uint8(np.nan_to_num(np.clip((db+35)/35,0,1),nan=0)*255)
    return np.stack([gray,gray,gray,np.uint8(valid)*255],axis=-1)


def imagery(path,sensor,folder,out):
    """Small initial preview plus lossless display tiles on the native 10 m grid.

    The contrast stretch is only for viewing. Original reflectance / Sigma0 is
    never replaced by the eight-bit display image for quantitative analysis.
    """
    folder.mkdir(parents=True,exist_ok=True);tiles=[]
    with rasterio.open(path) as src:
        preview_spacing=src.width*src.res[0]/1000
        bands=[1,2,3] if sensor=='optical' else [1]
        preview=src.read(bands,out_shape=(len(bands),1000,1000),resampling=Resampling.average)
        Image.fromarray(display(preview,sensor)).save(folder/'preview.webp',quality=85)
        for row in range(0,src.height,1000):
            for col in range(0,src.width,1000):
                w=Window(col,row,min(1000,src.width-col),min(1000,src.height-row))
                image=folder/f'{row}-{col}.webp'
                if not image.exists():
                    Image.fromarray(display(src.read(bands,window=w),sensor)).save(image,lossless=True,method=1)
                tiles.append({'image':image.relative_to(out).as_posix(),'rect':rect(rasterio.windows.bounds(w,src.transform))})
    return {'image':(folder/'preview.webp').relative_to(out).as_posix(),'tiles':tiles,
            'preview_spacing_m':preview_spacing,'detail_spacing_m':10,'quantitative_use':False}


def quality_overlay(mask,optical,folder,out):
    with rasterio.open(mask) as src:values=src.read(1)
    invalid=values<0 if optical else values==0
    image=Image.fromarray(invalid.astype('uint8')).convert('P')
    image.putpalette([0,0,0,255,176,70]+[0]*762)
    image.info['transparency']=bytes([0,110]+[0]*254)
    path=folder/'quality.png';image.save(path)
    return path.relative_to(out).as_posix()


def export(root,overview,out):
    root,overview,out=map(Path,[root,overview,out]);out.mkdir(parents=True,exist_ok=True)
    plans=[read(root/'regions'/r/'plan.json') for r in NAMES]
    # Validate every planned frame BEFORE exporting a map labelled completed.
    for p in plans:
        for sensor in ['optical','sar']:
            for group in p[sensor]:
                status=(root/'floes/scenes'/p['id']/group['items'][0]['id']/'status.json' if sensor=='optical'
                        else root/'detected/scenes'/p['id']/(group['date']+'-status.json'))
                if not status.exists() or read(status).get('expanded_signature')!=signature(p,group,sensor):
                    raise ValueError(f'Incomplete frame: {p["id"]} {sensor} {group["date"]}')
    all_floes=[];targets=[];optical=[];radar=[];coverage=[];table=[];assets={}
    inverse=Transformer.from_crs(3576,4326,always_xy=True).transform
    for p in plans:
        rid=p['id'];n=int(p['size_m']/10);unions={s:np.zeros((n,n),bool) for s in ['optical','sar']}
        for group in p['optical']:
            identifier=group['items'][0]['id'];folder=root/'optical-c1'/rid/identifier
            detections=root/'floes/scenes'/rid/identifier;status=read(detections/'status.json')
            optical.append(status);all_floes.extend(read(detections/'candidates.geojson')['features'])
            with rasterio.open(detections/'labels.tif') as src:unions['optical']|=src.read(1)>=0
            for name in ['source.json','source-index.tif']:
                copy(folder/name,out/'evidence'/rid/identifier/name)
            for name in ['status.json','labels.tif','candidates.geojson']:
                copy(detections/name,out/'evidence'/rid/identifier/name)
            assets[rid+':'+identifier]=imagery(folder/'rgb.tif','optical',out/'images'/rid/identifier,out)
            assets[rid+':'+identifier]['quality']=quality_overlay(detections/'labels.tif',True,out/'images'/rid/identifier,out)
            print('export optical',rid,group['date'],flush=True)
        for group in p['sar']:
            date=group['date'];detections=root/'detected/scenes'/rid
            status=read(detections/(date+'-status.json'));radar.append(status)
            scene_targets=read(detections/(date+'.geojson'))['features']
            native_info=read(root/'native'/rid/(date+'.json'))
            origins={s['source_id']:s['sources'][0]['annotation'] for s in native_info['mosaic_sources']}
            status['source_url']=origins[status['source_id']]
            for f in scene_targets:f['properties']['source_url']=origins[f['properties']['source_id']]
            targets.extend(scene_targets)
            with rasterio.open(detections/(date+'-mask.tif')) as src:unions['sar']|=src.read(1)>0
            copy(detections/(date+'-mask.tif'),out/'masks/sar'/rid/(date+'.tif'))
            copy(root/'native'/rid/(date+'.json'),out/'sources/sar'/rid/(date+'.json'))
            copy(root/'native'/rid/(date+'-source-index.tif'),out/'sources/sar'/rid/(date+'-source-index.tif'))
            for item in group['items']:
                key='_'.join(item['id'].split('_')[:8])
                for part in (root/'native-parts'/rid/key).glob('*.json'):
                    copy(part,out/'sources/sar/parts'/rid/key/part.name)
                for metadata in (root/'source-cache'/key).glob('*'):
                    if metadata.is_file():copy(metadata,out/'sources/sar/annotations'/key/metadata.name)
            assets[rid+':'+status['source_id']]=imagery(root/'native'/rid/(date+'.tif'),'sar',out/'images'/rid/status['source_id'],out)
            assets[rid+':'+status['source_id']]['quality']=quality_overlay(detections/(date+'-mask.tif'),False,out/'images'/rid/status['source_id'],out)
            print('export SAR',rid,date,flush=True)
        complete=[f for f in all_floes if f['properties']['region']==rid and not f['properties']['truncated']]
        stats={'region':rid,'name':NAMES[rid],'nominal_area_km2':p['nominal_area_km2'],
               'marine_area_km2':p['marine_area_km2'],'candidate_observations':len(complete),
               'threshold_stable_observations':sum(f['properties']['threshold_stable'] for f in complete),
               'isolated_stable_observations':sum(f['properties']['isolated_stable_candidate'] for f in complete),
               'radar_target_observations':sum(f['properties']['region']==rid for f in targets),
               'optical_usable_union_km2':round(float(unions['optical'].sum())*.0001,4),
               'radar_qualified_union_km2':round(float(unions['sar'].sum())*.0001,4)}
        table.append(stats)
        from ice_monitor.expanded_processing import profile
        for sensor,array in unions.items():
            dest=out/'masks/union'/f'{rid}-{sensor}.tif';dest.parent.mkdir(parents=True,exist_ok=True)
            with rasterio.open(dest,'w',**profile(p,1,'uint8',0)) as dst:
                dst.write(array.astype('uint8'),1);dst.update_tags(not_simultaneous='true',meaning='usable at least once across selected dates')
        coverage.append({'type':'Feature','geometry':mapping(transform(inverse,box(*p['bounds']))),'properties':dict(stats,full_arctic_complete=False)})
    persistent=repeated_locations(targets)
    complete=[f for f in all_floes if not f['properties']['truncated']]
    stable=[f for f in complete if f['properties']['threshold_stable']]
    isolated=[f for f in complete if f['properties']['isolated_stable_candidate']]
    for name,fs in [('floe-candidates',complete),('threshold-stable-candidates',stable),('isolated-stable-candidates',isolated),
                    ('excluded-truncated',[f for f in all_floes if f['properties']['truncated']]),('radar-targets',targets),
                    ('persistent-radar-targets',persistent),('confirmed-stamukhi',[]),('processed-areas',coverage)]:
        save(out/(name+'.geojson'),fc(fs))
    summary={'status':'partial_experimental_object_map','full_arctic_complete':False,
             'nominal_processed_area_km2':sum(p['nominal_area_km2'] for p in plans),
             'marine_area_km2':sum(p['marine_area_km2'] for p in plans),
             'optical_frames':len(optical),'sar_frames':len(radar),'candidate_observations':len(complete),
             'threshold_stable_observations':len(stable),'isolated_stable_observations':len(isolated),
             'radar_target_observations':len(targets),'persistent_radar_targets':len(persistent),'confirmed_stamukhi':0,
             'minimum_equivalent_diameter_m':30,'completeness_at_30m_validated':False,'regions':table,
             'optical_usable_union_km2':round(sum(t['optical_usable_union_km2'] for t in table),4),
             'radar_qualified_union_km2':round(sum(t['radar_qualified_union_km2'] for t in table),4),
             'union_area_note':'Union of different dates; not simultaneous coverage.',
             'note':'Counts are observations, not unique tracked floes. No independent grounding evidence.'}
    save(out/'summary.json',summary);save(out/'map-assets.json',assets)
    save(out/'processing-plan.json',{'regions':plans})
    save(out/'optical-processing.json',{'frames':optical,'full_arctic_complete':False})
    save(out/'radar-processing.json',{'frames':radar,'target_observations':len(targets),'full_arctic_complete':False})
    build(out,overview,root/'native',out/'map.html')
    checksums={str(p.relative_to(out)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted(out.rglob('*')) if p.is_file() and p.name!='checksums.json'}
    save(out/'checksums.json',checksums)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['root','overview','output']:p.add_argument('--'+key,required=True)
    a=p.parse_args();export(a.root,a.overview,a.output)
