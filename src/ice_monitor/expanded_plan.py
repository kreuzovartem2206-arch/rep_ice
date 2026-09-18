"""Plan wider winter regions using same-pass mosaics and explicit marine masks."""
import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
from affine import Affine
from pyproj import Transformer
from rasterio.features import geometry_mask
from shapely.geometry import box,shape,mapping
from shapely.ops import transform,unary_union

from .catalog import s2_search
from .processing import read_asset
from .cli import save


def plan_region(old, side, overview, output, iw_items, footprints):
    rid=old['id'];folder=Path(output)/'regions'/rid;folder.mkdir(parents=True,exist_ok=True)
    cx,cy=[round(v/10)*10 for v in old['projected_center']];half=side/2
    bounds=[cx-half,cy-half,cx+half,cy+half];extent=box(*bounds)
    ocean=shape(overview['ocean']).buffer(-1000);marine=extent.intersection(ocean)
    inverse=Transformer.from_crs(3576,4326,always_xy=True)
    ll=[inverse.transform(x,y) for x in [bounds[0],bounds[2]] for y in [bounds[1],bounds[3]]]
    lons,lats=zip(*ll)
    if max(lons)-min(lons)>180:raise ValueError('Split antimeridian region')
    # A wider region can sit across the edge of the old IW swath. Re-rank all
    # repeat-orbit series instead of pretending adjacent along-track slices fill it.
    passes=defaultdict(dict)
    for item in iw_items:
        p=item['properties'];key='_'.join(item['id'].split('_')[:8])
        if not footprints[item['id']].intersects(marine):continue
        passes[(p['platform'],p['sat:absolute_orbit'])][key]=item
    repeats=defaultdict(list)
    for (platform,orbit),products in passes.items():
        found=sorted(products.values(),key=lambda i:i['properties']['datetime'])
        covered=unary_union([footprints[i['id']] for i in found]).intersection(marine).area/1e6
        repeats[(platform,orbit%175)].append({'date':found[0]['properties']['datetime'][:10],'items':found,
                                            'catalogue_covered_marine_km2':covered,'absolute_orbit':orbit})
    choices=[]
    for key,series in repeats.items():
        series.sort(key=lambda g:g['date'])
        if len(series)<3:continue
        selected=[series[i] for i in np.unique(np.linspace(0,len(series)-1,min(4,len(series))).round().astype(int))]
        from datetime import date
        if (date.fromisoformat(selected[-1]['date'])-date.fromisoformat(selected[0]['date'])).days<24:continue
        score=(min(s['catalogue_covered_marine_km2'] for s in selected),sum(s['catalogue_covered_marine_km2'] for s in selected)/len(selected),len(series))
        choices.append((score,selected))
    if not choices:raise ValueError('No repeat IW series spanning at least 24 days')
    sar=max(choices,key=lambda pair:pair[0])[1]
    query={'bbox':[min(lons),min(lats),max(lons),max(lats)],'start':'2026-02-15T00:00:00Z','end':'2026-03-01T00:00:00Z'}
    cache=folder/'optical-catalogue.json'
    previous=json.loads(cache.read_text(encoding='utf-8')) if cache.exists() else {}
    if previous.get('query')==query:items=previous['items']
    else:
        items=s2_search(query['bbox'],query['start'],query['end'],max_items=1500)
        save(cache,{'query':query,'items':items})
    size=(int(side/100),)*2;affine=Affine(100,0,bounds[0],0,-100,bounds[3])
    ocean_preview=geometry_mask([mapping(marine)],size,affine,invert=True)
    groups=defaultdict(list);masks={}
    for item in items:
        key=item['properties']['s2:datatake_id']
        groups[key].append(item)
        target=folder/'preflight'/(item['id']+'.npz');target.parent.mkdir(exist_ok=True)
        if target.exists():
            with np.load(target) as cached:
                if cached['bounds'].tolist()!=bounds:raise ValueError('Preflight belongs to another extent')
                good=cached['good']
        else:
            scl=read_asset(item['assets']['scl'],{'crs':'EPSG:3576'},affine,size,True)
            good=np.isfinite(scl)&~np.isin(scl,[0,1,2,3,8,9,10])&ocean_preview
            np.savez_compressed(target,good=good,bounds=bounds)
        masks[item['id']]=good
    scored=[]
    for datatake,group in groups.items():
        coverage=np.zeros(size,bool)
        ordered=sorted(group,key=lambda item:int(masks[item['id']].sum()),reverse=True)
        useful=[]
        for item in ordered:
            good=masks[item['id']]
            if not (good&~coverage).any():continue
            useful.append(item);coverage|=good
        if not useful:continue
        score=float(coverage.sum()/max(1,ocean_preview.sum()))
        scored.append({'datatake':datatake,'date':useful[0]['properties']['datetime'][:10],
                       'preview_valid_fraction':score,'items':useful})
    selected=[];days=set()
    for group in sorted(scored,key=lambda s:(s['preview_valid_fraction'],s['date']),reverse=True):
        if group['date'] in days:continue
        selected.append(group);days.add(group['date'])
        if len(selected)==3:break
    plan={'id':rid,'crs':'EPSG:3576','size_m':side,'projected_center':[cx,cy],'center':old['center'],
          'bounds':bounds,'nominal_area_km2':side**2/1e6,'marine_area_km2':marine.area/1e6,
          'sar':sar,'optical':selected,'shoreline':'Natural Earth 1:50m with 1 km inward marine buffer; not a high-resolution coastal mask',
          'full_arctic_complete':False}
    save(folder/'plan.json',plan)
    print(rid,'marine',round(plan['marine_area_km2'],2),'km2; SAR products',[len(s['items']) for s in sar],
          'optical',[(s['date'],round(s['preview_valid_fraction'],3),len(s['items'])) for s in selected],flush=True)
    return plan


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['base-plan','overview','output']:p.add_argument('--'+name,required=True)
    p.add_argument('--side-km',type=int,default=50);p.add_argument('--regions',default='all')
    args=p.parse_args();side=args.side_km*1000
    if side<10000 or side>100000 or side%10000:raise ValueError('Use 10–100 km in 10 km increments')
    old=json.loads(Path(args.base_plan).read_text(encoding='utf-8'));overview=json.loads((Path(args.overview)/'projected.json').read_text(encoding='utf-8'))
    footprints={i:shape(g) for i,g in zip(overview['ids'],overview['footprints'])};items={}
    for file in (Path(args.overview)/'catalogue').glob('*.json'):
        for item in json.loads(file.read_text(encoding='utf-8'))['items']:
            if item['properties']['sar:instrument_mode']=='IW':items[item['id']]=item
    forward=Transformer.from_crs(4326,3576,always_xy=True).transform
    for item in items.values():
        if item['id'] not in footprints:footprints[item['id']]=transform(forward,shape(item['geometry'])).buffer(0)
    results=[]
    for region in old['regions']:
        if args.regions!='all' and region['id'] not in args.regions.split(','):continue
        results.append(plan_region(region,side,overview,args.output,list(items.values()),footprints))
    save(Path(args.output)/('planning-'+args.regions.replace(',','-')+'.json'),{'regions':results,'full_arctic_complete':False})


if __name__=='__main__':main()
