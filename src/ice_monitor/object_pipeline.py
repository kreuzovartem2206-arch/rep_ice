"""Reproducible bounded L1/L2 object screening over an explicit tile plan."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from pyproj import Transformer
from rasterio.features import geometry_mask
from shapely.geometry import box, shape, mapping

from .catalog import s2_search
from .cli import save
from .processing import read_asset, optical_mask, vectorize
from .sar_native import read_scene


def tile_grid(region):
    if region.get('crs')!='EPSG:3576':raise ValueError('Plan must use EPSG:3576')
    if region['size_m']<=0 or region['size_m']%20:raise ValueError('Tile side must be positive and divisible by 20 m')
    x,y=region['projected_center'];x,y=round(x/10)*10,round(y/10)*10
    half=region['size_m']/2;n=round(region['size_m']/10)
    if n>3000:raise ValueError('Split regions larger than 30 km')
    return [x-half,y-half,x+half,y+half],Affine(10,0,x-half,0,-10,y+half),(n,n)


def optical_region(region, root, ocean, start, end, limit=3):
    root=Path(root);bounds,affine,size=tile_grid(region)
    inverse=Transformer.from_crs(3576,4326,always_xy=True)
    corners=[inverse.transform(x,y) for x in [bounds[0],bounds[2]] for y in [bounds[1],bounds[3]]]
    lons,lats=zip(*corners)
    if max(lons)-min(lons)>180:raise ValueError('Split antimeridian optical tile')
    bbox=[min(lons),min(lats),max(lons),max(lats)]
    cache=root/'optical-catalogue'/(region['id']+'.json')
    query={'collection':'sentinel-2-c1-l2a','bbox':bbox,'start':start,'end':end}
    prior=json.loads(cache.read_text()) if cache.exists() else {}
    if prior.get('query')==query:items=prior['items']
    else:
        items=s2_search(bbox,start,end);save(cache,{'query':query,'items':items})
    # Rank actual local cloud/NoData quality, not whole-granule cloud percentage.
    scored=[];preview_size=(size[0]//10,size[1]//10)
    for item in items:
        score_path=root/'optical-preflight'/region['id']/(item['id']+'.json')
        sig=hashlib.sha256(json.dumps([bounds,item['id'],'scl-100m-v1']).encode()).hexdigest()
        stored=json.loads(score_path.read_text()) if score_path.exists() else {}
        if stored.get('signature')==sig:score=stored['valid_fraction']
        else:
            scl=read_asset(item['assets']['scl'],{'crs':'EPSG:3576'},affine*Affine.scale(10),preview_size,True)
            good=np.isfinite(scl)&~np.isin(scl,[0,1,2,3,8,9,10])
            score=float(good.mean());save(score_path,{'signature':sig,'source_id':item['id'],'valid_fraction':score})
        scored.append((score,item))
    selected=[];days=set()
    for score,item in sorted(scored,key=lambda x:(x[0],x[1]['properties']['datetime']),reverse=True):
        day=item['properties']['datetime'][:10]
        if day in days:continue
        selected.append((score,item));days.add(day)
        if len(selected)>=limit:break
    ocean_mask=geometry_mask([mapping(ocean)],size,affine,invert=True)
    cfg={'crs':'EPSG:3576','resolution_m':10,'cloud_buffer_m':60,'ndsi_threshold':.4,'green_min':.12,
         'min_diameter_m':30,'reliable_diameter_m':100}
    run=[]
    for preview_score,item in selected:
        folder=root/'optical-c1'/region['id']/item['id'];status_path=folder/'status.json'
        # New runs have a configuration signature, so stale products cannot silently
        # substitute for a changed coast mask or threshold configuration.
        signature=hashlib.sha256((json.dumps([bounds,cfg,item['id']])+ocean_mask.tobytes().hex()).encode()).hexdigest()
        if status_path.exists() and json.loads(status_path.read_text()).get('signature')==signature:
            run.append(json.loads(status_path.read_text()));continue
        green=read_asset(item['assets']['green'],cfg,affine,size)
        swir=read_asset(item['assets']['swir16'],cfg,affine,size)
        scl=read_asset(item['assets']['scl'],cfg,affine,size,True)
        ice,valid=optical_mask(green,swir,scl,ocean_mask,cfg)
        features=vectorize(ice,valid,affine,cfg)
        for f in features:
            f['properties'].update(source_id=item['id'],observed_at=item['properties']['datetime'],region=region['id'],
                                   source_level='L2A',source_collection=item['collection'],review_status='unreviewed')
        save(folder/'source.json',item);save(folder/'observations.geojson',{'type':'FeatureCollection','features':features})
        with rasterio.open(folder/'reflectance.tif','w',driver='GTiff',width=size[1],height=size[0],count=3,
                           dtype='float32',crs=cfg['crs'],transform=affine,nodata=np.nan,compress='deflate',tiled=True) as dst:
            for band,array in enumerate([green,swir,scl],1):dst.write(array,band)
            dst.update_tags(product='C1_L2A_surface_reflectance_scale_offset_applied_once')
        with rasterio.open(folder/'mask.tif','w',driver='GTiff',width=size[1],height=size[0],count=1,
                           dtype='uint8',crs=cfg['crs'],transform=affine,nodata=255,compress='deflate',tiled=True) as dst:
            dst.write(np.where(valid,ice.astype('uint8'),255),1)
        status={'signature':signature,'region':region['id'],'datetime':item['properties']['datetime'],
                'source_id':item['id'],'source_collection':item['collection'],'valid_fraction':float(valid.mean()),
                'bounds':bounds,'candidate_count':len(features),'preview_valid_fraction':preview_score}
        save(status_path,status);run.append(status)
        print('optical',region['id'],item['id'],round(valid.mean(),3),'valid',flush=True)
    save(root/'selected-optical'/(region['id']+'.json'),{'selected':run,'catalogue_scenes':len(items)})
    return run


def rgb_evidence(root, selected):
    for frame in selected:
        folder=Path(root)/'optical-c1'/frame['region']/frame['source_id']
        item=json.loads((folder/'source.json').read_text())
        with rasterio.open(folder/'reflectance.tif') as src:
            green=src.read(1);affine=src.transform;profile=src.profile
        target=folder/'rgb.tif'
        if target.exists():
            with rasterio.open(target) as prior:
                if prior.tags().get('signature')==frame['signature']:continue
        red=read_asset(item['assets']['red'],{'crs':'EPSG:3576'},affine,green.shape)
        blue=read_asset(item['assets']['blue'],{'crs':'EPSG:3576'},affine,green.shape)
        with rasterio.open(target,'w',**profile) as dst:
            dst.write(np.stack([red,green,blue]));dst.update_tags(signature=frame['signature'])
            for i,name in enumerate(['red_surface_reflectance','green_surface_reflectance','blue_surface_reflectance'],1):
                dst.set_band_description(i,name)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',required=True);p.add_argument('--overview',required=True)
    p.add_argument('--output',required=True);p.add_argument('--sensor',choices=['sar','optical','all'],default='all')
    p.add_argument('--region',default='all')
    p.add_argument('--rgb',action='store_true',help='Read native C1 red/blue for review images')
    args=p.parse_args();plan=json.loads(Path(args.plan).read_text());root=Path(args.output)
    ocean=shape(json.loads((Path(args.overview)/'projected.json').read_text())['ocean']).buffer(-1000)
    for region in plan['regions']:
        if args.region!='all' and region['id'] not in args.region.split(','):continue
        bounds,_,_=tile_grid(region)
        if not ocean.contains(box(*bounds)):raise ValueError('Tile is outside conservative offshore domain')
        if args.sensor in ['sar','all']:
            for scene in region['scenes']:
                date=scene['properties']['datetime'][:10]
                read_scene(scene['id'],bounds,root/'native'/region['id']/(date+'.tif'),root/'source-cache')
        if args.sensor in ['optical','all']:
            selected=optical_region(region,root,ocean,plan.get('optical_start','2026-02-15T00:00:00Z'),
                                    plan.get('optical_end','2026-03-01T00:00:00Z'))
            if args.rgb:rgb_evidence(root,selected)


if __name__=='__main__':main()
