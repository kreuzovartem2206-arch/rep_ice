"""Blockwise L1/L2 mosaics, full-region detection, and per-object provenance."""
import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from rasterio.windows import Window
from rasterio.features import geometry_mask
from pyproj import Transformer
from shapely.geometry import shape,box,mapping
from shapely.ops import transform

from .cli import save
from .processing import read_asset
from .sar_native import read_scene
from .floe_instances import segment
from .object_map import detect

# Keep GDAL worker threads alive across granules. Repeatedly tearing down and
# recreating GDAL/TLS workers can hang thread startup on Windows.
ASSET_POOL=ThreadPoolExecutor(max_workers=5,thread_name_prefix='cog-reader')

def windows(size, step=2000):
    for row in range(0,size,step):
        for col in range(0,size,step):
            yield Window(col,row,min(step,size-col),min(step,size-row))


def profile(plan,count,dtype='float32',nodata=np.nan):
    n=int(plan['size_m']/10);bounds=plan['bounds']
    if plan['crs']!='EPSG:3576' or not np.isfinite(bounds).all():
        raise ValueError('Expected finite bounds in EPSG:3576')
    if n*10!=plan['size_m'] or bounds[2]-bounds[0]!=n*10 or bounds[3]-bounds[1]!=n*10:
        raise ValueError('Bounds and requested 10 m grid extent disagree')
    return dict(driver='GTiff',width=n,height=n,count=count,dtype=dtype,crs='EPSG:3576',
                transform=Affine(10,0,bounds[0],0,-10,bounds[3]),nodata=nodata,
                compress='deflate',tiled=True,BIGTIFF='IF_SAFER')


def signature(plan,group,sensor):
    version='expanded-sar-v2' if sensor=='sar' else 'expanded-v1'
    return hashlib.sha256(json.dumps([plan['bounds'],[i['id'] for i in group['items']],sensor,version]).encode()).hexdigest()


def compose_sar(bands,indices,values,number):
    # Preserve valid co-pol even when cross-pol is below noise. Dropping both
    # would bias the independent co-pol background and its coverage fraction.
    take=np.isfinite(values[0])&(indices==0)
    bands[:,take]=values[:,take];indices[take]=number


def source_properties(features,index_path,items,frame_id,sensor):
    forward=Transformer.from_crs(4326,3576,always_xy=True).transform
    with rasterio.open(index_path) as src:
        for f in features:
            polygon=transform(forward,shape(f['geometry']));x0,y0,x1,y1=polygon.bounds
            left,top=(~src.transform)*(x0,y1);right,bottom=(~src.transform)*(x1,y0)
            left=max(0,int(np.floor(left)));top=max(0,int(np.floor(top)))
            right=min(src.width,int(np.ceil(right)));bottom=min(src.height,int(np.ceil(bottom)))
            if right<=left or bottom<=top:raise ValueError('Object outside source index')
            window=Window(left,top,right-left,bottom-top);indexes=src.read(1,window=window)
            within=geometry_mask([mapping(polygon)],indexes.shape,src.window_transform(window),invert=True)
            values,counts=np.unique(indexes[within],return_counts=True);good=values>0
            values,counts=values[good],counts[good]
            if not len(values):raise ValueError('Missing per-object provenance')
            selected=[items[int(i)-1] for i in values[np.argsort(-counts)]]
            p=f['properties'];primary=selected[0]
            p.update(frame_id=frame_id,source_id=primary['id'],source_ids=[s['id'] for s in selected],
                     observed_at=primary['datetime'],multi_scene_boundary=len(selected)>1,
                     source_attribution='majority of object pixels; all intersecting input scenes retained')
            if sensor=='optical':
                p['sun_elevation_deg']=primary['sun'];p['low_sun']=primary['sun']<20
    return features


def optical(plan, group, root, ocean):
    root=Path(root);rid=plan['id'];items=group['items'];frame_id=items[0]['id']
    folder=root/'optical-c1'/rid/frame_id;folder.mkdir(parents=True,exist_ok=True)
    output=root/'floes/scenes'/rid/frame_id;status_file=output/'status.json'
    sig=signature(plan,group,'optical')
    if status_file.exists() and json.loads(status_file.read_text(encoding='utf-8')).get('expanded_signature')==sig:
        print('cached optical',rid,group['date'],flush=True);return
    if any(i['collection']!='sentinel-2-c1-l2a' for i in items):raise ValueError('Use C1 radiometry only')
    pr=profile(plan,3);size=pr['width'];affine=pr['transform']
    forward=Transformer.from_crs(4326,3576,always_xy=True).transform
    footprints=[transform(forward,shape(i['geometry'])).buffer(0) for i in items]
    with rasterio.open(folder/'reflectance.tif','w',**pr) as refl, \
         rasterio.open(folder/'rgb.tif','w',**pr) as rgb, \
         rasterio.open(folder/'source-index.tif','w',**profile(plan,1,'uint16',0)) as origin:
        for window in windows(size):
            shape2=(int(window.height),int(window.width));local=affine*Affine.translation(window.col_off,window.row_off)
            bounds=rasterio.windows.bounds(window,affine);extent=box(*bounds)
            bands=np.full((3,*shape2),np.nan,dtype='float32');colors=bands.copy();indices=np.zeros(shape2,'uint16')
            selected_good=np.zeros(shape2,bool)
            for number,(item,footprint) in enumerate(zip(items,footprints),1):
                if selected_good.all():break
                if not footprint.intersects(extent):continue
                covered=geometry_mask([mapping(footprint)],shape2,local,invert=True)
                if not ((~selected_good)&covered).any():continue
                cfg={'crs':'EPSG:3576'}
                # Independent COG bands can be read concurrently. Each worker
                # owns its rasterio dataset; no shared GDAL handles.
                keys=['green','swir16','scl','red','blue']
                arrays=list(ASSET_POOL.map(lambda key:read_asset(item['assets'][key],cfg,local,shape2,key=='scl'),keys))
                green,swir,scl,red,blue=arrays
                present=np.isfinite(green)&np.isfinite(swir)&np.isfinite(scl)&(scl>0)
                good=present&~np.isin(scl,[1,2,3,8,9,10])&(green>=0)&(swir>=0)
                take=present&((indices==0)|(good&~selected_good))
                if not take.any():continue
                for dest,arrays in [(bands,[green,swir,scl]),(colors,[red,green,blue])]:
                    for b,values in enumerate(arrays):dest[b,take]=values[take]
                selected_good[take]=good[take];indices[take]=number
            refl.write(bands,window=window);rgb.write(colors,window=window);origin.write(indices,1,window=window)
            print('optical block',rid,group['date'],int(window.row_off),int(window.col_off),flush=True)
        refl.update_tags(product='C1_L2A_same_datatake_mosaic_scale_offset_applied_once')
    source=dict(items[0]);source['mosaic:sources']=items;source['mosaic:datatake']=group['datatake']
    save(folder/'source.json',source);save(folder/'status.json',{'bounds':plan['bounds'],'expanded_signature':sig})
    features,status=segment(folder,rid,ocean,output)
    sources=[{'id':i['id'],'datetime':i['properties']['datetime'],'sun':i['properties']['view:sun_elevation']} for i in items]
    source_properties(features,folder/'source-index.tif',sources,frame_id,'optical')
    save(output/'candidates.geojson',{'type':'FeatureCollection','features':features})
    status.update(expanded_signature=sig,source_ids=[i['id'] for i in items],frame_id=frame_id,
                  marine_area_km2=plan['marine_area_km2'],nominal_area_km2=plan['nominal_area_km2'])
    save(status_file,status)
    save(folder/'status.json',dict(status,valid_fraction=status['valid_area_km2']/plan['nominal_area_km2']))
    print('DONE optical',rid,group['date'],status['complete_candidates'],'complete',flush=True)


def sar(plan,group,root,ocean):
    root=Path(root);rid=plan['id'];date=group['date'];items=group['items']
    folder=root/'native'/rid;folder.mkdir(parents=True,exist_ok=True)
    path=folder/(date+'.tif');output=root/'detected/scenes'/rid;status_file=output/(date+'-status.json')
    sig=signature(plan,group,'sar')
    if status_file.exists() and json.loads(status_file.read_text(encoding='utf-8')).get('expanded_signature')==sig:
        print('cached SAR',rid,date,flush=True);return
    pr=profile(plan,4);size=pr['width'];affine=pr['transform']
    forward=Transformer.from_crs(4326,3576,always_xy=True).transform
    footprints=[transform(forward,shape(i['geometry'])).buffer(0) for i in items]
    source_info={}
    with rasterio.open(path,'w',**pr) as dst, \
         rasterio.open(folder/(date+'-source-index.tif'),'w',**profile(plan,1,'uint16',0)) as origin:
        for window in windows(size):
            shape2=(int(window.height),int(window.width));bounds=rasterio.windows.bounds(window,affine);extent=box(*bounds)
            bands=np.full((4,*shape2),np.nan,dtype='float32');indices=np.zeros(shape2,'uint16')
            for number,(item,footprint) in enumerate(zip(items,footprints),1):
                if not footprint.intersects(extent):continue
                key='_'.join(item['id'].split('_')[:8])
                part=root/'native-parts'/rid/key/(f'{int(window.row_off)}-{int(window.col_off)}.tif')
                info=read_scene(item['id'],bounds,part,root/'source-cache');source_info[number]=info
                with rasterio.open(part) as src:values=src.read()
                compose_sar(bands,indices,values,number)
            dst.write(bands,window=window);origin.write(indices,1,window=window)
            print('SAR block',rid,date,int(window.row_off),int(window.col_off),flush=True)
        dst.update_tags(product='experimental_annotation_calibrated_sigma0',source_level='L1',
                        geocoding='GCP_TPS_no_precise_orbit_update',composition='same_absolute_orbit_first_co_pol_pixel')
    if not source_info:raise ValueError('No intersecting SAR inputs')
    info=dict(source_info[min(source_info)]);frame_id=info['source_id']
    info.update(bounds=plan['bounds'],signature=sig,mosaic_sources=[source_info[k] for k in sorted(source_info)],
                mosaic_source_order=[{'index':n,'source_id':'_'.join(i['id'].split('_')[:8])} for n,i in enumerate(items,1)])
    save(path.with_suffix('.json'),info)
    features,status=detect(path,rid,output,ocean)
    sources=[{'id':'_'.join(i['id'].split('_')[:8]),'datetime':source_info[n]['datetime'] if n in source_info else i['properties']['datetime']}
             for n,i in enumerate(items,1)]
    source_properties(features,folder/(date+'-source-index.tif'),sources,frame_id,'sar')
    save(output/(date+'.geojson'),{'type':'FeatureCollection','features':features})
    status.update(expanded_signature=sig,frame_id=frame_id,source_ids=[s['id'] for s in sources],
                  marine_area_km2=plan['marine_area_km2'],nominal_area_km2=plan['nominal_area_km2'])
    save(status_file,status)
    print('DONE SAR',rid,date,status['target_count'],'targets',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['root','overview','sensor']:p.add_argument('--'+name,required=True)
    p.add_argument('--regions',default='all')
    args=p.parse_args();root=Path(args.root)
    if args.sensor not in ['optical','sar']:raise ValueError('Choose optical or sar')
    ocean=shape(json.loads((Path(args.overview)/'projected.json').read_text(encoding='utf-8'))['ocean']).buffer(-1000)
    for file in sorted((root/'regions').glob('*/plan.json')):
        plan=json.loads(file.read_text(encoding='utf-8'))
        if args.regions!='all' and plan['id'] not in args.regions.split(','):continue
        for group in plan[args.sensor]:
            (optical if args.sensor=='optical' else sar)(plan,group,root,ocean)


if __name__=='__main__':main()
