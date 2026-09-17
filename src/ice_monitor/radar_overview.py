"""Low resolution visual SAR atlas from L1 GRD; not calibrated ice classification."""
import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from pyproj import Transformer
from rasterio.control import GroundControlPoint
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.warp import reproject
from shapely.geometry import shape, box, mapping, Point
from shapely.prepared import prep

from .catalog import session
from .cli import save


def choose_scenes(projected, footprints, month, limit=100):
    """Greedy geographic coverage using a 25 km planning grid, latest-scene tie break."""
    ocean=shape(projected['ocean']);prepared=prep(ocean)
    x0,y0,x1,y1=ocean.bounds
    points=np.array([(x,y) for x in np.arange(x0,x1,25000) for y in np.arange(y0,y1,25000)
                     if prepared.contains(Point(x,y))])
    from shapely import contains_xy
    candidates=[]
    for i,feature in enumerate(footprints['features']):
        p=feature['properties']
        if not p['datetime'].startswith(month): continue
        geom=shape(projected['footprints'][i]); covered=contains_xy(geom,points[:,0],points[:,1])
        if covered.any(): candidates.append((p,covered))
    remaining=np.ones(len(points),dtype=bool);chosen=[]
    for _ in range(limit):
        if not candidates or not remaining.any():break
        best=max(range(len(candidates)),key=lambda i:(int((candidates[i][1]&remaining).sum()),candidates[i][0]['datetime']))
        props,covered=candidates.pop(best)
        added=int((covered&remaining).sum())
        if added==0:break
        chosen.append(props);remaining&=~covered
    return chosen,{'planning_points':len(points),'uncovered_planning_points':int(remaining.sum()),'limit':limit}


def read_preview(item, target_crs, max_side=700):
    """Reproject the published amplitude overview using projected GCPs.

    No physical backscatter values are claimed. A per-scene percentile stretch
    is a display operation only. Keeping this separate prevents its use by the
    calibrated SAR detector.
    """
    key=next((k for k in ['hh','vv'] if k in item['assets']),None)
    if not key:raise ValueError('No co-polarized channel')
    href=item['assets'][key]['href']
    if not href.startswith('s3://sentinel-s1-l1c/'):
        raise ValueError('Unexpected public data source')
    href=href.replace('s3://sentinel-s1-l1c/','https://sentinel-s1-l1c.s3.eu-central-1.amazonaws.com/')
    with rasterio.Env(GDAL_HTTP_TIMEOUT=60,GDAL_HTTP_MAX_RETRY=2,GDAL_HTTP_RETRY_DELAY=2,
                      GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR'):
        with rasterio.open(href) as src:
            factor=max(src.height,src.width)/max_side
            h,w=max(1,round(src.height/factor)),max(1,round(src.width/factor))
            values=src.read(1,out_shape=(h,w),resampling=Resampling.nearest).astype('float32')
            gcps,gcp_crs=src.gcps
            if not gcps or not gcp_crs:raise ValueError('GCP georeferencing missing')
            project=Transformer.from_crs(gcp_crs,target_crs,always_xy=True)
            converted=[]
            for g in gcps:
                x,y=project.transform(g.x,g.y)
                converted.append(GroundControlPoint(row=g.row*h/src.height,col=g.col*w/src.width,x=x,y=y,z=g.z))
    valid=values>0
    if valid.sum()<100:raise ValueError('Too few valid overview samples')
    log=np.log10(np.maximum(values,1))
    lo,hi=np.percentile(log[valid],[2,98])
    if hi<=lo:raise ValueError('Empty radiometric range')
    stretched=np.where(valid,1+np.clip((log-lo)/(hi-lo),0,1)*253,0).astype('uint8')
    return stretched,converted,{'source_id':item['id'],'href':href,'polarization':key.upper(),
                              'log10_dn_percentiles':[float(lo),float(hi)],'source_level':'L1',
                              'classification':False,'radiometrically_calibrated':False}


def create(folder, month='2026-01', max_scenes=100):
    folder=Path(folder)
    projected=json.loads((folder/'projected.json').read_text())
    footprints=json.loads((folder/'scene-footprints.geojson').read_text())
    chosen,planning=choose_scenes(projected,footprints,month,max_scenes)
    crs=projected['crs'];ocean=shape(projected['ocean']);step=2000
    x0,y0,x1,y1=ocean.bounds
    x0,y0=math.floor(x0/step)*step,math.floor(y0/step)*step
    w,h=math.ceil((x1-x0)/step),math.ceil((y1-y0)/step)
    affine=Affine(step,0,x0,0,-step,y0+h*step)
    sea=geometry_mask([mapping(ocean)],(h,w),affine,invert=True)
    mosaic=np.zeros((h,w),dtype='uint8');dates=np.zeros((h,w),dtype='uint16')
    provenance=np.zeros((h,w),dtype='uint16')
    report={'month':month,'pixel_m':step,'status':'visual_overview_only','planning':planning,
            'selected_scenes':len(chosen),'processed':[],'failed':[],
            'notice':'Per-scene stretched uncalibrated L1 amplitude; not ice extent, not 30 m detections, not a simultaneous observation.'}
    # Earlier first so recent selected observations win in overlaps.
    chosen=sorted(chosen,key=lambda p:p['datetime'])
    scene_geometries=dict(zip(projected['ids'],projected['footprints']))
    with session() as client:
        for number,p in enumerate(chosen,1):
            try:
                key='_'.join(p['id'].split('_')[:8])
                cached=folder/'radar-cache'/f'{key}.npz'
                meta_path=cached.with_suffix('.json')
                if cached.exists() and meta_path.exists():
                    z=np.load(cached);arr=z['arr'];gcp_values=z['gcps']
                    gcps=[GroundControlPoint(row=v[0],col=v[1],x=v[2],y=v[3],z=v[4]) for v in gcp_values]
                    meta=json.loads(meta_path.read_text())
                else:
                    url='https://earth-search.aws.element84.com/v1/collections/sentinel-1-grd/items/'+key
                    r=client.get(url,timeout=(15,60));r.raise_for_status()
                    item=r.json();arr,gcps,meta=read_preview(item,crs)
                    cached.parent.mkdir(exist_ok=True)
                    np.savez_compressed(cached,arr=arr,gcps=np.array([[g.row,g.col,g.x,g.y,g.z] for g in gcps]))
                    save(meta_path,meta)
                x=[g.x for g in gcps];y=[g.y for g in gcps]
                left=max(0,math.floor((min(x)-affine.c)/step)-2)
                right=min(w,math.ceil((max(x)-affine.c)/step)+2)
                top=max(0,math.floor((affine.f-max(y))/step)-2)
                bottom=min(h,math.ceil((affine.f-min(y))/step)+2)
                if right<=left or bottom<=top:continue
                local=np.zeros((bottom-top,right-left),dtype='uint8')
                reproject(arr,local,gcps=gcps,src_crs=crs,src_nodata=0,
                          dst_crs=crs,dst_transform=affine*Affine.translation(left,top),
                          dst_nodata=0,resampling=Resampling.nearest,MAX_GCP_ORDER=-1)
                patch=mosaic[top:bottom,left:right]
                # GCP interpolation can extend a few pixels outside the published
                # footprint. Never turn this interpolation skirt into coverage.
                footprint_mask=geometry_mask([scene_geometries[p['id']]],local.shape,
                    affine*Affine.translation(left,top),invert=True)
                use=(local>0)&sea[top:bottom,left:right]&footprint_mask
                patch[use]=local[use]
                date_code=int(p['datetime'][:10].replace('-',''))-20250000
                dates[top:bottom,left:right][use]=date_code
                provenance[top:bottom,left:right][use]=number
                report['processed'].append(dict(meta,datetime=p['datetime'],scene_index=number,valid_pixels=int(use.sum())))
                print(f'{month} overview {number}/{len(chosen)} {key} pixels {use.sum()}',flush=True)
            except Exception as exc:
                report['failed'].append({'id':p['id'],'error':str(exc)})
                print(f'Failed {p["id"]}: {type(exc).__name__}',flush=True)
            save(folder/f'radar-{month}-status.json',report)
    for name,arr in [('amplitude',mosaic),('date',dates),('scene-index',provenance)]:
        with rasterio.open(folder/f'radar-{month}-{name}.tif','w',driver='GTiff',width=w,height=h,
                           count=1,dtype=str(arr.dtype),crs=crs,transform=affine,nodata=0,
                           tiled=True,compress='deflate') as dst:
            dst.write(arr,1)
            dst.update_tags(product='uncalibrated_L1_visual_overview',month=month,
                            date_encoding='stored_value + 20250000 = YYYYMMDD')
    report['valid_ocean_fraction']=float(((mosaic>0)&sea).sum()/sea.sum())
    save(folder/f'radar-{month}-status.json',report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',default='runs/russian-arctic-winter')
    p.add_argument('--month',default='2026-01')
    p.add_argument('--max-scenes',type=int,default=100)
    args=p.parse_args();create(args.input,args.month,args.max_scenes)


if __name__=='__main__':main()
