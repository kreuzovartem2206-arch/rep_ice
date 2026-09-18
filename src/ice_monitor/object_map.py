"""Conservative dual-polarization target footprints and repeat-location hypotheses.

Targets are not automatically called ice floes or grounded ridges. The detected
radar footprint diameter is not the physical diameter of an ice floe.
"""
import argparse
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.features import shapes
from scipy import ndimage
from shapely.geometry import shape, mapping, box
from shapely.ops import transform
from shapely.strtree import STRtree

from .cli import save
from .regional import geographic_geometry

PARAMETERS={'minimum_radar_diameter_m':30,'co_contrast_db':6,'cross_contrast_db':3,
            'co_snr_db':6,'cross_snr_db':3,'outer_window_pixels':61,
            'guard_window_pixels':11,'maximum_aspect_ratio':10,
            'repeat_radius_m':75,'minimum_repeat_observations':3,'minimum_repeat_days':24,
            'maximum_gap_days':40,'stationarity_is_not_grounding':True}


def ring_mean(values, good, outer=61, guard=11):
    values=np.where(good,values,0).astype('float32')
    count=ndimage.uniform_filter(good.astype('float32'),outer,mode='constant')*outer**2
    count-=ndimage.uniform_filter(good.astype('float32'),guard,mode='constant')*guard**2
    total=ndimage.uniform_filter(values,outer,mode='constant')*outer**2
    total-=ndimage.uniform_filter(values,guard,mode='constant')*guard**2
    return np.divide(total,count,out=np.full_like(total,np.nan),where=count>0),count/(outer**2-guard**2)


def candidate_mask(co, cross, co_snr, cross_snr, ocean, spacing=10, parameters=None):
    p=parameters or PARAMETERS
    good_co=np.isfinite(co)&(co>0)&ocean
    good_cross=np.isfinite(cross)&(cross>0)&ocean
    bg_co,fraction_co=ring_mean(co,good_co,p['outer_window_pixels'],p['guard_window_pixels'])
    bg_cross,fraction_cross=ring_mean(cross,good_cross,p['outer_window_pixels'],p['guard_window_pixels'])
    valid=(good_co&good_cross&(co_snr>=p['co_snr_db'])&(cross_snr>=p['cross_snr_db'])
           &(fraction_co>=.9)&(fraction_cross>=.5))
    co_contrast=10*np.log10(np.maximum(co,1e-12)/np.maximum(bg_co,1e-12))
    cross_contrast=10*np.log10(np.maximum(cross,1e-12)/np.maximum(bg_cross,1e-12))
    raw=valid&(co_contrast>=p['co_contrast_db'])&(cross_contrast>=p['cross_contrast_db'])
    labels,n=ndimage.label(raw) # Four-connected, no dilation connecting speckle.
    areas=np.bincount(labels.ravel())*spacing**2
    keep=areas>=math.pi*(p['minimum_radar_diameter_m']/2)**2
    keep[0]=False
    # A target touching poor data or tile border is not treated as complete.
    border=ndimage.binary_dilation(~(good_co&good_cross))
    border[[0,-1],:]=True;border[:,[0,-1]]=True
    keep[np.unique(labels[border])]=False
    for label,slices in enumerate(ndimage.find_objects(labels),1):
        if not keep[label] or slices is None:continue
        height=slices[0].stop-slices[0].start;width=slices[1].stop-slices[1].start
        if max(width,height)/min(width,height)>p['maximum_aspect_ratio']:keep[label]=False
    retained=np.where(keep[labels],labels,0).astype('int32')
    return retained,valid,co_contrast,cross_contrast


def detect(path, region_id, output, ocean_geometry=None):
    path=Path(path);info=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    with rasterio.open(path) as src:
        if src.tags().get('product')!='experimental_annotation_calibrated_sigma0':
            raise ValueError('Expected native annotation-calibrated tile')
        co,co_snr,cross,cross_snr=src.read()
        affine,crs=src.transform,src.crs
        if src.res!=(10,10):raise ValueError('Unsupported sample spacing')
    if ocean_geometry is None:
        raise ValueError('Explicit ocean mask required; no implicit all-ocean assumption')
    from rasterio.features import geometry_mask
    ocean=geometry_mask([mapping(ocean_geometry)],co.shape,affine,invert=True)
    labels,valid,contrast,cross_contrast=candidate_mask(co,cross,co_snr,cross_snr,ocean)
    inverse=Transformer.from_crs(crs,4326,always_xy=True).transform
    features=[]
    for geometry,value in shapes(labels,mask=labels>0,transform=affine):
        polygon=shape(geometry);which=labels==int(value)
        x,y=inverse(polygon.centroid.x,polygon.centroid.y)
        props={'id':hashlib.sha256((info['source_id']+polygon.wkb_hex).encode()).hexdigest()[:16],
               'region':region_id,'class':'unclassified_radar_target','ice_identity':'unverified',
               'grounding':'unverified','confidence':'low','observed_at':info['datetime'],
               'source_id':info['source_id'],'source_level':'L1','radar_area_m2':round(polygon.area,1),
               'radar_equivalent_diameter_m':round(2*math.sqrt(polygon.area/math.pi),2),
               'physical_floe_diameter_m':None,'longitude':round(x,7),'latitude':round(y,7),
               'native_spacing_m':10,'effective_resolution_m':'20 x 22',
               'co_contrast_max_db':round(float(np.nanmax(contrast[which])),2),
               'cross_contrast_max_db':round(float(np.nanmax(cross_contrast[which])),2),
               'co_snr_median_db':round(float(np.nanmedian(co_snr[which])),2),
               'cross_snr_median_db':round(float(np.nanmedian(cross_snr[which])),2),
               'review_status':'unreviewed','possible_confusions':'ridge, ship, fast ice, unresolved land, speckle',
               'stream':f"{region_id}:{info['platform']}:{info['relative_orbit']}:{info['orbit_direction']}:{','.join(info['polarizations'])}"}
        features.append({'type':'Feature','geometry':geographic_geometry(transform(inverse,polygon)),'properties':props})
    folder=Path(output);folder.mkdir(parents=True,exist_ok=True)
    save(folder/(path.stem+'.geojson'),{'type':'FeatureCollection','features':features})
    mask=np.where(valid,1,0).astype('uint8');mask[labels>0]=2
    with rasterio.open(folder/(path.stem+'-mask.tif'),'w',driver='GTiff',width=mask.shape[1],height=mask.shape[0],
                       count=1,dtype='uint8',crs=crs,transform=affine,nodata=0,compress='deflate',tiled=True) as dst:
        dst.write(mask,1);dst.update_tags(classes='0=insufficient_signal_or_geometry;1=no_selected_target;2=unclassified_radar_target')
    status={'region':region_id,'source_id':info['source_id'],'datetime':info['datetime'],
            'target_count':len(features),'ocean_pixels':int(ocean.sum()),'qualified_pixels':int(valid.sum()),
            'qualified_area_km2':round(float(valid.sum())*.0001,4),'parameters':PARAMETERS,
            'raw_native_tile':str(path),'bounds':info['bounds'],'crs':info['crs']}
    save(folder/(path.stem+'-status.json'),status)
    return features,status


def repeated_locations(features, crs='EPSG:3576', parameters=None):
    """Mutually unique nearby matches; hypothesis only, no inferred grounding."""
    p=parameters or PARAMETERS;project=Transformer.from_crs(4326,crs,always_xy=True).transform
    streams={};tracks={}
    for f in features:
        props=f['properties'];streams.setdefault(props['stream'],{}).setdefault(props['observed_at'],[]).append(f)
    for frames in streams.values():
        prior=None
        for timestamp,frame in sorted(frames.items()):
            now=datetime.fromisoformat(timestamp.replace('Z','+00:00'))
            geometries=[transform(project,shape(f['geometry'])) for f in frame]
            matches={}
            if prior and (now-prior['time']).total_seconds()<=p['maximum_gap_days']*86400:
                tree=STRtree(prior['geometries']);pairs=[]
                for j,g in enumerate(geometries):
                    for i in tree.query(g.centroid.buffer(p['repeat_radius_m']),predicate='intersects'):
                        old=prior['geometries'][i]
                        if .5<=g.area/old.area<=2 and g.centroid.distance(old.centroid)<=p['repeat_radius_m']:
                            pairs.append((int(i),j))
                for i,j in pairs:
                    if sum(a==i for a,b in pairs)==1 and sum(b==j for a,b in pairs)==1:matches[j]=i
            ids=[]
            for j,(f,g) in enumerate(zip(frame,geometries)):
                track_id=prior['ids'][matches[j]] if j in matches else f['properties']['id']
                tracks.setdefault(track_id,[]).append((f,g,now));ids.append(track_id)
                f['properties']['repeat_track_id']=track_id
            prior={'geometries':geometries,'ids':ids,'time':now}
    persistent=[]
    for track_id,history in tracks.items():
        duration=(history[-1][2]-history[0][2]).total_seconds()/86400
        excursion=max(a[1].centroid.distance(b[1].centroid) for a in history for b in history)
        accepted=len(history)>=p['minimum_repeat_observations'] and duration>=p['minimum_repeat_days'] and excursion<=p['repeat_radius_m']
        for f,g,t in history:
            f['properties'].update(repeat_observations=len(history),repeat_span_days=round(duration,2),
                                   repeat_excursion_m=round(excursion,1),persistent_location_candidate=bool(accepted))
        if accepted:
            last=history[-1][0]
            props=dict(last['properties'],class_name='persistent_radar_target',
                       interpretation='Repeat signal near a fixed location; fast ice and association errors not excluded',
                       grounding='unverified',confirmed_stamukha=False,
                       observation_ids=[h[0]['properties']['id'] for h in history])
            persistent.append({'type':'Feature','geometry':last['geometry'],'properties':props})
    return persistent


def build(plan_path, native_root, overview_root, output):
    plan=json.loads(Path(plan_path).read_text(encoding='utf-8'));output=Path(output);output.mkdir(parents=True,exist_ok=True)
    # The coarse shoreline is an explicit limitation, not a 30 m land mask.
    projected=json.loads((Path(overview_root)/'projected.json').read_text(encoding='utf-8'))
    ocean=shape(projected['ocean']).buffer(-1000)
    all_features=[];statuses=[];coverage=[];missing=[]
    for region in plan['regions']:
        for scene in region['scenes']:
            date=scene['properties']['datetime'][:10]
            path=Path(native_root)/region['id']/(date+'.tif')
            if not path.exists():missing.append({'region':region['id'],'date':date});continue
            features,status=detect(path,region['id'],output/'scenes'/region['id'],ocean)
            all_features.extend(features);statuses.append(status)
            footprint=box(*status['bounds'])
            inverse=Transformer.from_crs(status['crs'],4326,always_xy=True).transform
            coverage.append({'type':'Feature','geometry':geographic_geometry(transform(inverse,footprint)),
                             'properties':{'region':region['id'],'datetime':status['datetime'],'processing':'native_10m_target_screening',
                                           'classification_validated':False,'source_id':status['source_id'],'qualified_area_km2':status['qualified_area_km2']}})
            print(region['id'],date,len(features),'targets',status['qualified_area_km2'],'qualified km2',flush=True)
    persistent=repeated_locations(all_features)
    for name,features in [('observations',all_features),('persistent-targets',persistent),('processed-coverage',coverage),('confirmed-stamukhi',[])]:
        save(output/(name+'.geojson'),{'type':'FeatureCollection','features':features})
    summary={'status':'partial_research_screening','full_arctic_complete':False,'planned_regions':len(plan['regions']),
             'processed_frames':len(statuses),'target_observations':len(all_features),
             'persistent_location_candidates':len(persistent),'confirmed_ice_floes':0,'confirmed_stamukhi':0,
             'missing_frames':missing,'parameters':PARAMETERS,'frames':statuses,
             'notice':'Radar targets are not a validated map of individual ice floes. Empty confirmed layer means no grounding evidence, not no stamukhi.'}
    save(output/'summary.json',summary)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True);parser.add_argument('--native',required=True)
    parser.add_argument('--overview',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();build(args.plan,args.native,args.overview,args.output)


if __name__=='__main__':main()
