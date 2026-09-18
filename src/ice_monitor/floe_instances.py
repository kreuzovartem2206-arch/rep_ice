"""Experimental instance contours from C1 L2A, with explicit split uncertainty.

This is an independently implemented marker-controlled distance watershed, not
a reproduction or validation of published floe-segmentation models.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.features import shapes, geometry_mask
from scipy import ndimage
from shapely.geometry import shape, mapping
from shapely.ops import transform
from skimage.morphology import h_maxima
from skimage.measure import label as connected_labels
from skimage.segmentation import watershed

from .cli import save
from .processing import optical_mask
from .regional import geographic_geometry

SETTINGS={'green_thresholds':[.15,.18,.21],'central_threshold':.18,'ndsi_threshold':.4,
          'minimum_diameter_m':30,'marker_prominence_pixels':3,'cloud_buffer_m':60,
          'stability_iou_threshold':.6,'method':'distance_watershed_experimental_v1'}


def split_instances(ice, prominence=3):
    distance=ndimage.distance_transform_edt(np.pad(ice,1))[1:-1,1:-1]
    maxima=h_maxima(distance,prominence).astype(bool)&ice
    components,count=ndimage.label(ice)
    # Isolated small components must not disappear merely because their radius
    # is below the marker prominence used to suppress extra peaks in large floes.
    has_marker=set(np.unique(components[maxima]))
    for label,slices in enumerate(ndimage.find_objects(components),1):
        if label in has_marker or slices is None:continue
        part=components[slices]==label
        local=np.where(part,distance[slices],-1)
        r,c=np.unravel_index(local.argmax(),local.shape)
        maxima[slices[0].start+r,slices[1].start+c]=True
    markers,_=ndimage.label(maxima)
    divided=watershed(-distance,markers,mask=ice,watershed_line=True,connectivity=1)
    return connected_labels(divided,background=0,connectivity=1).astype('int32')


def label_iou(reference, variant, label):
    mask=reference==label
    candidates,counts=np.unique(variant[mask],return_counts=True)
    scores=[]
    for other,intersection in zip(candidates,counts):
        if other:
            union=mask.sum()+np.count_nonzero(variant==other)-intersection
            scores.append(float(intersection/union))
    return max(scores,default=0)


def segment(folder, region, ocean_geometry, output):
    folder=Path(folder);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    item=json.loads((folder/'source.json').read_text())
    if item['collection']!='sentinel-2-c1-l2a':raise ValueError('Use unambiguous Collection 1 radiometry')
    with rasterio.open(folder/'reflectance.tif') as src:
        green,swir,scl=src.read();affine,crs=src.transform,src.crs
        if src.res!=(10,10):raise ValueError('Expected 10 m analysis grid')
    ocean=geometry_mask([mapping(ocean_geometry)],green.shape,affine,invert=True)
    cfg={'resolution_m':10,'cloud_buffer_m':SETTINGS['cloud_buffer_m'],'ndsi_threshold':.4}
    variants=[];central=None;central_ice=None;valid=None
    for threshold in SETTINGS['green_thresholds']:
        ice,good=optical_mask(green,swir,scl,ocean,dict(cfg,green_min=threshold))
        labels=split_instances(ice,SETTINGS['marker_prominence_pixels'])
        variants.append(labels)
        if threshold==SETTINGS['central_threshold']:central,central_ice,valid=labels,ice,good
    sizes=np.bincount(central.ravel())
    min_pixels=math.ceil(math.pi*(SETTINGS['minimum_diameter_m']/2)**2/100)
    keep=sizes>=min_pixels;keep[0]=False
    border=ndimage.binary_dilation(~valid)
    border[[0,-1],:]=True;border[:,[0,-1]]=True
    truncated=set(np.unique(central[border]))
    inferred_line=central_ice&(central==0)
    inferred_adjacent=ndimage.binary_dilation(inferred_line)
    inverse=Transformer.from_crs(crs,4326,always_xy=True).transform
    features=[];kept=np.where(keep[central],central,0).astype('int32')
    for geometry,label_float in shapes(kept,mask=kept>0,transform=affine):
        label=int(label_float);polygon=shape(geometry)
        mask=central==label;boundary=mask&~ndimage.binary_erosion(mask)
        inferred_fraction=float((boundary&inferred_adjacent).sum()/max(1,boundary.sum()))
        stability=min(label_iou(central,v,label) for v in variants)
        lon,lat=inverse(polygon.centroid.x,polygon.centroid.y)
        is_truncated=label in truncated
        props={'id':hashlib.sha256((item['id']+region+polygon.wkb_hex+SETTINGS['method']).encode()).hexdigest()[:16],
               'class':'floe_instance_candidate','ice_identity':'optical_candidate','grounding':'unverified',
               'confirmed_stamukha':False,'confidence':'low','review_status':'unreviewed',
               'region':region,'observed_at':item['properties']['datetime'],'source_id':item['id'],
               'source_collection':item['collection'],'source_level':'L2A','area_m2':round(polygon.area,1),
               'equivalent_diameter_m':round(2*math.sqrt(polygon.area/math.pi),2),
               'longitude':round(lon,7),'latitude':round(lat,7),'truncated':is_truncated,
               'minimum_threshold_iou':round(stability,3),'threshold_stable':bool(stability>=SETTINGS['stability_iou_threshold']),
               'inferred_boundary_fraction':round(inferred_fraction,3),'touching_floes_may_be_split':inferred_fraction>0,
               'isolated_stable_candidate':bool(not is_truncated and stability>=SETTINGS['stability_iou_threshold'] and inferred_fraction==0),
               'low_sun':item['properties'].get('view:sun_elevation',0)<20,
               'sun_elevation_deg':item['properties'].get('view:sun_elevation'),
               'processing_method':SETTINGS['method'],'native_green_m':10,'native_swir_m':20,
               'notice':'Candidate outline; shape splitting can divide one floe or merge multiple floes; 30 m completeness unvalidated'}
        features.append({'type':'Feature','geometry':geographic_geometry(transform(inverse,polygon)),'properties':props})
    save(output/'candidates.geojson',{'type':'FeatureCollection','features':features})
    with rasterio.open(output/'labels.tif','w',driver='GTiff',width=kept.shape[1],height=kept.shape[0],count=1,
                       dtype='int32',crs=crs,transform=affine,nodata=-1,compress='deflate',tiled=True) as dst:
        dst.write(np.where(valid,kept,-1).astype('int32'),1)
    complete=[f for f in features if not f['properties']['truncated']]
    status={'source_id':item['id'],'region':region,'datetime':item['properties']['datetime'],
            'candidate_count':len(features),'complete_candidates':len(complete),
            'complete_stable_candidates':sum(f['properties']['threshold_stable'] for f in complete),
            'valid_area_km2':round(float(valid.sum())*.0001,4),'settings':SETTINGS,
            'negative_reflectance_fraction':float(((green<0)|(swir<0)).mean()),
            'bounds':json.loads((folder/'status.json').read_text())['bounds'],'source_level':'L2A'}
    save(output/'status.json',status)
    return features,status


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--optical',required=True)
    p.add_argument('--overview',required=True);p.add_argument('--output',required=True)
    p.add_argument('--selection',required=True,help='Directory of selected-optical region manifests')
    args=p.parse_args()
    projected=json.loads((Path(args.overview)/'projected.json').read_text())
    ocean=shape(projected['ocean']).buffer(-1000)
    features=[];statuses=[]
    sources=[]
    for manifest in sorted(Path(args.selection).glob('*.json')):
        for selected in json.loads(manifest.read_text())['selected']:
            sources.append(Path(args.optical)/selected['region']/selected['source_id']/'source.json')
    if not sources:raise ValueError('No selected optical frames')
    for source in sources:
        folder=source.parent;region=folder.parent.name
        fs,status=segment(folder,region,ocean,Path(args.output)/'scenes'/region/folder.name)
        features.extend(fs);statuses.append(status)
        print(region,folder.name,status['complete_candidates'],'complete',status['complete_stable_candidates'],'stable',flush=True)
    save(Path(args.output)/'floe-candidates.geojson',{'type':'FeatureCollection','features':features})
    save(Path(args.output)/'summary.json',{'full_arctic_complete':False,'settings':SETTINGS,'frames':statuses,
         'complete_candidates':sum(s['complete_candidates'] for s in statuses),
         'complete_stable_candidates':sum(s['complete_stable_candidates'] for s in statuses)})


if __name__=='__main__':main()
