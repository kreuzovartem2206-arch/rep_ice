"""Combine existing winter products on one offline, projected geographic map."""
import argparse
import base64
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
import rasterio
from pyproj import Transformer
from shapely.geometry import shape
from shapely.ops import transform


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def data_url(image,kind='WEBP'):
    stream=io.BytesIO();image.save(stream,format=kind)
    return 'data:image/'+kind.lower()+';base64,'+base64.b64encode(stream.getvalue()).decode()
def paths(geometry):
    g=shape(geometry) if isinstance(geometry,dict) else geometry
    polys=list(g.geoms) if hasattr(g,'geoms') else [g]
    rings=[]
    for p in polys:
        for ring in [p.exterior,*p.interiors]:
            rings.append('M'+'L'.join(f'{x:.2f},{-y:.2f}' for x,y in ring.coords)+'Z')
    return ''.join(rings)
def rect(bounds):
    x0,y0,x1,y1=bounds
    return [x0,-y1,x1-x0,y1-y0]


def build(results, overview, native, output):
    results,overview,native,output=map(Path,[results,overview,native,output])
    projected=read(overview/'projected.json');summary=read(results/'summary.json')
    plan=read(results/'processing-plan.json');optical=read(results/'optical-processing.json')
    if projected['crs']!='EPSG:3576' or any(r['crs']!='EPSG:3576' for r in plan['regions']):
        raise ValueError('All map inputs must use EPSG:3576')
    radar=read(results/'radar-processing.json');forward=Transformer.from_crs(4326,3576,always_xy=True)
    candidates=read(results/'floe-candidates.geojson')['features']
    targets=read(results/'radar-targets.geojson')['features']
    assets=read(results/'map-assets.json') if (results/'map-assets.json').exists() else {}
    def objects(features):
        return [{'path':paths(transform(forward.transform,shape(f['geometry']))),'properties':f['properties']} for f in features]
    regions=[]
    for region in plan['regions']:
        rid=region['id'];stats=next(r for r in summary['regions'] if r['region']==rid)
        frames=[]
        for frame in optical['frames']:
            if frame['region']!=rid:continue
            source=results/'evidence'/rid/frame['source_id']
            asset=assets.get(rid+':'+frame['source_id'])
            image=asset['image'] if asset else 'data:image/jpeg;base64,'+base64.b64encode((source/'image.jpg').read_bytes()).decode()
            frames.append({'sensor':'optical','id':frame['source_id'],'date':frame['datetime'][:10],
                           'image':image,'rect':rect(frame['bounds']),'usable':frame['valid_area_km2'],
                           'tiles':asset['tiles'] if asset else [],
                           'quality':asset.get('quality') if asset else None,
                           'objects':objects([f for f in candidates if f['properties'].get('frame_id',f['properties']['source_id'])==frame['source_id'] and f['properties']['region']==rid]),
                           'source':'https://earth-search.aws.element84.com/v1/collections/sentinel-2-c1-l2a/items/'+frame['source_id']})
        for frame in radar['frames']:
            if frame['region']!=rid:continue
            date=frame['datetime'][:10]
            asset=assets.get(rid+':'+frame['source_id'])
            if asset:
                frames.append({'sensor':'radar','id':frame['source_id'],'date':date,
                               'image':asset['image'],'tiles':asset['tiles'],'rect':rect(frame['bounds']),
                               'quality':asset.get('quality'),
                               'usable':frame['qualified_area_km2'],
                               'objects':objects([f for f in targets if f['properties'].get('frame_id',f['properties']['source_id'])==frame['source_id'] and f['properties']['region']==rid]),
                               'source':frame['source_url']})
                continue
            with rasterio.open(native/rid/(date+'.tif')) as src:
                if str(src.crs)!='EPSG:3576':raise ValueError('Unexpected native raster CRS')
                signal=src.read(1);bounds=list(src.bounds)
            # Display only. Analytical Sigma0 is not altered.
            valid=np.isfinite(signal)&(signal>0)
            db=np.full_like(signal,-40);np.log10(signal,out=db,where=valid);db[valid]*=10
            gray=np.uint8(np.nan_to_num(np.clip((db+35)/35,0,1),nan=0)*255)
            rgba=np.stack([gray,gray,gray,np.uint8(valid)*255],axis=-1)
            frames.append({'sensor':'radar','id':frame['source_id'],'date':date,
                           'image':data_url(Image.fromarray(rgba)),'rect':rect(bounds),
                           'usable':frame['qualified_area_km2'],
                           'objects':objects([f for f in targets if f['properties']['source_id']==frame['source_id'] and f['properties']['region']==rid]),
                           'source':'https://earth-search.aws.element84.com/v1/collections/sentinel-1-grd/items/'+frame['source_id']})
        frames.sort(key=lambda f:(f['sensor'],f['date']))
        x,y=region['projected_center']
        regions.append({'id':rid,'name':stats['name'],'center':[x,-y],'rect':frames[0]['rect'],
                        'stats':stats,'frames':frames})
    mosaics=[]
    for month in ['2025-12','2026-01','2026-02']:
        with rasterio.open(overview/f'radar-{month}-amplitude.tif') as src:
            if str(src.crs)!='EPSG:3576':raise ValueError('Unexpected mosaic CRS')
            a=src.read(1,masked=True);bounds=list(src.bounds)
        # Overview is an existing display amplitude product, not calibrated Sigma0.
        v=a.compressed();lo,hi=np.percentile(v,[2,98]);gray=np.uint8(np.clip((a.filled(lo)-lo)/max(hi-lo,1e-9),0,1)*255)
        valid=~np.ma.getmaskarray(a)
        image=Image.fromarray(np.stack([gray,gray,gray,np.uint8(valid)*190],axis=-1))
        mosaics.append({'month':month,'rect':rect(bounds),'image':data_url(image),
                        'notice':'Radar visual mosaic; 2 km pixels, different acquisition dates; no object classification'})
    graticule=[]
    for lat in [65,70,75,80,85]:
        coords=[forward.transform(lon,lat) for lon in np.linspace(28,191,220)]
        graticule.append({'path':'M'+'L'.join(f'{x:.1f},{-y:.1f}' for x,y in coords),'label':str(lat)+'° с. ш.',
                          'at':[coords[len(coords)//2][0],-coords[len(coords)//2][1]]})
    for lon in [30,60,90,120,150,180]:
        coords=[forward.transform(lon,lat) for lat in np.linspace(64,89.5,100)]
        graticule.append({'path':'M'+'L'.join(f'{x:.1f},{-y:.1f}' for x,y in coords),'label':str(lon)+'°',
                          'at':[coords[0][0],-coords[0][1]]})
    data={'regions':regions,'mosaics':mosaics,'land':paths(projected['land']),
          'domain':paths(projected['ocean']),'bounds':rect(shape(projected['ocean']).bounds),
          'graticule':graticule,'summary':summary}
    template=Path(__file__).with_name('arctic_map_template.html').read_text(encoding='utf-8')
    html=template.replace('__MAP_DATA__',json.dumps(data,ensure_ascii=False,separators=(',',':')).replace('</','<\\/'))
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(html,encoding='utf-8')
    counts={'regions':len(regions),'optical_frames':sum(f['sensor']=='optical' for r in regions for f in r['frames']),
            'radar_frames':sum(f['sensor']=='radar' for r in regions for f in r['frames']),
            'optical_objects':sum(len(f['objects']) for r in regions for f in r['frames'] if f['sensor']=='optical'),
            'radar_objects':sum(len(f['objects']) for r in regions for f in r['frames'] if f['sensor']=='radar'),
            'overview_months':len(mosaics),'html_bytes':output.stat().st_size,'self_contained':not bool(assets)}
    assert counts['optical_objects']==summary['candidate_observations']
    assert counts['radar_objects']==radar['target_observations']
    output.with_suffix('.validation.json').write_text(json.dumps(counts,indent=2),encoding='utf-8')
    print(counts)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['results','overview','native','output']:p.add_argument('--'+name,required=True)
    a=p.parse_args();build(a.results,a.overview,a.native,a.output)
