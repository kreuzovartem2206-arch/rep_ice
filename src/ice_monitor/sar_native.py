"""Windowed Sentinel-1 IW L1 reading and annotated Sigma0 calibration.

Uses native measurement samples, never overview pixels. GCP geocoding is an
experimental ocean approximation, not a replacement for validated precise-orbit
range-Doppler processing. Absolute position accuracy is not asserted.
"""
import hashlib
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from pyproj import Transformer
from rasterio.control import GroundControlPoint
from rasterio.enums import Resampling
from rasterio.transform import GCPTransformer
from rasterio.warp import reproject
from rasterio.windows import Window
from scipy.interpolate import interp1d

from .catalog import session
from .cli import save

PUBLIC_ROOT = 'https://sentinel-s1-l1c.s3.eu-central-1.amazonaws.com/'


def public_href(value):
    if not value.startswith('s3://sentinel-s1-l1c/'):
        raise ValueError('Unexpected measurement source')
    return value.replace('s3://sentinel-s1-l1c/', PUBLIC_ROOT, 1)


def cached_get(client, url, path):
    path=Path(path)
    if not path.exists():
        response=client.get(url,timeout=(15,90));response.raise_for_status()
        path.parent.mkdir(parents=True,exist_ok=True)
        partial=path.with_suffix(path.suffix+'.part')
        partial.write_bytes(response.content);partial.replace(path)
    return path.read_bytes()


def interpolate_vectors(root, vector_tag, value_tag, rows, cols):
    """Bilinear LUT interpolation at original measurement line/sample indices."""
    vectors=sorted(root.findall('.//'+vector_tag),key=lambda v:float(v.findtext('line')))
    if len(vectors)<2:raise ValueError('Missing calibration/noise vectors')
    lines=np.array([float(v.findtext('line')) for v in vectors])
    if np.any(np.diff(lines)<=0):raise ValueError('Duplicate or unordered LUT lines')
    if rows.min()<lines[0] or rows.max()>lines[-1]:raise ValueError('LUT does not cover source rows')
    samples=[]
    for v in vectors:
        pixels=np.fromstring(v.findtext('pixel'),sep=' ')
        values=np.fromstring(v.findtext(value_tag),sep=' ')
        if len(pixels)!=len(values) or np.any(np.diff(pixels)<=0):raise ValueError('Invalid LUT samples')
        if cols.min()<pixels[0] or cols.max()>pixels[-1]:raise ValueError('LUT does not cover source columns')
        samples.append(np.interp(cols,pixels,values))
    return interp1d(lines,np.asarray(samples),axis=0,bounds_error=True)(rows).astype('float32')


def noise_power(root, rows, cols):
    # IPF >=2.90: annotated range noise multiplied by azimuth gain.
    if root.find('.//noiseRangeVector') is not None:
        noise=interpolate_vectors(root,'noiseRangeVector','noiseRangeLut',rows,cols)
        azimuth=np.full(noise.shape,np.nan,dtype='float32')
        for v in root.findall('.//noiseAzimuthVector'):
            r=(rows>=int(v.findtext('firstAzimuthLine')))&(rows<=int(v.findtext('lastAzimuthLine')))
            c=(cols>=int(v.findtext('firstRangeSample')))&(cols<=int(v.findtext('lastRangeSample')))
            if not r.any() or not c.any():continue
            lines=np.fromstring(v.findtext('line'),sep=' ')
            values=np.fromstring(v.findtext('noiseAzimuthLut'),sep=' ')
            azimuth[np.ix_(r,c)]=np.interp(rows[r],lines,values)[:,None]
        return noise*azimuth
    # Legacy products have no separate azimuth modulation.
    return interpolate_vectors(root,'noiseVector','noiseLut',rows,cols)


def calibrate(dn, calibration, noise, rows, cols, already_denoised=False):
    lut=interpolate_vectors(calibration,'calibrationVector','sigmaNought',rows,cols)
    eta=noise_power(noise,rows,cols)
    power=dn.astype('float32')**2
    signal=power if already_denoised else power-eta
    valid=(dn>0)&np.isfinite(eta)&(eta>0)&(lut>0)&(signal>0)
    sigma=np.full(dn.shape,np.nan,dtype='float32')
    snr=np.full(dn.shape,np.nan,dtype='float32')
    sigma[valid]=signal[valid]/lut[valid]**2
    snr[valid]=10*np.log10(signal[valid]/eta[valid])
    return sigma,snr


def annotation_root(client,item,pol,cache):
    key='schema-product-'+pol
    url=public_href(item['assets'][key]['href'])
    root=ET.fromstring(cached_get(client,url,cache/(key+'.xml')))
    if root.tag=='product':return root,url
    # Some Earth Search items expose RFI under schema-product. Resolve actual
    # annotation from the published bucket listing; never parse RFI as product.
    prefix=item['assets'][pol]['href'].split('/measurement/')[0].replace('s3://sentinel-s1-l1c/','')+'/annotation/'
    listing=client.get(PUBLIC_ROOT,params={'list-type':'2','prefix':prefix},timeout=(15,60))
    listing.raise_for_status()
    keys=[e.text for e in ET.fromstring(listing.content).iter() if e.tag.endswith('}Key')]
    expected='iw-'+pol+'.xml'
    matches=[k for k in keys if k.rsplit('/',1)[-1]==expected and '/rfi/' not in k]
    if len(matches)!=1:raise ValueError('Ambiguous product annotation')
    url=PUBLIC_ROOT+matches[0]
    root=ET.fromstring(cached_get(client,url,cache/('annotation-'+pol+'.xml')))
    if root.tag!='product':raise ValueError('Missing product annotation')
    return root,url


def read_scene(item_id, bounds, output, cache, crs='EPSG:3576', spacing=10):
    """Read one ocean tile, both polarizations, retaining SNR and source metadata."""
    if spacing!=10:raise ValueError('IW object branch requires the explicit 10 m grid')
    cache=Path(cache);output=Path(output)
    key='_'.join(item_id.split('_')[:8]);folder=cache/key
    signature=hashlib.sha256(json.dumps([key,list(bounds),crs,spacing,'native-v1']).encode()).hexdigest()
    manifest=output.with_suffix('.json')
    if output.exists() and manifest.exists():
        info=json.loads(manifest.read_text())
        if info.get('signature')==signature:return info
        raise ValueError('Output belongs to another tile/configuration')
    xmin,ymin,xmax,ymax=bounds
    w,h=round((xmax-xmin)/spacing),round((ymax-ymin)/spacing)
    if w*h>9000000:raise ValueError('Split ocean tile: more than 9 million pixels')
    affine=Affine(spacing,0,xmin,0,-spacing,ymax)
    bands=[];labels=[];sources=[]
    with session() as client:
        item=json.loads(cached_get(client,'https://earth-search.aws.element84.com/v1/collections/sentinel-1-grd/items/'+key,folder/'item.json'))
        props=item['properties']
        if props.get('sar:instrument_mode')!='IW' or str(props.get('s1:processing_level'))!='1' or props.get('s1:resolution')!='high':
            raise ValueError('30 m candidate branch only accepts IW GRDH L1')
        pols=['vv','vh'] if 'vv' in item['assets'] else ['hh','hv']
        if not all(p in item['assets'] for p in pols):raise ValueError('Dual polarization required')
        for pol in pols:
            annotation,annotation_url=annotation_root(client,item,pol,folder)
            denoise=annotation.findtext('.//thermalNoiseCorrectionPerformed')
            if denoise not in ['true','false']:raise ValueError('Denoising state is unknown')
            href=public_href(item['assets'][pol]['href'])
            with rasterio.Env(GDAL_HTTP_TIMEOUT=90,GDAL_HTTP_MAX_RETRY=3,GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR'):
                with rasterio.open(href) as src:
                    gcps,gcp_crs=src.gcps
                    if not gcps or not gcp_crs:raise ValueError('Missing GCP geolocation')
                    project=Transformer.from_crs(gcp_crs,crs,always_xy=True)
                    projected=[]
                    for g in gcps:
                        x,y=project.transform(g.x,g.y)
                        projected.append(GroundControlPoint(row=g.row,col=g.col,x=x,y=y,z=g.z))
                    xs,ys=np.meshgrid(np.linspace(xmin,xmax,9),np.linspace(ymin,ymax,9))
                    with GCPTransformer(projected,tps=True) as transformer:
                        rr,cc=transformer.rowcol(xs.ravel(),ys.ravel(),op=lambda value:value)
                    left=max(0,math.floor(min(cc))-100);right=min(src.width,math.ceil(max(cc))+100)
                    top=max(0,math.floor(min(rr))-100);bottom=min(src.height,math.ceil(max(rr))+100)
                    if right<=left or bottom<=top:raise ValueError('Tile outside source raster')
                    if (right-left)*(bottom-top)>18000000:raise ValueError('Excessive native read window')
                    window=Window(left,top,right-left,bottom-top)
                    # No out_shape, no overview or reduced-resolution read.
                    dn=src.read(1,window=window)
            rows=np.arange(top,bottom);cols=np.arange(left,right)
            roots={};urls={}
            for kind in ['calibration','noise']:
                asset='schema-'+kind+'-'+pol;url=public_href(item['assets'][asset]['href'])
                roots[kind]=ET.fromstring(cached_get(client,url,folder/(asset+'.xml')));urls[kind]=url
            sigma,snr=calibrate(dn,roots['calibration'],roots['noise'],rows,cols,denoise=='true')
            local_gcps=[GroundControlPoint(row=g.row-top,col=g.col-left,x=g.x,y=g.y,z=g.z) for g in projected]
            for name,array in [(pol+'_sigma0',sigma),(pol+'_snr_db',snr)]:
                result=np.full((h,w),np.nan,dtype='float32')
                reproject(array,result,gcps=local_gcps,src_crs=crs,src_nodata=np.nan,
                          dst_crs=crs,dst_transform=affine,dst_nodata=np.nan,
                          resampling=Resampling.nearest,MAX_GCP_ORDER=-1)
                bands.append(result);labels.append(name)
            sources.append({'polarization':pol.upper(),'measurement':href,'annotation':annotation_url,
                            **urls,'native_window':[left,top,right-left,bottom-top],
                            'noise_previously_removed':denoise=='true'})
            print(f'{key} {pol}: native {dn.shape}, finite {np.isfinite(sigma).mean():.3f}',flush=True)
    output.parent.mkdir(parents=True,exist_ok=True)
    with rasterio.open(output,'w',driver='GTiff',width=w,height=h,count=len(bands),dtype='float32',
                       crs=crs,transform=affine,nodata=np.nan,compress='deflate',tiled=True) as dst:
        for i,(label,array) in enumerate(zip(labels,bands),1):dst.write(array,i);dst.set_band_description(i,label)
        dst.update_tags(product='experimental_annotation_calibrated_sigma0',source_level='L1',geocoding='GCP_TPS_no_precise_orbit_update')
    info={'signature':signature,'source_id':key,'datetime':props['datetime'],'input_level':'L1',
          'mode':'IW','native_spacing_m':10,'effective_resolution_m':[20,22],
          'platform':props['platform'],'relative_orbit':props['sat:relative_orbit'],
          'orbit_direction':props['sat:orbit_state'],'polarizations':pols,'bands':labels,
          'bounds':list(bounds),'crs':crs,'sources':sources,
          'geolocation_validated':False,'calibration_independently_validated':False,
          'notice':'Annotation-calibrated prototype; not a validated ice classifier or a grounding measurement.'}
    save(manifest,info)
    return info
