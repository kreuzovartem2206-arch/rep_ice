import argparse
import json
import math
import os
import shutil
import zipfile
from pathlib import Path


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as PolygonPatch
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import numpy as np
import rasterio
from pyproj import Transformer
from shapely import segmentize, make_valid
from shapely.geometry import shape, mapping, box, MultiPolygon
from shapely.geometry.polygon import orient
from shapely.ops import transform, unary_union

parser=argparse.ArgumentParser(description="Export the winter SAR overview maps; no ice classification")
parser.add_argument('--input',default='runs/russian-arctic-winter')
parser.add_argument('--output',default='runs/winter-maps')
parser.add_argument('--config',default='config/russian_arctic_winter.json')
args=parser.parse_args()
work=Path(args.input);out=Path(args.output);out.mkdir(exist_ok=True,parents=True)
projected=json.loads((work/'projected.json').read_text())
summary=json.loads((work/'summary.json').read_text())
grid=json.loads((work/'coverage-grid.geojson').read_text())
cfg=json.loads(Path(args.config).read_text())
crs=projected['crs'];forward=Transformer.from_crs(4326,crs,always_xy=True).transform
inverse=Transformer.from_crs(crs,4326,always_xy=True).transform
ocean=shape(projected['ocean']);land=shape(projected['land'])
x0,y0,x1,y1=ocean.bounds
labels=[{'text':'Баренцево|море','at':[43,74.5],'small':True},
        {'text':'Карское|море','at':[77,76],'small':True},
        {'text':'Море|Лаптевых','at':[123,76.5],'small':True},
        {'text':'Восточно-Сибирское|море','at':[159,74.5],'small':True},
        {'text':'Чукотское|море','at':[-175,72],'small':True},
        {'text':'Арктический бассейн','at':[105,86],'small':False}]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11})

def polygons(geom):
    if geom.geom_type=='Polygon':return [geom]
    return [p for part in geom.geoms for p in polygons(part)] if hasattr(geom,'geoms') else []

def draw_geom(ax,geom,face,edge=None,alpha=1,zorder=2):
    patches=[PolygonPatch(np.asarray(p.exterior.coords)/1000,closed=True) for p in polygons(geom)]
    collection=PatchCollection(patches,facecolor=face,edgecolor=edge or 'none',linewidth=.25,alpha=alpha,zorder=zorder)
    ax.add_collection(collection)

def base(ax):
    ax.set_facecolor('#f6f8fc');ax.set_aspect('equal')
    ax.set_xlim(x0/1000-100,x1/1000+100);ax.set_ylim(y0/1000-100,y1/1000+100)
    ax.set_axis_off()
    draw_geom(ax,ocean,'#dce5ed',zorder=0)

def adorn(ax):
    draw_geom(ax,land,'#b7c1c9','#7f919d')
    for label in labels:
        x,y=forward(*label['at'])
        ax.text(x/1000,y/1000,label['text'].replace('|','\n'),ha='center',va='center',fontsize=10,color='#182f43',
                bbox={'boxstyle':'round,pad=.22','facecolor':'white','edgecolor':'none','alpha':.75})
    # 500 km scale in the equal-area projected overview, not a navigation scale.
    xs=x0/1000+250;ys=y0/1000+100
    ax.plot([xs,xs+500],[ys,ys],color='#243c50',lw=2)
    ax.text(xs+250,ys+70,'500 км',ha='center',color='#243c50')

month_names={'2025-12':'Декабрь 2025','2026-01':'Январь 2026','2026-02':'Февраль 2026'}
mosaic_info=[]
for month,_,_ in cfg['months']:
    status=json.loads((work/f'radar-{month}-status.json').read_text())
    if 'valid_ocean_fraction' not in status:raise RuntimeError(f'{month} still running')
    with rasterio.open(work/f'radar-{month}-amplitude.tif') as src:
        arr=src.read(1);bounds=src.bounds
    fig,ax=plt.subplots(figsize=(14,8.4));fig.subplots_adjust(left=.015,right=.985,bottom=.12,top=.84)
    base(ax)
    display=np.ma.masked_equal(arr,0)
    ax.imshow(display,cmap='gray',vmin=1,vmax=254,origin='upper',interpolation='nearest',zorder=1,
              extent=[bounds.left/1000,bounds.right/1000,bounds.bottom/1000,bounds.top/1000])
    adorn(ax)
    fig.suptitle(f'Российская Арктика · {month_names[month]}',x=.04,y=.97,ha='left',fontsize=21,fontweight='bold')
    coverage=100*status['valid_ocean_fraction']
    fig.text(.04,.89,f'Радиолокационный обзор L1 · сетка 2 км · {len(status["processed"])} сцен · заполнено {coverage:.1f}% морской области',fontsize=12)
    fig.text(.04,.075,'Серое изображение — относительная амплитуда SAR. Голубые пробелы — нет данных в этой мозаике.',fontsize=11,color='#3b5368')
    fig.text(.04,.044,'Съёмки разных дней; это не карта классификации льда или стамух. Объекты 30 м на обзоре не различимы.',fontsize=11,color='#3b5368')
    fig.text(.04,.012,'Contains modified Copernicus Sentinel data 2025–2026 · Earth Search / AWS · Natural Earth · EPSG:3576',fontsize=9,color='#546d7d')
    fig.savefig(out/f'radar-{month}.png',dpi=160);plt.close(fig)
    mosaic_info.append({'month':month,'processed_scenes':len(status['processed']),
                        'valid_mosaic_percent':round(coverage,2),'failed_scenes':len(status['failed']),
                        'first_date':min(x['datetime'] for x in status['processed']),
                        'last_date':max(x['datetime'] for x in status['processed'])})
    for suffix in ['amplitude.tif','date.tif','scene-index.tif','status.json']:
        shutil.copyfile(work/f'radar-{month}-{suffix}',out/f'radar-{month}-{suffix}')

# Comparable footprint coverage maps: Jan IW/EW with the exact same scale.
fig,axes=plt.subplots(1,2,figsize=(16,7));fig.subplots_adjust(left=.01,right=.99,bottom=.16,top=.78,wspace=.015)
for ax,mode in zip(axes,['IW','EW']):
    base(ax)
    for f in grid['features']:
        ix,iy=map(int,f['properties']['id'].split(':'))
        cell=box(ix*100000,iy*100000,(ix+1)*100000,(iy+1)*100000).intersection(ocean)
        fraction=f['properties']['stats']['2026-01'][mode]['coverage_fraction']
        draw_geom(ax,cell,plt.cm.Blues(fraction))
    adorn(ax)
    s=next(s for s in summary['summary'] if s['month']=='2026-01' and s['mode']==mode)
    ax.set_title(f'{mode}: {s["footprint_coverage_percent"]:.2f}% области\n{s["scenes"]} сцен',fontsize=14)
fig.suptitle('Зимняя доступность Sentinel-1 · январь 2026',x=.04,ha='left',fontsize=21,fontweight='bold')
fig.text(.04,.89,'Контуры доступных сцен. Наличие сцены не означает обнаружение льдины.',fontsize=12)
cax=fig.add_axes([.30,.10,.40,.018]);fig.colorbar(ScalarMappable(norm=Normalize(0,100),cmap='Blues'),cax=cax,orientation='horizontal',label='Площадь ячейки внутри контуров сцен, %')
fig.text(.04,.015,'Морская аналитическая область: 6,25 млн км². Границы не являются государственными. CDSE / Natural Earth.',fontsize=10)
fig.savefig(out/'coverage-IW-EW-january.png',dpi=160);plt.close(fig)


for name in ['coverage-grid.geojson','scene-footprints.geojson','domain.geojson','summary.json','summary.csv']:
    shutil.copyfile(work/name,out/name)
(out/'mosaic-summary.json').write_text(json.dumps(mosaic_info,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(mosaic_info,ensure_ascii=False))
