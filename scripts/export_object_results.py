"""Export real observations, a portable review map and scientific quicklooks."""
import argparse
import base64
import hashlib
import io
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from PIL import Image
from pyproj import Transformer
from shapely.geometry import shape, mapping
from shapely.ops import transform
from build_arctic_map import build as build_arctic_map

NAMES={'barents':'Баренцево море','kara':'Карское море','laptev':'Море Лаптевых',
       'east_siberian':'Восточно-Сибирское море','chukchi':'Чукотское море'}


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def save(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
def fc(features):return {'type':'FeatureCollection','features':features}
def polygons(g):return list(g.geoms) if hasattr(g,'geoms') else [g]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True);p.add_argument('--overview',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();root=Path(args.input);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    plan=read(root/'plan.json');all_floes=read(root/'floes/floe-candidates.geojson')['features']
    complete=[f for f in all_floes if not f['properties']['truncated']]
    stable=[f for f in complete if f['properties']['threshold_stable']]
    isolated=[f for f in complete if f['properties']['isolated_stable_candidate']]
    for name,features in [('floe-candidates',complete),('threshold-stable-candidates',stable),
                          ('isolated-stable-candidates',isolated),
                          ('excluded-truncated',[f for f in all_floes if f['properties']['truncated']])]:
        save(out/(name+'.geojson'),fc(features))
    for source,target in [('observations','radar-targets'),('persistent-targets','persistent-radar-targets'),
                          ('confirmed-stamukhi','confirmed-stamukhi'),('processed-coverage','sar-coverage')]:
        shutil.copy2(root/'detected'/(source+'.geojson'),out/(target+'.geojson'))
    save(out/'processing-plan.json',plan)
    sar=read(root/'detected/summary.json');optical=read(root/'floes/summary.json')
    save(out/'radar-processing.json',sar);save(out/'optical-processing.json',optical)
    forward=Transformer.from_crs(4326,3576,always_xy=True).transform
    ui=[];coverage=[];table=[]
    for region in plan['regions']:
        selected=read(root/'selected-optical'/(region['id']+'.json'))['selected']
        for selected_frame in selected:
            identifier=selected_frame['source_id'];folder=root/'optical-c1'/region['id']/identifier
            fs=[f for f in complete if f['properties']['source_id']==identifier and f['properties']['region']==region['id']]
            frame_status=next(s for s in optical['frames'] if s['source_id']==identifier and s['region']==region['id'])
            with rasterio.open(folder/'reflectance.tif') as src:
                green=src.read(1);affine=src.transform
            if (folder/'rgb.tif').exists():
                with rasterio.open(folder/'rgb.tif') as src:rgb=np.moveaxis(src.read(),0,-1)
                kind='Цветной снимок'
            else:rgb=np.repeat(green[:,:,None],3,axis=2);kind='Канал B03, оттенки серого'
            # Fixed display stretch, never used for scientific segmentation.
            display=np.nan_to_num(np.clip(rgb/.65,0,1),nan=0)**(1/1.6)
            im=Image.fromarray(np.uint8(display*255));stream=io.BytesIO();im.save(stream,format='JPEG',quality=86)
            image_url='data:image/jpeg;base64,'+base64.b64encode(stream.getvalue()).decode()
            pix=[]
            for f in fs:
                g=transform(forward,shape(f['geometry']))
                g=transform(lambda x,y,z=None:((np.asarray(x)-affine.c)/affine.a,(np.asarray(y)-affine.f)/affine.e),g)
                pix.append({'geometry':mapping(g),'properties':f['properties']})
            ui.append({'region':region['id'],'name':NAMES[region['id']],'id':identifier,'date':selected_frame['datetime'][:10],
                       'valid':round(selected_frame['valid_fraction']*100,1),'image':image_url,'kind':kind,'features':pix,
                       'sun':read(folder/'source.json')['properties'].get('view:sun_elevation'),
                       'stable':sum(f['properties']['threshold_stable'] for f in fs),
                       'isolated':sum(f['properties']['isolated_stable_candidate'] for f in fs)})
            destination=out/'evidence'/region['id']/identifier;destination.mkdir(parents=True,exist_ok=True)
            im.save(destination/'image.jpg',quality=92)
            for name in ['source.json','status.json']:shutil.copy2(folder/name,destination/name)
            label_folder=root/'floes/scenes'/region['id']/identifier
            for name in ['labels.tif','candidates.geojson']:shutil.copy2(label_folder/name,destination/name)
            fig,ax=plt.subplots(figsize=(9,9));ax.imshow(display,extent=[0,10,10,0])
            for f in pix:
                color='#36ffb2' if f['properties']['threshold_stable'] else '#ffaf31'
                for poly in polygons(shape(f['geometry'])):
                    coords=np.array(poly.exterior.coords)/100
                    ax.plot(coords[:,0],coords[:,1],color=color,lw=.8)
            ax.set(xlabel='км от западной границы участка (EPSG:3576)',ylabel='км от северной границы участка',
                   title=f"{NAMES[region['id']]} · {selected_frame['datetime'][:10]}\n{len(fs)} кандидатов; {frame_status['complete_stable_candidates']} устойчивых к порогу")
            fig.text(.5,.025,'Зелёный: устойчивый контур · оранжевый: неустойчивый · подтверждения стамух нет',ha='center',fontsize=9)
            fig.tight_layout(rect=[0,.04,1,1]);fig.savefig(destination/'map.png',dpi=150);plt.close(fig)
        counts=[f for f in complete if f['properties']['region']==region['id']]
        table.append({'region':region['id'],'name':NAMES[region['id']],'candidate_observations':len(counts),
                      'threshold_stable_observations':sum(f['properties']['threshold_stable'] for f in counts),
                      'radar_target_observations':sum(f['properties']['region']==region['id'] for f in read(out/'radar-targets.geojson')['features'])})
        # Nominal processed tile, not a claim that every pixel is usable.
        from ice_monitor.object_pipeline import tile_grid
        from shapely.geometry import box
        inverse=Transformer.from_crs(3576,4326,always_xy=True).transform
        bounds,_,_=tile_grid(region)
        coverage.append({'type':'Feature','geometry':mapping(transform(inverse,box(*bounds))),
                         'properties':{'region':region['id'],'nominal_area_km2':100,'full_arctic_complete':False}})
    save(out/'processed-areas.geojson',fc(coverage))
    for meta in (root/'native').glob('*/*.json'):
        dest=out/'sources/sar'/meta.parent.name/meta.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(meta,dest)
    projected=read(Path(args.overview)/'projected.json')
    fig,ax=plt.subplots(figsize=(12,6.4));ax.set_facecolor('#e6f0f4')
    for g in polygons(shape(projected['land'])):
        xy=np.array(g.exterior.coords)/1000;ax.fill(xy[:,0],xy[:,1],color='#d0d7d7',ec='#7c898c',lw=.3)
    ocean=shape(projected['ocean'])
    for g in polygons(ocean):
        xy=np.array(g.exterior.coords)/1000;ax.plot(xy[:,0],xy[:,1],color='#668e9e',lw=.5)
    offsets={'barents':(-5,-18),'kara':(8,-18),'laptev':(8,-18),'east_siberian':(-100,18),'chukchi':(-170,18)}
    for r in plan['regions']:
        x,y=np.array(r['projected_center'])/1000
        ax.plot(x,y,'o',ms=7,color='#bf3e28');ax.annotate(NAMES[r['id']],(x,y),xytext=offsets[r['id']],textcoords='offset points',fontsize=10)
    x0,y0,x1,y1=ocean.bounds;ax.set_xlim(x0/1000-100,x1/1000+100);ax.set_ylim(y0/1000-100,y1/1000+100)
    ax.set_aspect('equal');ax.set(xlabel='EPSG:3576 · км',ylabel='EPSG:3576 · км',
       title='Зима 2025/26 · участки поиска отдельных объектов\n5 участков по 10 × 10 км; остальная область не обработана детектором')
    fig.text(.5,.018,'Метки увеличены для видимости. Контур — аналитическая область; не государственная граница. Берег: Natural Earth.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.04,1,1]);fig.savefig(out/'coverage.png',dpi=160);plt.close(fig)
    summary={'status':'partial_experimental_object_map','full_arctic_complete':False,'nominal_processed_area_km2':500,
             'optical_frames':len(ui),'sar_frames':sar['processed_frames'],'candidate_observations':len(complete),
             'threshold_stable_observations':len(stable),'confirmed_stamukhi':0,'persistent_radar_targets':len(read(out/'persistent-radar-targets.geojson')['features']),
             'isolated_stable_observations':len(isolated),
             'minimum_equivalent_diameter_m':30,'completeness_at_30m_validated':False,'regions':table,
             'note':'Counts are observations, not unique tracked floes; stability is not correctness. No grounding evidence was available.'}
    save(out/'summary.json',summary)
    examples=sorted(isolated,key=lambda f:f['properties']['equivalent_diameter_m'],reverse=True)[:8]
    if isolated:examples.append(min(isolated,key=lambda f:f['properties']['equivalent_diameter_m']))
    fig,axes=plt.subplots(3,6,figsize=(15,10))
    for i,f in enumerate(examples):
        row,col=divmod(i,3);p=f['properties'];frame=next(v for v in ui if v['id']==p['source_id'] and v['region']==p['region'])
        rgb=np.asarray(Image.open(io.BytesIO(base64.b64decode(frame['image'].split(',')[1]))))
        pix=next(x for x in frame['features'] if x['properties']['id']==p['id'])
        geom=shape(pix['geometry']);cx,cy=np.clip([geom.centroid.x,geom.centroid.y],24.5,974.5)
        for panel in range(2):
            ax=axes[row,col*2+panel];ax.imshow(rgb);ax.set_xlim(cx-25,cx+25);ax.set_ylim(cy+25,cy-25)
            ax.set_xticks([]);ax.set_yticks([])
            ax.set_title((p['observed_at'][:10]+' · '+str(p['equivalent_diameter_m'])+' м') if panel==0 else 'Контур кандидата',fontsize=9)
            if panel:
                for g in polygons(geom):
                    xy=np.asarray(g.exterior.coords);ax.plot(xy[:,0],xy[:,1],color='#36ffb2',lw=1)
    fig.suptitle('Отдельные устойчивые кандидаты · Баренцево море\nКаждый фрагмент 500 × 500 м; слева снимок, справа контур. Классификация не подтверждена.',fontsize=14)
    fig.subplots_adjust(left=.015,right=.99,bottom=.025,top=.85,wspace=.08,hspace=.32)
    fig.savefig(out/'candidate-examples.png',dpi=150);plt.close(fig)
    template=Path(__file__).with_name('object_review_template.html').read_text(encoding='utf-8')
    overview_b64=base64.b64encode((out/'coverage.png').read_bytes()).decode()
    html=template.replace('__FRAMES__',json.dumps(ui,ensure_ascii=False,separators=(',',':'))).replace('__OVERVIEW__',overview_b64)
    (out/'detail-map.html').write_text(html,encoding='utf-8')
    shutil.copy2(out/'evidence/kara/S2B_T43XDB_20260226T073758_L2A/map.png',out/'kara-candidates.png')
    checks={'all_geometries_valid':all(shape(f['geometry']).is_valid for f in all_floes),
            'all_complete_diameters_at_least_30m':all(f['properties']['equivalent_diameter_m']>=30 for f in complete),
            'no_truncated_in_main_layer':all(not f['properties']['truncated'] for f in complete),
            'no_unfounded_grounding_claim':all(not f['properties']['confirmed_stamukha'] for f in all_floes),
            'frames_match_selection':len(ui)==len(optical['frames'])}
    if not all(checks.values()):raise ValueError(checks)
    save(out/'export-checks.json',checks)
    build_arctic_map(out, args.overview, root/'native', out/'map.html')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
