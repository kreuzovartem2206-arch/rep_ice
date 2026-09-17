"""Regional L1 coverage atlas. Footprint availability is never an ice detection."""
import argparse
import csv
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from pyproj import Transformer
from shapely import make_valid, segmentize
from shapely.geometry import box, shape, mapping, MultiPolygon
from shapely.geometry.polygon import orient
from shapely.affinity import translate
from shapely.ops import transform, unary_union
from shapely.strtree import STRtree

from .catalog import session
from .cli import save

STAC = "https://stac.dataspace.copernicus.eu/v1/search"
LAND_URL = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_land.geojson"


def in_interval(value, start, end):
    parse = lambda s: datetime.fromisoformat(s.replace("Z", "+00:00"))
    return parse(start) <= parse(value) < parse(end)


def geographic_geometry(geom):
    """RFC 7946 polygons cut at the antimeridian for GIS clients."""
    if geom.bounds[2] - geom.bounds[0] > 180:
        geom = transform(lambda x,y,z=None: (np.where(np.asarray(x)<0, np.asarray(x)+360, x), y), geom)
    geom = make_valid(geom)
    def polygons(g):
        if g.geom_type == "Polygon":
            return [orient(g, sign=1)] if not g.is_empty else []
        return [p for child in getattr(g, "geoms", []) for p in polygons(child)]
    west = geom.intersection(box(-180, -90, 180, 90))
    east = translate(geom.intersection(box(180, -90, 360, 90)), xoff=-360)
    return mapping(MultiPolygon(polygons(west) + polygons(east)))


def acquisition_key(item):
    # The final checksum/COG suffix is packaging, not a separate observation.
    return "_".join(item["id"].split("_")[:8])


def search_region(bbox, start, end, destination, limit=20000):
    destination = Path(destination)
    request = {"bbox": bbox, "start": start, "end_exclusive": end}
    if destination.exists():
        cached = json.loads(destination.read_text())
        if cached.get("request") == request and cached.get("complete"):
            return [x for x in cached["items"] if in_interval(x["properties"]["datetime"], start, end)]
    params = {"collections":"sentinel-1-grd", "bbox":",".join(map(str,bbox)),
              "datetime":f"{start}/{end}", "limit":1000,
              "fields":"id,geometry,properties.datetime,properties.sar:instrument_mode,properties.processing:level,properties.sar:polarizations,properties.sat:absolute_orbit,properties.platform,properties._private.product_size"}
    items, seen, pages, url = [], set(), 0, STAC
    with session() as client:
        while url:
            if url in seen:
                raise RuntimeError("Repeated pagination URL")
            seen.add(url)
            r = client.get(url,params=params,timeout=(15,120));r.raise_for_status()
            data = r.json()
            items.extend(x for x in data["features"] if in_interval(x["properties"]["datetime"], start, end))
            if len(items)>limit:
                raise RuntimeError("Regional item limit reached; split the query")
            link=next((x for x in data.get("links",[]) if x["rel"]=="next"),None)
            if link and link.get("method","GET")!="GET":
                raise RuntimeError("Unexpected pagination method")
            url,params=(link["href"] if link else None),None
            pages+=1
            print(f"{start[:7]} bbox {bbox[0]}: page {pages}, {len(items)} scenes",flush=True)
    save(destination,{"complete":True,"request":request,"items":items,"pages":pages,
                      "retrieved_at":datetime.now(timezone.utc).isoformat()})
    return items


def domain_geometry(config, land):
    forward=Transformer.from_crs(4326,config["crs"],always_xy=True).transform
    envelopes=[box(*b) for b in config["query_bboxes"]]
    # Densification preserves parallels through the polar projection.
    projected=[transform(forward,segmentize(p,.25)) for p in envelopes]
    domain=unary_union(projected)
    pieces=[]
    for feature in land["features"]:
        geom=make_valid(shape(feature["geometry"]))
        for envelope in envelopes:
            if geom.intersects(envelope):
                clipped=geom.intersection(envelope)
                if not clipped.is_empty:
                    pieces.append(transform(forward,segmentize(clipped,.2)))
    land_projected=unary_union(pieces)
    return make_valid(domain.difference(land_projected)),land_projected


def project_footprint(geometry, crs):
    """Unwrap rings crossing 180 before planar polygon operations."""
    geom=shape(geometry)
    def unwrap(x,y,z=None):
        a=np.asarray(x)
        return np.where(a<0,a+360,a),y
    if geom.bounds[2]-geom.bounds[0]>180:
        geom=transform(unwrap,geom)
    geom=make_valid(geom)
    forward=Transformer.from_crs(4326,crs,always_xy=True).transform
    return make_valid(transform(forward,segmentize(geom,.25)))


def build(config, folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    land_file=folder/"natural-earth-land.geojson"
    if not land_file.exists():
        with session() as client:
            r=client.get(LAND_URL,timeout=(15,90));r.raise_for_status();land_file.write_bytes(r.content)
    land=json.loads(land_file.read_text())
    ocean,land_projected=domain_geometry(config,land)
    items={}
    for month,start,end in config["months"]:
        for n,bbox in enumerate(config["query_bboxes"]):
            for item in search_region(bbox,start,end,folder/"catalogue"/f"{month}-{n}.json",config["max_items_per_query"]):
                if item["properties"].get("processing:level")!="L1":
                    raise ValueError("Unexpected input level in L1 coverage catalogue")
                items.setdefault(acquisition_key(item),item)
    selected,geometries=[],[]
    for key,item in sorted(items.items()):
        geometry=project_footprint(item["geometry"],config["crs"])
        if geometry.intersects(ocean):
            geometry=geometry.intersection(ocean)
            if geometry.area>1:
                selected.append(item);geometries.append(geometry)
    tree=STRtree(geometries)
    inverse=Transformer.from_crs(config["crs"],4326,always_xy=True).transform
    step=config["analysis_cell_m"]
    x0,y0,x1,y1=ocean.bounds
    cells,footprints=[],[]
    for index,item in enumerate(selected):
        p=item["properties"]
        footprints.append({"type":"Feature","geometry":geographic_geometry(shape(item["geometry"])),"properties":{
            "id":item["id"],"datetime":p["datetime"],"mode":p["sar:instrument_mode"],
            "source_level":"L1","processed":False,"size_bytes":p.get("_private",{}).get("product_size")}})
    for x in np.arange(math.floor(x0/step)*step,x1,step):
        for y in np.arange(math.floor(y0/step)*step,y1,step):
            cell=box(x,y,x+step,y+step).intersection(ocean)
            if cell.area<step**2*.01:
                continue
            indexes=tree.query(cell,predicate="intersects")
            stats={}
            for month,_,_ in config["months"]:
                stats[month]={}
                for mode in ["IW","EW","ALL"]:
                    matching=[int(i) for i in indexes if selected[i]["properties"]["datetime"].startswith(month)
                              and (mode=="ALL" or selected[i]["properties"]["sar:instrument_mode"]==mode)]
                    coverage=unary_union([geometries[i].intersection(cell) for i in matching]) if matching else None
                    dates={selected[i]["properties"]["datetime"][:10] for i in matching}
                    stats[month][mode]={"scenes":len(matching),"days":len(dates),
                        "coverage_fraction":round(coverage.area/cell.area,4) if coverage else 0}
            geom=geographic_geometry(transform(inverse,cell.simplify(1500,preserve_topology=True)))
            cells.append({"type":"Feature","geometry":geom,"properties":{
                "id":f"{int(x/step)}:{int(y/step)}","area_km2":round(cell.area/1e6,1),
                "center":list(transform(inverse,cell.representative_point()).coords)[0],
                "stats":stats,"processing_status":"catalogue_only","ice_objects":None}})
    summaries=[]
    for month,_,_ in config["months"]:
        for mode in ["ALL","IW","EW"]:
            matching=[i for i,p in enumerate(selected) if p["properties"]["datetime"].startswith(month)
                      and (mode=="ALL" or p["properties"]["sar:instrument_mode"]==mode)]
            coverage=unary_union([geometries[i] for i in matching])
            sizes=[selected[i]["properties"].get("_private",{}).get("product_size") for i in matching]
            sizes=[s for s in sizes if isinstance(s,(int,float)) and s>0]
            summaries.append({"month":month,"mode":mode,"scenes":len(matching),
                "footprint_coverage_percent":round(100*coverage.area/ocean.area,2),
                "known_size_gb_lower_bound":round(sum(sizes)/1e9,2),
                "scenes_with_size":len(sizes),"size_metadata_complete":len(sizes)==len(matching)})
    save(folder/"coverage-grid.geojson",{"type":"FeatureCollection","features":cells})
    save(folder/"scene-footprints.geojson",{"type":"FeatureCollection","features":footprints})
    save(folder/"domain.geojson",{"type":"FeatureCollection","features":[{"type":"Feature","properties":{"role":"analysis_envelope"},"geometry":geographic_geometry(transform(inverse,ocean.simplify(1500)))}]})
    # Projected cache for reproducible maps and raster mosaics, explicitly labelled CRS.
    save(folder/"projected.json",{"crs":config["crs"],"ocean":mapping(ocean),"land":mapping(land_projected),
                                 "footprints":[mapping(g.simplify(1000)) for g in geometries],
                                 "ids":[x["id"] for x in selected]})
    save(folder/"summary.json",{"status":"catalogue_coverage_only","source":STAC,"input_level":"L1",
          "scope_note":config["scope_note"],"area_km2":round(ocean.area/1e6),
          "unique_scenes":len(selected),"grid_cells":len(cells),"summary":summaries,
          "generated_at":datetime.now(timezone.utc).isoformat(),
          "ice_objects_processed":False,"map_is_synoptic":False})
    with (folder/"summary.csv").open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(summaries[0]));writer.writeheader();writer.writerows(summaries)
    print(json.dumps({"unique_scenes":len(selected),"cells":len(cells),"summary":summaries}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",default="config/russian_arctic_winter.json")
    p.add_argument("--output",default="runs/russian-arctic-winter")
    args=p.parse_args()
    build(json.loads(Path(args.config).read_text()),args.output)


if __name__=="__main__":
    main()
