import math
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from pyproj import CRS, Transformer
from rasterio.enums import Resampling
from rasterio.features import geometry_mask, shapes
from rasterio.vrt import WarpedVRT
from scipy import ndimage
from shapely.geometry import mapping, shape
from shapely.ops import transform as transform_geometry


def grid(config, ocean):
    crs = CRS(config["crs"])
    if not crs.is_projected or crs.axis_info[0].unit_name != "metre":
        raise ValueError("Analysis CRS must be projected in metres")
    forward = Transformer.from_crs(4326, crs, always_xy=True).transform
    polygon = transform_geometry(forward, ocean)
    step = config["resolution_m"]
    x0, y0, x1, y1 = polygon.bounds
    x0, y0 = math.floor(x0 / step) * step, math.floor(y0 / step) * step
    width, height = math.ceil((x1 - x0) / step), math.ceil((y1 - y0) / step)
    if width * height > config["max_pixels"]:
        raise ValueError("AOI too large; tile it before processing")
    affine = Affine(step, 0, x0, 0, -step, y0 + height * step)
    mask = geometry_mask([mapping(polygon)], (height, width), affine, invert=True)
    if not mask.any():
        raise ValueError("Empty ocean AOI")
    return affine, mask


def read_asset(asset, config, affine, size, categorical=False):
    with rasterio.Env(GDAL_HTTP_MAX_RETRY=3, GDAL_HTTP_RETRY_DELAY=2,
                      GDAL_HTTP_TIMEOUT=90, GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
        with rasterio.open(asset["href"]) as src:
            with WarpedVRT(src, crs=config["crs"], transform=affine,
                           width=size[1], height=size[0], nodata=float("nan"),
                           dtype="float32", resampling=Resampling.nearest if categorical else Resampling.bilinear) as vrt:
                values = vrt.read(1, masked=True).filled(np.nan)
    if categorical:
        return values
    meta = asset.get("raster:bands", [{}])[0]
    if "scale" not in meta or "offset" not in meta:
        raise ValueError("Missing per-asset scale/offset; refusing to guess reflectance calibration")
    return values * meta["scale"] + meta["offset"]


def optical_mask(green, swir, scl, ocean, config):
    # SCL 11 (snow/ice) MUST remain valid. Clouds are not evidence of no ice.
    bad = ~np.isfinite(scl) | np.isin(scl, [0, 1, 2, 3, 8, 9, 10])
    iterations = math.ceil(config["cloud_buffer_m"] / config["resolution_m"])
    if iterations:
        bad = ndimage.binary_dilation(bad, iterations=iterations)
    valid = ocean & ~bad & np.isfinite(green) & np.isfinite(swir) & (green >= 0) & (swir >= 0)
    denominator = green + swir
    ndsi = np.divide(green - swir, denominator, out=np.full_like(green, np.nan),
                     where=denominator > 1e-6)
    ice = valid & (ndsi >= config["ndsi_threshold"]) & (green >= config["green_min"])
    return ice, valid


def vectorize(ice, valid, affine, config, sensor="S2", effective_resolution=20):
    labels, count = ndimage.label(ice)  # Four neighbours: no diagonal self-intersections.
    area_pixel = abs(affine.a * affine.e)
    areas = np.bincount(labels.ravel()) * area_pixel
    min_area = math.pi * (config["min_diameter_m"] / 2) ** 2
    keep = areas >= min_area
    keep[0] = False
    retained = np.where(keep[labels], labels, 0).astype("int32")
    edge = ndimage.binary_dilation(~valid)
    edge[[0, -1], :] = True
    edge[:, [0, -1]] = True
    edge_labels = set(labels[edge & ice].tolist())
    inverse = Transformer.from_crs(config["crs"], 4326, always_xy=True).transform
    result = []
    for geom, value in shapes(retained, mask=retained > 0, transform=affine):
        polygon = shape(geom)
        diameter = 2 * math.sqrt(polygon.area / math.pi)
        truncated = int(value) in edge_labels
        reliable = max(config["reliable_diameter_m"], 5 * effective_resolution)
        props = {"class": "ice_candidate" if sensor == "S2" else "radar_target_candidate",
                 "sensor": sensor, "area_m2": round(polygon.area, 2),
                 "equivalent_diameter_m": round(diameter, 2),
                 "effective_resolution_m": effective_resolution,
                 "confidence": "low" if truncated or diameter < reliable or sensor != "S2" else "medium",
                 "truncated": truncated, "grounding": "unverified"}
        result.append({"type": "Feature", "geometry": mapping(transform_geometry(inverse, polygon)),
                       "properties": props})
    return result


def process_s2(item, config, affine, ocean):
    if item.get("collection") not in ["sentinel-2-l2a", "sentinel-2-c1-l2a"]:
        raise ValueError("Expected Sentinel-2 L2A; unsupported product level")
    if (item.get("collection") == "sentinel-2-l2a"
            and item.get("properties", {}).get("earthsearch:boa_offset_applied") is True
            and any(item["assets"][k].get("raster:bands", [{}])[0].get("offset", 0) != 0
                    for k in ["green", "swir16"])):
        raise ValueError("Conflicting legacy BOA offset metadata; use sentinel-2-c1-l2a")
    arrays = [read_asset(item["assets"][key], config, affine, ocean.shape, key == "scl")
              for key in ["green", "swir16", "scl"]]
    ice, valid = optical_mask(*arrays, ocean, config)
    features = vectorize(ice, valid, affine, config)
    return features, valid, ice


def process_sar(path, config, affine, ocean, mode, pol, band=1):
    """Experimental bright-target detector. Input: calibrated, geocoded linear sigma0."""
    if mode not in ["IW", "EW"] or pol not in ["HH", "HV", "VV", "VH"]:
        raise ValueError("Specify known SAR acquisition mode and polarization")
    with rasterio.open(path) as src:
        if src.tags().get("product") == "uncalibrated_L1_visual_overview":
            raise ValueError("Visual overview is not calibrated linear sigma0")
        if not src.crs:
            raise ValueError("SAR input must be geocoded")
        with WarpedVRT(src, crs=config["crs"], transform=affine,
                       height=ocean.shape[0], width=ocean.shape[1], dtype="float32",
                       nodata=float("nan"), resampling=Resampling.bilinear) as vrt:
            sigma = vrt.read(band, masked=True).filled(np.nan)
    valid = ocean & np.isfinite(sigma) & (sigma > 0)
    # Reject likely dB input rather than silently miscalibrating it.
    if np.any(ocean) and np.count_nonzero(ocean & (sigma < 0)) > 0.01 * ocean.sum():
        raise ValueError("Expected linear sigma0, not dB or uncalibrated DN")
    db = 10 * np.log10(np.maximum(sigma, 1e-12))
    finite = db[valid]
    if not finite.size:
        return [], valid, np.zeros_like(ocean)
    filled = np.where(valid, db, np.median(finite))
    background = ndimage.median_filter(filled, size=15)
    # Local contrast baseline; ships, wind fronts and ridges can trigger it.
    target = valid & (db - background >= 6)
    features = vectorize(target, valid, affine, config, "S1", 22 if mode == "IW" else 93)
    for feature in features:
        feature["properties"].update({"mode": mode, "polarization": pol, "band": band})
    return features, valid, target


def write_mask(path, valid, ice, affine, config):
    data = np.where(valid, ice.astype("uint8"), 255)
    with rasterio.open(path, "w", driver="GTiff", height=data.shape[0], width=data.shape[1],
                       count=1, dtype="uint8", crs=config["crs"], transform=affine,
                       nodata=255, compress="deflate", tiled=True) as dst:
        dst.write(data, 1)
