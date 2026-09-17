"""Public catalogue discovery; all pagination is bounded and explicit."""
import json
import os
from pathlib import Path
from uuid import UUID

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

EARTH_SEARCH = "https://earth-search.aws.element84.com/v1/search"
CDSE = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"


def session():
    result = requests.Session()
    retry = Retry(total=4, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504],
                  allowed_methods=["GET"], respect_retry_after_header=True)
    result.mount("https://", HTTPAdapter(max_retries=retry))
    return result


def s2_search(bbox, start, end, max_items=300):
    params = {"collections": "sentinel-2-l2a", "bbox": ",".join(map(str, bbox)),
              "datetime": f"{start}/{end}", "limit": 100,
              "sortby": "+properties.datetime"}
    items, seen = [], set()
    url = EARTH_SEARCH
    with session() as client:
        while url:
            response = client.get(url, params=params, timeout=(15, 90))
            response.raise_for_status()
            page = response.json()
            for item in page["features"]:
                if item["id"] not in seen:
                    items.append(item)
                    seen.add(item["id"])
            next_link = next((x for x in page.get("links", []) if x["rel"] == "next"), None)
            if len(items) > max_items or (len(items) >= max_items and next_link):
                raise RuntimeError("Catalogue limit reached; split date range or AOI. Nothing was silently truncated.")
            if next_link and next_link.get("method", "GET") != "GET":
                raise RuntimeError("Unexpected STAC pagination method")
            url, params = (next_link["href"] if next_link else None), None
    return sorted(items, key=lambda x: (x["properties"]["datetime"], x["id"]))


def s1_search(bbox, start, end, max_items=300):
    west, south, east, north = bbox
    wkt = f"POLYGON(({west} {south},{east} {south},{east} {north},{west} {north},{west} {south}))"
    params = {"$filter": "Collection/Name eq 'SENTINEL-1' and "
              "Attributes/OData.CSC.StringAttribute/any(a:a/Name eq 'productType' and a/OData.CSC.StringAttribute/Value eq 'GRD') and "
              f"OData.CSC.Intersects(area=geography'SRID=4326;{wkt}') and "
              f"ContentDate/Start ge {start} and ContentDate/Start le {end}",
              "$orderby": "ContentDate/Start asc", "$top": 100, "$expand": "Attributes"}
    result, url = [], CDSE
    with session() as client:
        while url:
            response = client.get(url, params=params, timeout=(15, 90))
            response.raise_for_status()
            page = response.json()
            result.extend(page["value"])
            url, params = page.get("@odata.nextLink"), None
            if len(result) > max_items or (len(result) >= max_items and url):
                raise RuntimeError("Catalogue limit reached; split date range or AOI")
    return result


def download_s1(product_id, destination):
    """Download a CDSE product with a short-lived token supplied in the environment."""
    product_id = str(UUID(product_id))
    token = os.environ.get("CDSE_ACCESS_TOKEN")
    if not token:
        raise ValueError("Set CDSE_ACCESS_TOKEN (free CDSE account); never put it in config or Git.")
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    url = f"https://download.dataspace.copernicus.eu/odata/v1/Products({product_id})/$value"
    with session() as client:
        # Requests strips authorization on redirects to other hosts by default.
        with client.get(url, headers={"Authorization": f"Bearer {token}"},
                        stream=True, timeout=(15, 180)) as response:
            response.raise_for_status()
            with partial.open("wb") as stream:
                for chunk in response.iter_content(1024 * 1024):
                    stream.write(chunk)
    import zipfile
    if not zipfile.is_zipfile(partial):
        raise ValueError("Downloaded response is not a ZIP product")
    partial.replace(destination)
    return str(destination)
