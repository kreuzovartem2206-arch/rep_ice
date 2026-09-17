"""Sequential CDSE -> SNAP -> candidate ingestion, with explicit external prerequisites."""
import argparse
import json
import os
import shutil
import sqlite3
import subprocess
from pathlib import Path
from types import SimpleNamespace

from .catalog import download_s1, s1_search


def product_parameters(product):
    attrs = {a["Name"]: a["Value"] for a in product["Attributes"]}
    if attrs.get("processingLevel") != "LEVEL1" or "GRD" not in product["Name"]:
        raise ValueError("Expected original Sentinel-1 L1 GRD")
    mode = attrs["operationalMode"]
    if mode not in ["IW", "EW"]:
        raise ValueError("Only IW/EW supported")
    if (mode == "IW" and "_IW_GRDH_" not in product["Name"]) or (mode == "EW" and "_EW_GRDM_" not in product["Name"]):
        raise ValueError("Resolution model supports IW GRDH and EW GRDM only")
    pols = attrs["polarisationChannels"].split("&")
    pol = next((p for p in ["VH", "HV", "VV", "HH"] if p in pols), None)
    if not pol:
        raise ValueError("Unsupported polarization")
    orbit = int(attrs["relativeOrbitNumber"])
    direction = attrs["orbitDirection"].lower()
    if direction not in ["ascending", "descending"] or not 1 <= orbit <= 175:
        raise ValueError("Invalid orbit metadata")
    return {"mode": mode, "polarization": pol, "relative_orbit": orbit,
            "direction": direction, "datetime": product["ContentDate"]["Start"],
            "product_id": product["Name"], "band": 1}


def execute(config, ocean, output, start, end, gpt, graph, max_scenes=2):
    from .cli import save, run_sar, connect
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "processed": 0, "cached": 0, "queued": 0, "failed": []}
    try:
        # Validate configuration before downloading or reusing any processed file.
        db = connect(root, config)
        db.close()
        executable = shutil.which(gpt)
        if not executable:
            raise ValueError("ESA SNAP gpt not found; install SNAP Microwave Toolbox and supply its path")
        if not os.environ.get("CDSE_ACCESS_TOKEN"):
            raise ValueError("CDSE_ACCESS_TOKEN is required; token renewal is managed by the operator")
        graph = Path(graph).resolve(strict=True)
        if config["crs"] != "EPSG:32643":
            raise ValueError("The supplied SNAP pilot graph uses EPSG:32643; adapt it before changing CRS")
        products = s1_search(config["bbox"], start, end, config["max_items"])
        save(root / "catalogue-s1.json", products)
        seen = set()
        if (root / "state.sqlite").exists():
            with sqlite3.connect(root / "state.sqlite") as db:
                seen = {row[0] for row in db.execute("SELECT id FROM frames")}
        attempted = 0
        for product in products:
            try:
                params = product_parameters(product)
                key = f"{params['product_id']}:{params['polarization']}:band1"
                if key in seen:
                    report["cached"] += 1
                    continue
                if attempted >= max_scenes:
                    report["queued"] += 1
                    continue
                attempted += 1
                # UUID directories avoid unsafe catalogue filenames.
                from uuid import UUID
                folder = root / "sar-products" / str(UUID(product["Id"]))
                folder.mkdir(parents=True, exist_ok=True)
                save(folder / "product.json", product)
                archive, raster = folder / "source.zip", folder / "sigma0.tif"
                if not archive.exists():
                    download_s1(product["Id"], archive)
                if not raster.exists():
                    # Buffered region reduces subset interpolation artefacts.
                    west, south, east, north = config["bbox"]
                    west, south, east, north = west-.05, south-.05, east+.05, north+.05
                    region = f"POLYGON(({west} {south},{east} {south},{east} {north},{west} {north},{west} {south}))"
                    pending = folder / "sigma0.pending.tif"
                    orbit_type = os.environ.get("SNAP_ORBIT_TYPE", "Sentinel Precise (Auto Download)")
                    command = [executable, str(graph), f"-Pinput={archive.resolve()}",
                               f"-Poutput={pending.resolve()}", f"-Ppolarization={params['polarization']}",
                               f"-Pspacing={10 if params['mode']=='IW' else 40}",
                               f"-PorbitType={orbit_type}", f"-Pregion={region}"]
                    with (folder / "snap.log").open("w", encoding="utf-8") as log:
                        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True,
                                       timeout=7200, shell=False)
                    if not pending.exists():
                        raise RuntimeError("SNAP returned without producing the expected GeoTIFF")
                    pending.replace(raster)
                run_sar(SimpleNamespace(input=str(raster), output=str(root), **params), config, ocean)
                report["processed"] += 1
            except Exception as exc:
                report["failed"].append({"id": product.get("Id"), "error": str(exc)})
        report["status"] = "degraded" if report["failed"] else "ok"
    except Exception as exc:
        report.update({"status": "error", "error": str(exc)})
    from datetime import datetime, timezone
    report["generated_at"] = datetime.now(timezone.utc).isoformat()
    save(root / "sar-status.json", report)
    return 0 if report["status"] == "ok" else 1


def main():
    from .cli import configuration, dates
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/kara.json")
    parser.add_argument("--output", default="runs/kara")
    parser.add_argument("--gpt", default="gpt")
    parser.add_argument("--graph", default="snap/s1_ocean.xml")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--max-scenes", type=int, default=2)
    args = parser.parse_args()
    if args.max_scenes < 1:
        parser.error("max-scenes must be positive")
    config, ocean = configuration(args.config)
    start, end = dates(args, config)
    return execute(config, ocean, args.output, start, end, args.gpt, args.graph, args.max_scenes)


if __name__ == "__main__":
    raise SystemExit(main())
