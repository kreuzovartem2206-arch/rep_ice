import argparse
import hashlib
import json
import logging
import sqlite3
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from shapely.geometry import box, shape
from shapely.ops import unary_union

from .catalog import download_s1, s1_search, s2_search
from .processing import grid, process_s2, process_sar, write_mask
from .tracking import instant, track

LOG = logging.getLogger("ice-monitor")


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def configuration(path):
    path = Path(path)
    config = json.loads(path.read_text(encoding="utf-8"))
    west, south, east, north = config["bbox"]
    if not (-180 <= west < east <= 180 and 60 <= south < north <= 90):
        raise ValueError("Expected Arctic bbox; split antimeridian AOIs into two")
    for key in ["resolution_m", "min_diameter_m", "max_pixels", "max_items", "lookback_days",
                "max_track_gap_hours", "min_track_gap_minutes", "max_speed_m_s", "position_uncertainty_m"]:
        if config[key] <= 0:
            raise ValueError(f"{key} must be positive")
    if not 0 < config["min_valid_fraction"] <= 1:
        raise ValueError("min_valid_fraction must be in (0,1]")
    obj = json.loads((path.parent / config["ocean_aoi"]).read_text(encoding="utf-8"))
    ocean = unary_union([shape(f["geometry"]) for f in obj["features"]]).intersection(box(*config["bbox"]))
    if ocean.is_empty or not ocean.is_valid:
        raise ValueError("Ocean AOI is empty or invalid")
    config["fingerprint"] = hashlib.sha256((json.dumps(config, sort_keys=True) + ocean.wkt).encode()).hexdigest()
    return config, ocean


def dates(args, config):
    end = instant(args.end) if args.end else datetime.now(timezone.utc)
    start = instant(args.start) if args.start else end - timedelta(days=config["lookback_days"])
    if start >= end:
        raise ValueError("start must be before end")
    return start.isoformat().replace("+00:00", "Z"), end.isoformat().replace("+00:00", "Z")


def connect(root, config):
    root.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(root / "state.sqlite", timeout=30)
    db.execute("CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT)")
    db.execute("CREATE TABLE IF NOT EXISTS frames(id TEXT PRIMARY KEY, body TEXT NOT NULL)")
    prior = db.execute("SELECT value FROM metadata WHERE key='config'").fetchone()
    if prior and prior[0] != config["fingerprint"]:
        db.close()
        raise ValueError("Configuration changed: use a new output directory to preserve provenance")
    db.execute("INSERT OR IGNORE INTO metadata VALUES ('config',?)", (config["fingerprint"],))
    db.commit()
    return db


def export(db, root, config, run):
    frames = [json.loads(row[0]) for row in db.execute("SELECT body FROM frames")]
    features, events = track(frames, config)
    usable = [f for f in frames if f["valid_fraction"] >= config["min_valid_fraction"]]
    latest = max((instant(f["datetime"]) for f in usable), default=None)
    now = datetime.now(timezone.utc)
    age = (now - latest).total_seconds() / 3600 if latest else None
    freshness = {}
    for sensor in ["S2", "S1"]:
        last = max((instant(f["datetime"]) for f in usable if f["stream"].startswith(sensor)), default=None)
        hours = (now-last).total_seconds()/3600 if last else None
        freshness[sensor] = {"last_usable_observation": last.isoformat() if last else None,
                             "age_hours": hours, "stale": hours is None or hours > config["stale_hours"]}
    run.update({"generated_at": now.isoformat(), "last_usable_observation": latest.isoformat() if latest else None,
                "age_hours": age, "stale": age is None or age > config["stale_hours"],
                "total_scenes": len(frames), "total_observations": len(features),
                "config_fingerprint": config["fingerprint"],
                "freshness_by_sensor": freshness,
                "small_objects_optical_stale": freshness["S2"]["stale"],
                "notice": "Research candidates. No observation does not mean no ice. Grounding is unverified."})
    save(root / "observations.geojson", {"type": "FeatureCollection", "features": features})
    save(root / "events.json", events)
    save(root / "coverage.json", [{k:f[k] for k in ["id", "datetime", "valid_fraction", "stream"]} for f in frames])
    save(root / "status.json", run)


def run_optical(args, config, ocean):
    root = Path(args.output)
    affine, mask = grid(config, ocean)
    db = connect(root, config)
    run = {"status": "ok", "processed": 0, "cached": 0, "failed": [], "partial": []}
    try:
        start, end = dates(args, config)
        items = s2_search(config["bbox"], start, end, config["max_items"])
        run.update({"search_start": start, "search_end": end, "found": len(items)})
        save(root / "search.json", {"type": "FeatureCollection", "features": items})
        for item in items:
            key = item["id"]
            if db.execute("SELECT 1 FROM frames WHERE id=?", (key,)).fetchone():
                run["cached"] += 1
                continue
            try:
                LOG.info("Processing %s", key)
                features, valid, ice = process_s2(item, config, affine, mask)
                fraction = float(valid.sum() / mask.sum())
                frame = {"id": key, "datetime": item["properties"]["datetime"], "stream": "S2",
                         "valid_fraction": fraction, "features": features,
                         "level": "L2A", "source": item}
                scene_dir = root / "scenes" / hashlib.sha256(key.encode()).hexdigest()[:20]
                scene_dir.mkdir(parents=True, exist_ok=True)
                write_mask(scene_dir / "mask.tif", valid, ice, affine, config)
                save(scene_dir / "scene.json", frame)
                db.execute("INSERT INTO frames VALUES (?,?)", (key, json.dumps(frame)))
                db.commit()
                run["processed"] += 1
                if fraction < config["min_valid_fraction"]:
                    run["partial"].append(key)
            except Exception as exc:
                LOG.exception("Scene failed: %s", key)
                run["failed"].append({"id": key, "error": str(exc)})
        run["status"] = "degraded" if run["failed"] else ("no_new_catalogue_items" if not items else "ok")
    except Exception as exc:
        run.update({"status": "error", "error": str(exc)})
        LOG.exception("Catalogue/run failure")
    finally:
        save(root / "optical-status.json", run)
        export(db, root, config, run)
        db.close()
    return 1 if run["status"] in ["degraded", "error"] else 0


def run_sar(args, config, ocean):
    root = Path(args.output)
    instant(args.datetime)
    affine, mask = grid(config, ocean)
    db = connect(root, config)
    try:
        key = f"{args.product_id}:{args.polarization}:band{args.band}"
        if db.execute("SELECT 1 FROM frames WHERE id=?", (key,)).fetchone():
            export(db, root, config, {"status": "ok", "cached": 1})
            return 0
        features, valid, ice = process_sar(args.input, config, affine, mask, args.mode, args.polarization, args.band)
        frame = {"id": key, "datetime": args.datetime,
                 "stream": f"S1:{args.mode}:{args.polarization}:{args.relative_orbit}:{args.direction}",
                 "valid_fraction": float(valid.sum() / mask.sum()), "features": features,
                 "level": "L1-GRD-derived", "input_sha256": file_hash(args.input)}
        scene_dir = root / "scenes" / hashlib.sha256(key.encode()).hexdigest()[:20]
        scene_dir.mkdir(parents=True, exist_ok=True)
        write_mask(scene_dir / "mask.tif", valid, ice, affine, config)
        save(scene_dir / "scene.json", frame)
        db.execute("INSERT INTO frames VALUES (?,?)", (key, json.dumps(frame)))
        db.commit()
        export(db, root, config, {"status": "ok", "processed": 1, "sar_experimental": True})
    finally:
        db.close()
    return 0


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Kara Sea L1/L2 ice monitoring research pilot")
    parser.add_argument("--config", default="config/kara.json")
    parser.add_argument("--output", default="runs/kara")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ["search", "run", "watch"]:
        command = sub.add_parser(name)
        command.add_argument("--start")
        command.add_argument("--end")
        if name == "search":
            command.add_argument("--sensor", choices=["s1", "s2"], default="s2")
        if name == "watch":
            command.add_argument("--interval-minutes", type=float, default=30)
            command.add_argument("--sar-gpt", help="Opt in to sequential SAR processing; SNAP executable")
            command.add_argument("--sar-graph", default="snap/s1_ocean.xml")
            command.add_argument("--sar-max-scenes", type=int, default=2)
    download = sub.add_parser("download-s1")
    download.add_argument("product_id")
    download.add_argument("destination")
    sar = sub.add_parser("ingest-sar")
    sar.add_argument("--input", required=True)
    sar.add_argument("--datetime", required=True)
    sar.add_argument("--product-id", required=True)
    sar.add_argument("--mode", choices=["IW", "EW"], required=True)
    sar.add_argument("--polarization", choices=["HH", "HV", "VV", "VH"], required=True)
    sar.add_argument("--band", type=int, default=1)
    sar.add_argument("--relative-orbit", type=int, required=True)
    sar.add_argument("--direction", choices=["ascending", "descending"], required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.command == "download-s1":
        print(download_s1(args.product_id, args.destination))
        return 0
    config, ocean = configuration(args.config)
    if args.command == "search":
        start, end = dates(args, config)
        finder = s1_search if args.sensor == "s1" else s2_search
        items = finder(config["bbox"], start, end, config["max_items"])
        save(Path(args.output) / f"catalogue-{args.sensor}.json", items)
        print(json.dumps({"sensor": args.sensor, "count": len(items), "start": start, "end": end}))
        return 0
    if args.command == "ingest-sar":
        return run_sar(args, config, ocean)
    if args.command == "watch":
        if args.start or args.end or args.interval_minutes < 1 or args.sar_max_scenes < 1:
            raise ValueError("watch requires a rolling window and interval >= 1 minute")
        while True:
            run_optical(args, config, ocean)
            if args.sar_gpt:
                from .sar_worker import execute
                start, end = dates(args, config)
                execute(config, ocean, args.output, start, end, args.sar_gpt,
                        args.sar_graph, args.sar_max_scenes)
            time.sleep(args.interval_minutes * 60)
    return run_optical(args, config, ocean)


if __name__ == "__main__":
    sys.exit(main())
