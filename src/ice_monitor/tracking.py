"""Conservative baseline association; ambiguous pairs do not create velocities."""
import math
from datetime import datetime
from uuid import NAMESPACE_URL, uuid5

import numpy as np
from pyproj import Transformer
from shapely.affinity import translate
from shapely.geometry import shape
from shapely.ops import transform


def instant(value):
    value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if value.tzinfo is None:
        raise ValueError("Timestamps require an explicit timezone")
    return value


def associate(previous, current, elapsed, config):
    """Only unambiguous bidirectional candidates; keep split/merge candidates separate."""
    if not previous or not current:
        return {}, set(), set()
    distance = config["max_speed_m_s"] * elapsed + 2 * config["position_uncertainty_m"]
    candidates = np.zeros((len(previous), len(current)), dtype=bool)
    for i, left in enumerate(previous):
        for j, right in enumerate(current):
            ratio = right.area / max(left.area, 1)
            if not 0.5 <= ratio <= 2 or left.centroid.distance(right.centroid) > distance:
                continue
            moved = translate(left, right.centroid.x-left.centroid.x, right.centroid.y-left.centroid.y)
            iou = moved.intersection(right).area / max(moved.union(right).area, 1)
            candidates[i, j] = iou >= 0.4
    matches = {}
    ambiguous_current, ambiguous_previous = set(), set()
    for i, j in zip(*np.where(candidates)):
        if candidates[i].sum() == 1 and candidates[:, j].sum() == 1:
            matches[int(j)] = int(i)
        else:
            ambiguous_current.add(int(j))
            ambiguous_previous.add(int(i))
    return matches, ambiguous_current, ambiguous_previous


def track(frames, config):
    project = Transformer.from_crs(4326, config["crs"], always_xy=True).transform
    tracks, output, events = {}, [], []
    # Independent sensor/orbit streams avoid interpreting viewing geometry changes as drift.
    streams = {}
    for frame in sorted(frames, key=lambda x: (instant(x["datetime"]), x["id"])):
        now = instant(frame["datetime"])
        stream = frame["stream"]
        prior = streams.get(stream)
        if frame["valid_fraction"] < config["min_valid_fraction"]:
            continue
        if prior and (now - prior["time"]).total_seconds() < config["min_track_gap_minutes"] * 60:
            # Same overpass tiles must not count as independent temporal evidence.
            continue
        current = frame["features"]
        polygons = [transform(project, shape(f["geometry"])) for f in current]
        elapsed = (now - prior["time"]).total_seconds() if prior else 0
        matches, ambiguous, _ = ({}, set(), set())
        if prior and elapsed <= config["max_track_gap_hours"] * 3600:
            matches, ambiguous, _ = associate(prior["polygons"], polygons, elapsed, config)
        ids = []
        for index, (feature, polygon) in enumerate(zip(current, polygons)):
            props = dict(feature["properties"])
            props.update({"observed_at": frame["datetime"], "source_id": frame["id"],
                          "stream": stream, "motion": "unknown", "speed_m_s": None,
                          "association": "unmatched", "grounding": "unverified"})
            eligible = not props["truncated"] and props["confidence"] != "low"
            if index in matches:
                old_index = matches[index]
                track_id = prior["ids"][old_index]
                history = tracks[track_id]
                shift = polygon.centroid.distance(prior["polygons"][old_index].centroid)
                props.update({"association": "baseline_match", "speed_m_s": round(shift / elapsed, 6),
                              "speed_uncertainty_m_s": round(2 * config["position_uncertainty_m"] / elapsed, 6),
                              "motion": "moving_candidate" if shift > 2 * config["position_uncertainty_m"] else "unresolved"})
            else:
                track_id = str(uuid5(NAMESPACE_URL, frame["id"] + ":" + str(index)))
                history = []
                tracks[track_id] = history
                props["association"] = "ambiguous" if index in ambiguous else "unmatched"
                events.append({"type": "association_ambiguous" if index in ambiguous else "newly_observed",
                               "track_id": track_id, "observed_at": frame["datetime"],
                               "requires_review": True})
            history.append({"time": now, "point": polygon.centroid, "eligible": eligible})
            # Stationarity is only a hypothesis; no automatic stamukha classification.
            duration = (now - history[0]["time"]).total_seconds() / 3600
            excursion = max(polygon.centroid.distance(p["point"]) for p in history)
            if (len(history) >= config["stationary_observations"]
                    and duration >= config["stationary_hours"]
                    and excursion <= config["stationary_radius_m"]
                    and all(p["eligible"] for p in history)):
                props["motion"] = "stationary_candidate"
                props["review_reason"] = "Distinguish fast ice, grounded ridge, iceberg, land and association error"
            props.update({"track_id": track_id, "observation_count": len(history)})
            output.append({"type": "Feature", "geometry": feature["geometry"], "properties": props})
            ids.append(track_id)
        streams[stream] = {"time": now, "polygons": polygons, "ids": ids}
    return output, events
