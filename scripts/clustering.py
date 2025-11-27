# Merge detections across images using DBSCAN to estimate unique seals

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from pyproj import Transformer
from collections import defaultdict
import math


def compute_unique_counts(grouped_coords, max_counts, eps_min=3.0, eps_max=20.0, scale_factor=1.6):
    """
    Estimate unique pinniped counts per (location, date) using spatial clustering of
    all detection-derived coordinates across images.

    Inputs:
      grouped_coords: dict keyed by (location, date) -> list[(lat, lon)]
      max_counts: dict keyed by (location, date) -> highest single-image pinniped_count
      eps_min, eps_max: clamp range for adaptive DBSCAN radius (meters)
      scale_factor: multiplier on median nearest-neighbor distance

    Logic:
      1. Project lat/lon to Web Mercator meters for Euclidean distance.
      2. If only one point -> unique_count = 1 (then enforce lower bound).
      3. Compute nearest-neighbor distances (k=2) and take median of distances[:,1].
      4. eps = clip(median * scale_factor, eps_min, eps_max). Fallback to mid if NN fails.
      5. Run DBSCAN (min_samples=1) so isolated points remain distinct.
      6. Count clusters = number of unique labels.
      7. Enforce lower bound: unique_count = max(cluster_count, max_counts[(loc,date)]).
      8. Light post-merge: If cluster centroids are closer than eps/2, merge them to avoid over-splitting passes.

    Returns:
      List of dicts: {survey_location, date, unique_count}
    """
    transformer = Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True)
    results = []

    for (loc, date), points in grouped_coords.items():
        lower_bound = max_counts.get((loc, date), 0)

        if not points:
            results.append({"survey_location": loc, "date": date, "unique_count": lower_bound})
            continue

        # Project to meters (lon, lat order)
        coords_xy = np.array([transformer.transform(lon, lat) for lat, lon in points], dtype=float)

        if len(coords_xy) == 1:
            cluster_count = 1
        else:
            try:
                nn = NearestNeighbors(n_neighbors=2).fit(coords_xy)
                distances, _ = nn.kneighbors(coords_xy)
                median_dist = float(np.median(distances[:, 1]))
                eps = float(np.clip(median_dist * scale_factor, eps_min, eps_max))
            except Exception:
                eps = (eps_min + eps_max) / 2.0  # fallback

            clustering = DBSCAN(eps=eps, min_samples=1).fit(coords_xy)
            labels = clustering.labels_

            # Compute cluster centroids
            centroids = []
            for lbl in set(labels):
                member_pts = coords_xy[labels == lbl]
                centroids.append(member_pts.mean(axis=0))
            centroids = np.array(centroids)

            # Optional post-merge of very close clusters (< eps/2)
            if len(centroids) > 1:
                merged = []
                used = set()
                for i in range(len(centroids)):
                    if i in used:
                        continue
                    group = [i]
                    for j in range(i + 1, len(centroids)):
                        if j in used:
                            continue
                        d = np.linalg.norm(centroids[i] - centroids[j])
                        if d < eps / 2.0:
                            group.append(j)
                            used.add(j)
                    used.update(group)
                    merged.append(group)
                cluster_count = len(merged)
            else:
                cluster_count = 1

        unique_count = max(cluster_count, lower_bound)
        results.append({"survey_location": loc, "date": date, "unique_count": int(unique_count)})

    return results
