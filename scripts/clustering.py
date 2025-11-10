# Merge detections across images using DBSCAN to estimate unique seals

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from pyproj import Transformer

def compute_unique_counts(grouped_coords, max_counts):
    transformer = Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True)
    folder_summary_records = []

    for (loc, date), coords_data in grouped_coords.items():
        if not coords_data:
            folder_summary_records.append({"survey_location": loc, "date": date, "unique_pinniped_count": 0})
            continue

        coords, _, _ = zip(*coords_data)
        coords_xy = np.array([transformer.transform(lon, lat) for lat, lon in coords])

        if len(coords_xy) == 1:
            count = 1
        else:
            nn = NearestNeighbors(n_neighbors=2).fit(coords_xy)
            distances, _ = nn.kneighbors(coords_xy)
            median_dist = np.median(distances[:, 1])
            eps = np.clip(median_dist * 1.75, 1.7, 3.65)
            clustering = DBSCAN(eps=eps, min_samples=1).fit(coords_xy)
            count = len(set(clustering.labels_))

        count = max(count, max_counts.get((loc, date), 0))
        folder_summary_records.append({
            "survey_location": loc,
            "date": date,
            "unique_pinniped_count": count
        })

    return folder_summary_records
