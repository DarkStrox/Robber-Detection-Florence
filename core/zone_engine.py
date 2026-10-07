import os
import json
import cv2
import numpy as np

class ZoneEngine:
    """
    Polygon Restricted Zone Manager.
    Allows defining polygon zones for cash registers / counter areas
    and checking if person bounding boxes intersect with the zone.
    """
    def __init__(self, config_path="zones.json"):
        self.config_path = config_path
        self.zones = self.load_zones()

    def load_zones(self):
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"[ZoneEngine] Could not load zones file: {e}")
        return {}

    def save_zones(self):
        with open(self.config_path, "w") as f:
            json.dump(self.zones, f, indent=4)
        print(f"[ZoneEngine] Saved zones to {self.config_path}")

    def add_zone(self, zone_name, polygon_points):
        self.zones[zone_name] = polygon_points
        self.save_zones()

    def is_inside_zone(self, bbox, zone_name="cash_counter"):
        if zone_name not in self.zones or len(self.zones[zone_name]) < 3:
            return False

        # Calculate center bottom of person bbox (feet/standing position)
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2.0
        bottom_y = y2

        poly_pts = np.array(self.zones[zone_name], dtype=np.int32)
        dist = cv2.pointPolygonTest(poly_pts, (float(center_x), float(bottom_y)), measureDist=False)
        
        # dist >= 0 means point is inside or on boundary
        return dist >= 0

    def draw_zones(self, frame):
        for zone_name, pts in self.zones.items():
            if len(pts) >= 3:
                poly_pts = np.array(pts, dtype=np.int32)
                cv2.polylines(frame, [poly_pts], isClosed=True, color=(0, 255, 255), thickness=2)
                
                # Draw label
                x, y = pts[0]
                cv2.putText(frame, f"ZONE: {zone_name}", (x, max(y - 10, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        return frame
