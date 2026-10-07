import os
import json
import cv2
import numpy as np

class ZoneManager:
    """
    Stage 1: Polygon Restricted Zone Manager.
    Supports normalized ratio coordinates (0.0 to 1.0) and pixel coordinates,
    making it resolution-independent across 320x240, 720p, 1080p videos.
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
                print(f"[ZoneManager] Could not load zones file: {e}")
        return {
            "cash_counter": [[0.35, 0.20], [0.98, 0.20], [0.98, 0.95], [0.35, 0.95]]
        }

    def save_zones(self):
        with open(self.config_path, "w") as f:
            json.dump(self.zones, f, indent=4)
        print(f"[ZoneManager] Saved zones to {self.config_path}")

    def add_zone(self, zone_name, polygon_points):
        self.zones[zone_name] = polygon_points
        self.save_zones()

    def _get_pixel_polygon(self, zone_name, frame_width=320, frame_height=240):
        if zone_name not in self.zones or len(self.zones[zone_name]) < 3:
            return np.array([], dtype=np.int32)
        pts = self.zones[zone_name]
        pixel_pts = []
        for p in pts:
            px = int(p[0] * frame_width) if (isinstance(p[0], (float, int)) and p[0] <= 1.0) else int(p[0])
            py = int(p[1] * frame_height) if (isinstance(p[1], (float, int)) and p[1] <= 1.0) else int(p[1])
            pixel_pts.append([px, py])
        return np.array(pixel_pts, dtype=np.int32)

    def is_inside_zone(self, bbox, zone_name="cash_counter", frame_width=320, frame_height=240):
        poly_pts = self._get_pixel_polygon(zone_name, frame_width, frame_height)
        if len(poly_pts) < 3:
            return True

        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        bottom_y = y2
        
        pts_to_test = [
            (center_x, center_y),
            (center_x, bottom_y),
            (x1, y1), (x2, y1), (x1, y2), (x2, y2)
        ]
        
        for pt in pts_to_test:
            if cv2.pointPolygonTest(poly_pts, (float(pt[0]), float(pt[1])), measureDist=False) >= 0:
                return True
                
        return False

    def draw_zones(self, frame):
        h, w, _ = frame.shape
        for zone_name, _ in self.zones.items():
            poly_pts = self._get_pixel_polygon(zone_name, w, h)
            if len(poly_pts) >= 3:
                cv2.polylines(frame, [poly_pts], isClosed=True, color=(0, 255, 255), thickness=1 if w <= 400 else 2)
                
                x, y = poly_pts[0]
                cv2.putText(frame, f"ZONE: {zone_name}", (x, max(y - 5, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35 if w <= 400 else 0.5, (0, 255, 255), 1)
        return frame
