import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

def load_point_cloud_from_bin(bin_file_path):
    point_cloud = np.fromfile(bin_file_path, dtype=np.float32).reshape(-1, 4)  # Assuming each point has x, y, z, intensity
    return point_cloud[:, :3]  # Extract only x, y, z coordinates

def load_labels_from_file(label_file_path):
    labels = np.fromfile(label_file_path, dtype=np.uint32)
    return labels

class RangeProjection:
    def __init__(self, points, labels, proj_H=64, proj_W=1024, fov_up=25.0, fov_down=-22.5):
        self.points = points
        self.labels = labels
        self.proj_H = proj_H
        self.proj_W = proj_W
        self.fov_up = fov_up
        self.fov_down = fov_down

    def project_points(self):
        fov_up = self.fov_up / 180.0 * np.pi
        fov_down = self.fov_down / 180.0 * np.pi
        fov = abs(fov_down) + abs(fov_up)

        depth = np.linalg.norm(self.points, 2, axis=1)

        scan_x = self.points[:, 0]
        scan_y = self.points[:, 1]
        scan_z = self.points[:, 2]

        pitch = np.arcsin(np.divide(scan_z, depth, out=np.zeros_like(scan_z), where=depth!=0))

        yaw = -np.arctan2(scan_y, scan_x)

        proj_x = 0.5 * (yaw / np.pi + 1.0)
        proj_y = 1.0 - (pitch + abs(fov_down)) / fov

        proj_x *= self.proj_W
        proj_y *= self.proj_H

        proj_x = np.floor(proj_x)
        proj_x = np.minimum(self.proj_W - 1, proj_x)
        proj_x = np.maximum(0, proj_x).astype(np.int32)

        proj_y = np.floor(proj_y)
        proj_y = np.minimum(self.proj_H - 1, proj_y)
        proj_y = np.maximum(0, proj_y).astype(np.int32)

        return proj_x, proj_y

# Define color palette
color_palette = {
    0: {"color": [0, 0, 0],  "name": "void"},
    1: {"color": [108, 64, 20],   "name": "dirt"},
    3: {"color": [0, 102, 0],   "name": "grass"},
    4: {"color": [0, 255, 0],  "name": "tree"},
    5: {"color": [0, 153, 153],  "name": "pole"},
    6: {"color": [0, 128, 255],  "name": "water"},
    7: {"color": [0, 0, 255],  "name": "sky"},
    8: {"color": [255, 255, 0],  "name": "vehicle"},
    9: {"color": [255, 0, 127],  "name": "object"},
    10: {"color": [64, 64, 64],  "name": "asphalt"},
    12: {"color": [255, 0, 0],  "name": "building"},
    15: {"color": [102, 0, 0],  "name": "log"},
    17: {"color": [204, 153, 255],  "name": "person"},
    18: {"color": [102, 0, 204],  "name": "fence"},
    19: {"color": [255, 153, 204],  "name": "bush"},
    23: {"color": [197,197,197],  "name": "concrete"},
    27: {"color": [41, 121, 255],  "name": "barrier"},
    31: {"color": [134, 255, 239],  "name": "puddle"},
    33: {"color": [99, 66, 34],  "name": "mud"},
    34: {"color": [110, 22, 138],  "name": "rubble"}
}

# Load point cloud from BIN file
bin_file_path = "/path/to/000104.bin"
points = load_point_cloud_from_bin(bin_file_path)

# Load labels from LABEL file
label_file_path = "/path/to/000104.label"
labels = load_labels_from_file(label_file_path)

# Project points to range image
projection = RangeProjection(points, labels)
proj_x, proj_y = projection.project_points()

# Create range image with colors
range_image_color = np.zeros((projection.proj_H, projection.proj_W, 3), dtype=np.uint8)
for x, y, label in zip(proj_x, proj_y, labels):
    if 0 <= x < projection.proj_W and 0 <= y < projection.proj_H:
        color = color_palette.get(label, {"color": [0, 0, 0]})["color"]
        range_image_color[y, x] = color

# Visualize range image with colors
plt.imshow(range_image_color)
plt.title('Projected Range Image with Labels')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
