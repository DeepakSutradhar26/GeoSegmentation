import cv2
import torch
import numpy as np
import geopandas as gpd
from torch.utils.data import Dataset

import config

class GeoDataset(Dataset):
    def __init__(self, data_paths):
        self.data_paths = data_paths

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        image_path = self.data_paths[idx][0]
        geojson_path = self.data_paths[idx][1]

        # Load image
        image = cv2.imread(image_path)
        H, W = image.shape[:2]

        # Load geojson
        gdf = gpd.read_file(geojson_path)

        # Lower/Upper bound
        minx, miny, maxx, maxy = gdf.total_bounds

        # Fill the mask
        def geo_to_pixel(x, y):
            px = (x - minx) / (maxx - minx) * W
            py = (maxy - y) / (maxy - miny) * H
            return int(px), int(py)
        
        mask = np.zeros((H, W), dtype=np.uint8)

        for geom in gdf.geometry:
            polys = [geom]

            for poly in polys:
                coords = np.array(
                    [geo_to_pixel(x, y) for x, y in poly.exterior.coords],
                    dtype=np.int32
                )

                cv2.fillPoly(mask, [coords], 1)

        # Resize for faster calculations
        image_512 = cv2.resize(
            image, 
            (config.TILE_SIZE, config.TILE_SIZE), 
            interpolation=cv2.INTER_LINEAR
            )
        mask_512 = cv2.resize(
            mask, 
            (config.TILE_SIZE, config.TILE_SIZE), 
            interpolation=cv2.INTER_NEAREST
            )
        
        # BGR -> RGB
        image_512 = cv2.cvtColor(image_512, cv2.COLOR_BGR2RGB)

        # Convert to tensors
        image_512 = torch.tensor(image_512, dtype=torch.float32).permute(2, 0, 1) / 255.0
        mask_512  = torch.tensor(mask_512,  dtype=torch.float32)
        
        return image_512, mask_512
    

class DroneDataset(Dataset):
    def __init__(self, data_paths):
        self.data_paths = data_paths

    def __len__(self):
        return len(self.data_paths)
    
    def __getitem__(self, idx):
        image_path = self.data_paths[idx][0]
        mask_path = self.data_paths[idx][1]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        binary_mask = (mask > 4).astype('uint8')

        # Resize for faster calculations
        image_512 = cv2.resize(
            image, 
            (config.TILE_SIZE, config.TILE_SIZE), 
            interpolation=cv2.INTER_LINEAR
            )
        mask_512 = cv2.resize(
            binary_mask, 
            (config.TILE_SIZE, config.TILE_SIZE), 
            interpolation=cv2.INTER_NEAREST
            )

        image_512 = torch.tensor(image_512, dtype=torch.float32).permute(2, 0, 1) / 255.0
        mask_512  = torch.tensor(mask_512,  dtype=torch.float32)

        return image_512, mask_512

