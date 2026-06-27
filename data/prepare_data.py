import os
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader

import config
from data.dataset import GeoDataset

IMAGE_PATH = os.path.join(config.DATA_PATH, 'aerial_imagery_pack')

data_paths = []
for img_path in os.listdir(IMAGE_PATH):
    id = img_path.split('.')[0]
    aerial_img_path = os.path.join(IMAGE_PATH, img_path)
    binary_mask_path = os.path.join(IMAGE_PATH, id + '.geojson')
    data_paths.append(aerial_img_path, binary_mask_path)

train_paths, val_paths = train_test_split(
    data_paths,
    test_size=0.2,
    random_state=42
)

train_dataset = GeoDataset(
    train_paths
)

val_dataset = GeoDataset(
    val_paths
)

train_loader = DataLoader(
    train_dataset,
    batch_size=config.BATCH_SIZE,
    num_workers=config.NUM_WORKERS,
    drop_last=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=config.BATCH_SIZE,
    num_workers=config.NUM_WORKERS,
    drop_last=True
)