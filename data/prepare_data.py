import os
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

import config
from data.dataset import GeoDataset, DroneDataset

IMAGE_PATH = os.path.join(config.DATA_PATH, 'dataset/semantic_drone_dataset/original_images')
MASK_PATH = os.path.join(config.DATA_PATH, 'dataset/semantic_drone_dataset/label_images_semantic')

data_paths = []
for img_file in os.listdir(IMAGE_PATH):
    id = img_file.split('.')[0]
    aerial_img_path = os.path.join(IMAGE_PATH, img_file)
    binary_mask_path = os.path.join(MASK_PATH, f'{id}.png')
    data_paths.append([aerial_img_path, binary_mask_path])

train_paths, val_paths = train_test_split(
    data_paths,
    test_size=0.2,
    random_state=42
)

train_dataset = DroneDataset(
    train_paths
)

val_dataset = DroneDataset(
    val_paths
)

train_loader = DataLoader(
    train_dataset,
    batch_size=config.BATCH_SIZE,
    num_workers=config.NUM_WORKERS,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=config.BATCH_SIZE,
    num_workers=config.NUM_WORKERS,
)