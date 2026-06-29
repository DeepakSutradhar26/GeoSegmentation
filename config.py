import torch
import kagglehub

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TILE_SIZE = 512
THRESHOLD = 0.5
EPOCHS = 10
BATCH_SIZE = 8
NUM_WORKERS = 2
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-6

DATA_PATH = kagglehub.dataset_download("bulentsiyah/semantic-drone-dataset")