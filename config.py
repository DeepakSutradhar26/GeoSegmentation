import torch
import kagglehub

# Hyperparameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 20
BATCH_SIZE = 8
NUM_WORKERS = 2
LR = 1e-5
WEIGHT_DECAY = 1e-6

DATA_PATH = kagglehub.dataset_download("deep262003/geospace-features")