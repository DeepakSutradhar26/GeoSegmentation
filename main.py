import config
from train import train_model
from models.segformer import SegFormerBinary

model = SegFormerBinary().to(config.DEVICE)
train_model(model)