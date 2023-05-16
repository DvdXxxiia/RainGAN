import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)
import cv2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = UNET(in_channels=3, out_channels=1).to(DEVICE)
load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)
x=cv2.imread('data/test_images/1.png')
pred= torch.sigmoid(model(x))
cv2.imshow('',pred)
cv2.waitKey()
