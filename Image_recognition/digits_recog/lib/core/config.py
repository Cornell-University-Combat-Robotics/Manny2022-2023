import torch
import os

INPUT_IMAGE_HEIGHT = 28
INPUT_IMAGE_WIDTH = 28
INIT_LR = 0.01
BATCH_SIZE = 500
NUM_EPOCHS = 10

# determine the device to be used for training and evaluation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# determine if we will be pinning memory during data loading
PIN_MEMORY = True if DEVICE == "cuda" else False

BASE_OUTPUT = "output"
PLOT_PATH = os.path.join(BASE_OUTPUT, "losses.png")
MODEL_PATH = os.path.join(BASE_OUTPUT, "digits_recog.pth")