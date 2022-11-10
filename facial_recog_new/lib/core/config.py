B_TRAIN_IMG_PATH = "/Users/richmjin/Desktop/facial_recog/lib/dataset/Blaze1"
S_TRAIN_IMG_PATH = "/Users/richmjin/Desktop/facial_recog/lib/dataset/Seb1"
R_TRAIN_IMG_PATH = "/Users/richmjin/Desktop/facial_recog/lib/dataset/Rick1"

INPUT_IMAGE_HEIGHT = 256
INPUT_IMAGE_WIDTH = 256

BATCH_SIZE = 64

# PIN_MEMORY = True if DEVICE == "cuda" else False
PIN_MEMORY = False

INIT_LR = 1e-3

NUM_EPOCHS = 10

PLOT_PATH = "/Users/richmjin/Desktop/facial_recog/output/losses.png"
MODEL_PATH = "/Users/richmjin/Desktop/facial_recog/output/face_recog.pth"