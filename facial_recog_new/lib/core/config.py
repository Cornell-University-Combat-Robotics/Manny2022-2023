B_TRAIN_IMG_PATH = "/home/firmware/crc_fa22/Manny2022-2023/facial_recog_new/lib/dataset/Blaze1"
S_TRAIN_IMG_PATH = "/home/firmware/crc_fa22/Manny2022-2023/facial_recog_new/lib/dataset/Seb1"
R_TRAIN_IMG_PATH = "/home/firmware/crc_fa22/Manny2022-2023/facial_recog_new/lib/dataset/Rick1"

INPUT_IMAGE_HEIGHT = 256
INPUT_IMAGE_WIDTH = 256

BATCH_SIZE = 64

# PIN_MEMORY = True if DEVICE == "cuda" else False
PIN_MEMORY = False

INIT_LR = 1e-3

NUM_EPOCHS = 1

PLOT_PATH = "/home/firmware/crc_fa22/Manny2022-2023/facial_recog_new/output/losses.png"
MODEL_PATH = "/home/firmware/crc_fa22/Manny2022-2023/facial_recog_new/output/face_recog.pth"