B_TRAIN_IMG_PATH = "/Users/richmjin/Desktop/facial_recog/lib/dataset/Blaze1"
S_TRAIN_IMG_PATH = "/Users/richmjin/Desktop/facial_recog/lib/dataset/Seb1"
INPUT_IMAGE_HEIGHT = 256
INPUT_IMAGE_WIDTH = 256

# batch size < number of all samples
    # pros: requires less data, trains faster
    # cons: less accurate
# determining the optimal batch size requires testing
BATCH_SIZE = 2

# PIN_MEMORY = True if DEVICE == "cuda" else False
PIN_MEMORY = False

# stands for initial learning rate
# traditional default value is 0.1 or 0.01
# (I think) represents how drastically we should change the weights of the model after each training iteraction
INIT_LR = 0.01

# too high could mean over fitting, but too low could mean under fitting
# once again, determining the optimal number of epochs requires testing
NUM_EPOCHS = 10

PLOT_PATH = "/Users/richmjin/Desktop/facial_recog/output/losses.png"
MODEL_PATH = "/Users/richmjin/Desktop/facial_recog/output/face_recog.pth"