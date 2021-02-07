import torch
BATCH_SIZE = 256
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
HID_DIM = 512
N_HEADS = 8
PF_DIM = 512
N_LAYERS = 3
MAX_LEN = 50
LR = 0.0005
N_EPOCHS = 20
DROPOUT = 0.1
N_EXP = 16
CAPACITY_FACTOR = 1.0
