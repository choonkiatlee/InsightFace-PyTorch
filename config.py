import logging
import os

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # sets device for model and PyTorch tensors
# device = torch.device('cpu')

# Model parameters
im_size = 112
channel = 3
emb_size = 512

# Training parameters
num_workers = 4  # for data-loading; right now, only 1 works with h5py
grad_clip = 5.  # clip gradients at an absolute value of
print_freq = 100  # print training/validation stats  every __ batches
checkpoint = None  # path to checkpoint, None if none

# Data parameters
num_classes = 85742
num_samples = 5822653
DATA_DIR = '/datadrive/data'
# faces_ms1m_folder = 'data/faces_ms1m_112x112'
# faces_ms1m_folder = 'data/ms1m-retinaface-t1'
faces_ms1m_folder = DATA_DIR + '/faces_emore'
path_imgidx = os.path.join(faces_ms1m_folder, 'train.idx')
path_imgrec = os.path.join(faces_ms1m_folder, 'train.rec')
IMG_DIR = DATA_DIR + '/images_batched'
pickle_file = DATA_DIR + '/faces_ms1m_112x112.pickle'

img_batch_size = 128

LFW_DIR = os.path.join(DATA_DIR, 'LFW')

PRETRAINED_DIR = "pretrained_model"


def get_logger():
    logger = logging.getLogger()
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(levelname)s \t%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


logger = get_logger()
