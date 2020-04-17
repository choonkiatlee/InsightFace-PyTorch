import torch

import math
import os
import pickle
import tarfile
import time

import cv2 as cv
import numpy as np
import scipy.stats
import torch
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm

from config import device, LFW_DIR
from data_gen import data_transforms
from utils import align_face, get_central_face_attributes, get_all_face_attributes, draw_bboxes


from models import resnet18, resnet34, resnet50, resnet101, resnet152, ArcMarginModel, MobileFaceNet
from utils import parse_args
from config import emb_size


if __name__ == "__main__":
    args = parse_args()


    print("Loading Model...")
    if args.network == "r101":
        model = resnet101(args)
        model.load_state_dict(torch.load('pretrained_model/r101/insight-face-v3.pt'))
    elif args.network == "mfacenet":
        model = MobileFaceNet(embedding_size=emb_size)
        model.load_state_dict(torch.load('pretrained_model/mfacenet/model_mobilefacenet.pth'))

    model = model.to(device)
    model.eval()


    print("Scripting Model")
    scripted_model = torch.jit.script(model)

    print("Saving Model")
    if args.network == "r101":
        scripted_model.save('pretrained_model/r101/scripted_model.zip')
    elif args.network == "mfacenet":
        scripted_model.save('pretrained_model/mfacenet/scripted_model.zip')

