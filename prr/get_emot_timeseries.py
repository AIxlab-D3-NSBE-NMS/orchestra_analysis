import cv2
from deepface import DeepFace
import numpy as np
from tqdm import tqdm
import pandas as pd
import PIL
from PIL import Image
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import seaborn as sns



def read_image(fullpath):
    img = Image.open(fullpath)
    return np.array(img.convert('RGB'))

def get_emotions_from_frame(img, crop = None):
    face = DeepFace.analyze(img,
                            actions='emotion',
                            enforce_detection=False,
                            detector_backend='retinaface',
                            align=True,
                            expand_percentage=0,
                            silent=False,
                            anti_spoofing=False)
    return face


myface = "vlcsnap-2026-03-19-18h46m22s488.png"
img = read_image(myface)
face = get_emotions_from_frame(img)

print(face)
