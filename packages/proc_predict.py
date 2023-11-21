import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import cv2
import matplotlib.pyplot as plt


def show_res(opic1, model):
    pic1 = cv2.cvtColor(opic1, cv2.COLOR_BGR2GRAY)
    pic1 = cv2.resize(pic1, (48, 48), interpolation=cv2.INTER_LINEAR)
    pic1 = pic1.astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(3, 3))
    pic1 = clahe.apply(pic1)
    pic1 = cv2.fastNlMeansDenoising(pic1, None, h=10, templateWindowSize=7, searchWindowSize=21)
    pic1 = pic1/255

    x = torch.tensor(pic1, dtype=torch.float32)
    x = x.unsqueeze(0).unsqueeze(0)
    
    with torch.no_grad():
        out = F.softmax(model(x), dim=1)[0]
    out = np.array(out)
    out = [round(k*100, 2) for k in out]


    emo_map = {
        'Angry': 0,
        'Disgust': 1,
        'Fear': 2,
        'Happy': 3,
        'Sad': 4,
        'Surprise': 5,
        'Neutral': 6,
    }


    results = [(e, p) for e, p in zip(emo_map.keys(), out)]
    results = sorted(results, key=lambda k: k[1], reverse=True)
    results = [k for k in results if k[1]>0]
    if len(results)>3:
        results = results[:3]


    fig, ax = plt.subplots()
    ax.imshow(opic1)
    title = ""
    for em, i in results:
        title = title + em + ": " + str(i) + "%   "
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    
    return fig
