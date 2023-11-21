import os
import random
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import base64
from flask import Flask, request, render_template, flash, redirect

from model_architecture.architecture import resnet
from detect_emotion import proc_predict
from io import BytesIO
    

def plot_to_base64(fig):
    image_stream = BytesIO()
    fig.savefig(image_stream, format='png')
    image_base64 = base64.b64encode(image_stream.getvalue()).decode('utf-8')
    plt.close(fig)
    return image_base64


model_path = r"C:\\Users\\hp\\OneDrive\\Documents\\FirstApp\\model.pt"
torch.manual_seed(9)
model = resnet()
model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
model.eval()


app = Flask(__name__)
app.secret_key = 'myskey'

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/upload', methods=['POST'])
def get_file():

    file = request.files['image']

    image_stream = BytesIO(file.read())
    image_array = np.frombuffer(image_stream.getvalue(), dtype=np.uint8)
    image_cv2 = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    out_fig = proc_predict.show_res(image_cv2, model)
    image_base64 = plot_to_base64(out_fig)

    return render_template('result.html', image_base64=image_base64)
    

if __name__ == "__main__":
    app.run(debug=True, threaded=False)