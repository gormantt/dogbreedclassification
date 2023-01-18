#Author: Tim Gorman
#Date: 2023/01/18

from __future__ import absolute_import
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import base64
import numpy
import io
import json
from PIL import Image

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def model_fn(model_dir):
    device = torch.device('cpu')
    model = Net()
    with open(os.path.join(model_dir, "dogbreed_resnet50.pt"), "rb") as f:
        model.load_state_dict(torch.load(f, map_location = device))
    model.eval()
    return model

def Net():
    model = models.resnet50(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False   

    num_features=model.fc.in_features
    model.fc = nn.Sequential(
                   nn.Linear(num_features, 133))
    return model

def input_fn(request_body, request_content_type):
    assert request_content_type=='application/json'
    device = torch.device('cpu')
    data = json.loads(request_body)['inputs']
    data = Image.open(io.BytesIO(base64.b64decode(base64.decodebytes(data.encode('ascii')))))
    transform = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
    data = transform(data).unsqueeze(0)
    data.to(device)
    #data = torch.tensor(data, dtype=torch.float32, device=device)
    return data

def predict_fn(input_object, model):
    with torch.no_grad():
        device = torch.device('cpu')
        model = model.to(device)
        input_data = input_object.to(device)
        model.eval()
        prediction = model(input_data)
    return prediction

# def output_fn(predictions, content_type):
#     assert content_type == 'application/json'
#     res = predictions.cpu().numpy().tolist()
#     return json.dumps(res)