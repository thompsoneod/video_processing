import os
import torch

from ultralytics import YOLO


available_gpu = [i for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else 'cpu'

def inf_model(media_source, weights,name):
    print(media_source)
    model = YOLO(weights)
    model.predict(source=media_source, imgsz=1280, save_txt=True, project = 'predictions', name=name, exist_ok=True, device=available_gpu)
  
# no USE CASE for this function yet 
# def train_model(media_source, weights):
#     model = YOLO(weights)
#     model.train(data='unk_test.yaml', epochs=30, imgsz=1280, augment=True, device=available_gpu)
    
    #### NEED TO CREATE YAML CREATION FUNCTION, Possible dataset creation function