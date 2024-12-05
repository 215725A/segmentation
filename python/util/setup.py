import os

import torch

import segmentation_models_pytorch as smp

def setup():
  DATA_DIR = '../data/CamVid'

  if not os.path.exists(DATA_DIR):
    print('Loading data ...')
    os.system('git clone https://github.com/alexgkendall/SegNet-Tutorial ./data')
    print('Done!')
  
  x_train_dir = os.path.join(DATA_DIR, 'train')
  y_train_dir = os.path.join(DATA_DIR, 'trainannot')

  x_valid_dir = os.path.join(DATA_DIR, 'val')
  y_valid_dir = os.path.join(DATA_DIR, 'valannot')

  x_test_dir = os.path.join(DATA_DIR, 'test')
  y_test_dir = os.path.join(DATA_DIR, 'testannot')

  x_datas = {'x_train_dir': x_train_dir, 'x_valid_dir': x_valid_dir, 'x_test_dir': x_test_dir}
  y_datas = {'y_train_dir': y_train_dir, 'y_valid_dir': y_valid_dir, 'y_test_dir': y_test_dir}

  return x_datas, y_datas


def load_model(model_path):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
  model = torch.load(model_path).to(device)
  model.eval()
  return model


def set_model(model_path, model_name):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
  CLASSES = ['sky', 'building', 'pole', 'road', 'pavement', 'tree', 'signsymbol', 'fence', 'car', 'pedestrian', 'bicyclist', 'unlabelled']

  model = smp.Unet(model_name, encoder_weights='imagenet', classes=len(CLASSES), activation=None)
  
  # モデルの重み（state_dict）をロード
  state_dict = torch.load(model_path, map_location=device)
  
  # モデルに重みを適用
  model.load_state_dict(state_dict)
  
  model = model.to(device)
  model.eval()
  
  return model

def create_save_dir(output_path):
  dirname = os.path.dirname(output_path)

  os.makedirs(dirname, exist_ok=True)