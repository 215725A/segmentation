import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import segmentation_models_pytorch as smp

from util.seg_dataset import Dataset
from util.setup import setup
from util.yaml import load_yaml
from util.augmentation import get_training_augmentation, get_validation_augmentation
from util.learning import fit
from util.plot import plot_loss, plot_score, plot_acc

import argparse

def main(config_path):
  max_lr = 1e-3
  epoch = 100
  weight_decay = 1e-4
  
  # Setting Image's Classes
  CLASSES = ['sky', 'building', 'pole', 'road', 'pavement', 'tree', 'signsymbol', 'fence', 'car', 'pedestrian', 'bicyclist', 'unlabelled']

  config = load_yaml(config_path)
  model_name = config['model_name']
  output_dir = config['output_dir']

  model = smp.Unet(model_name, encoder_weights='imagenet', classes=len(CLASSES), activation=None)

  x_datas, y_datas = setup()

  train_dataset = Dataset(
    images_dir=x_datas['x_train_dir'],
    masks_dir=y_datas['y_train_dir'],
    augmentation=get_training_augmentation(),
    classes=CLASSES,
  )

  valid_dataset = Dataset(
    images_dir=x_datas['x_valid_dir'],
    masks_dir=y_datas['y_valid_dir'],
    augmentation=get_validation_augmentation(),
    classes=CLASSES
  )

  train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
  valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)

  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay)
  sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epoch, steps_per_epoch=len(train_loader))

  history = fit(epoch, model, train_loader, valid_loader, criterion, optimizer, sched, len(CLASSES), output_dir)


  plot_loss(history, output_dir)
  plot_score(history, output_dir)
  plot_acc(history, output_dir)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-c', '--config', help='Load YML')

  args = parser.parse_args()

  main(args.config)