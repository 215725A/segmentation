import os

import cv2
import argparse
import numpy as np

from util.setup import setup, set_model, create_save_dir
from util.seg_dataset import VideoDataset
from util.augmentation import get_validation_augmentation
from util.func import segment_image, apply_color_map
from util.yaml import load_yaml

def main(config_path):
  # Setting Image's Classes
  CLASSES = ['sky', 'building', 'pole', 'road', 'pavement', 'tree', 'signsymbol', 'fence', 'car', 'pedestrian', 'bicyclist', 'unlabelled']

  color_map = [
        [135, 206, 235],  # sky
        [128, 0, 0],      # building
        [0, 128, 0],      # pole
        [128, 128, 128],  # road
        [255, 255, 255],  # pavement
        [0, 128, 0],      # tree
        [255, 0, 0],      # signsymbol
        [128, 128, 0],    # fence
        [0, 0, 255],      # car
        [255, 192, 203],  # pedestrian
        [0, 255, 255],    # bicyclist
        [0, 0, 0]         # unlabelled
    ]

  x_datas, y_datas = setup()

  video_path = '../videos/moviename.mp4'
  
  config = load_yaml(config_path)
  model_name = config['model_name']
  output_dir = config['output_dir']

  model_dir = f'../outputs/pts/{output_dir}'

  models = sorted(os.listdir(model_dir))

  video_dataset = VideoDataset(video_path, classes=CLASSES, augmentation=get_validation_augmentation())

  cap = cv2.VideoCapture(video_path)
  fps = cap.get(cv2.CAP_PROP_FPS)
  height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  fourcc = cv2.VideoWriter_fourcc(*'mp4v')


  for mod in models:
    output_path_1 = f'../videos/{output_dir}/{os.path.splitext(os.path.basename(mod))[0]}/result_blur.mp4'
    output_path_2 = f'../videos/{output_dir}/{os.path.splitext(os.path.basename(mod))[0]}/result_blur_colormap.mp4'
    model_path = f'{model_dir}/{mod}'

    create_save_dir(output_path_1)
    
    model = set_model(model_path, model_name)

    video_writer_1 = cv2.VideoWriter(output_path_1, fourcc, fps, (width, height))
    video_writer_2 = cv2.VideoWriter(output_path_2, fourcc, fps, (width, height))

    for i in range(len(video_dataset)):
      image, _ = video_dataset[i]

      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

      blur = cv2.GaussianBlur(image, (5, 5), 0)

      mask = segment_image(model, blur)
      mask = apply_color_map(mask, color_map)

      mask_resized = cv2.resize(mask.astype(np.uint8), (width, height))

      # result = cv2.addWeighted(image, 0.5, cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR), 0.5, 0)
      result = cv2.addWeighted(image, 0.5, mask_resized, 0.5, 0)

      video_writer_1.write(result)
      video_writer_2.write(mask_resized)

    
    print(f"End of write to {output_path_1}, {output_path_2}")
    video_writer_1.release()
    video_writer_2.release()

  cap.release()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-c', '--config', help='Load YML')

  args = parser.parse_args()

  main(args.config)