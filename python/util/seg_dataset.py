import os
import cv2
import numpy as np

import torch
from torch.utils.data import Dataset as BaseDataset

from torchvision import transforms as T

class Dataset(BaseDataset):
  """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
  
  def __init__(
      self,
      images_dir,
      masks_dir,
      classes=None,
      augmentation=None,
      preprocessing=None,
  ):
    self.ids = os.listdir(images_dir)
    self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
    self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]

    self.CLASSES = ['sky', 'building', 'pole', 'road', 'pavement', 
                    'tree', 'signsymbol', 'fence', 'car', 
                    'pedestrian', 'bicyclist', 'unlabelled']

    self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

    self.augmentation = augmentation
    self.preprocessing = preprocessing

    self.mean = [0.485, 0.456, 0.406]
    self.std = [0.229, 0.224, 0.225]
  
  def __getitem__(self, i):
    # Read Data
    image = cv2.imread(self.images_fps[i])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(self.masks_fps[i], 0)

    if self.augmentation:
      sample = self.augmentation(image=image, mask=mask)
      image, mask = sample['image'], sample['mask']
    
    if self.preprocessing:
      sample = self.preprocessing(image=image, mask=mask)
      image, mask = sample['image'], sample['mask']
    
    t = T.Compose([T.ToTensor(), T.Normalize(self.mean, self.std)])
    image = t(image)
    mask = torch.from_numpy(mask).long()

    return image, mask
  
  def __len__(self):
    return len(self.ids)


class VideoDataset(BaseDataset):
  """Video Dataset. Read frames from a video file and apply augmentations and preprocessing transformations.

  Args:
      video_path (str): path to the video file
      class_values (list): values of classes to extract from segmentation mask
      augmentation (albumentations.Compose): data transformation pipeline
      preprocessing (albumentations.Compose): data preprocessing
  """
  
  def __init__(self, video_path, classes=None, augmentation=None, preprocessing=None):
      self.video_path = video_path
      self.augmentation = augmentation
      self.preprocessing = preprocessing
      self.CLASSES = ['sky', 'building', 'pole', 'road', 'pavement', 
                      'tree', 'signsymbol', 'fence', 'car', 
                      'pedestrian', 'bicyclist', 'unlabelled']
      self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
      
      # Open the video file
      self.cap = cv2.VideoCapture(video_path)
      self.frames = []
      
      # Extract frames from the video
      while self.cap.isOpened():
          ret, frame = self.cap.read()
          if not ret:
              break
          self.frames.append(frame)
      
      self.cap.release()
      
  def __getitem__(self, i):
      # Read a frame
      image = self.frames[i]
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      
      # Create a dummy mask (you can replace this with your own logic)
      mask = np.zeros(image.shape[:2], dtype=np.uint8)
      
      # Apply augmentations
      if self.augmentation:
          sample = self.augmentation(image=image, mask=mask)
          image, mask = sample['image'], sample['mask']
      
      # Apply preprocessing
      if self.preprocessing:
          sample = self.preprocessing(image=image, mask=mask)
          image, mask = sample['image'], sample['mask']
      
      mask = torch.from_numpy(mask).long()
      
      return image, mask
      
  def __len__(self):
      return len(self.frames)