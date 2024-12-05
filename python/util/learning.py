import torch
import time

import numpy as np

from tqdm import tqdm

from .verification import mIoU, pixel_accuracy
from .setup import create_save_dir

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def fit(epochs, model, train_loader, val_loader, criterion, optimizer, scheduler, n_classes, output_dir, patch=False):
  torch.cuda.empty_cache()
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  train_losses = []
  test_losses = []
  val_iou = []
  val_acc = []
  train_iou = []
  train_acc = []
  lrs = []

  min_loss = np.inf
  decrease = 1
  not_improve = 0

  model.to(device)

  fit_time = time.perf_counter()

  for e in range(epochs):
    since = time.perf_counter()
    running_loss = 0
    iou_score = 0
    accuracy = 0

    # Training Loop
    model.train()

    for data in tqdm(train_loader):
      image_tiles, mask_tiles = data
      
      if patch:
        _, _, c, h, w = image_tiles.size()

        image_tiles = image_tiles.view(-1, c, h, w)
        mask_tiles = mask_tiles.view(-1, h, w)
      
      image = image_tiles.to(device)
      mask = mask_tiles.to(device)

      # Forward
      output = model(image)
      loss = criterion(output, mask)

      # Evaluation metrics
      iou_score += mIoU(output, mask, n_classes)
      accuracy += pixel_accuracy(output, mask)

      # Backward
      loss.backward()
      optimizer.step() # Update Weight
      optimizer.zero_grad() # Reset Gradient

      # Step the Learning Rate
      lrs.append(get_lr(optimizer))
      scheduler.step()

      running_loss += loss.item()
    
    else:
      model.eval()
      test_loss = 0
      test_accuracy = 0
      val_iou_score = 0

      # Validation Loop
      with torch.no_grad():
        for data in tqdm(val_loader):
          # Reshape to 9 Patches From Single Image, Delete Batch Size
          image_tiles, mask_tiles = data

          if patch:
            _, _, c, h, w = image_tiles.view(-1, c, h, w)

            image_tiles = image_tiles.view(-1, c, h, w)
            mask_tiles = mask_tiles.view(-1, h, w)
          
          image = image_tiles.to(device)
          mask = mask_tiles.to(device)

          output = model(image)

          # Evaluation Metrics
          val_iou_score += mIoU(output, mask, n_classes)
          test_accuracy += pixel_accuracy(output, mask)

          # Loss
          loss = criterion(output, mask)
          test_loss += loss.item()

    # Calculate mean for each batch
    train_losses.append(running_loss/len(train_loader))
    test_losses.append(test_loss/len(val_loader))

    losses = test_loss/len(val_loader)

    if min_loss > losses:
      print(f'Loss Decreasing... {min_loss:.3f} >> {losses:.3f}')
      min_loss = losses
      decrease += 1
    
    if (test_loss/len(val_loader)) > min_loss:
      not_improve += 1
      min_loss = (test_loss/len(val_loader))
      print(f'Loss Not Decrease for {not_improve} time')
      if not_improve == 20:
        print('Loss not decrease for 20 times, Save Best Model')
        output_path = f'../outputs/pts/{output_dir}/Learning_best.pt'
        torch.save(model.state_dict(), output_path)

    if (e+1) % 5 == 0:
      print('Saving Model...')
      output_path = f'../outputs/pts/{output_dir}/Learning_{e+1:03d}.pt'
      create_save_dir(output_path)
      torch.save(model.state_dict(), output_path)

    # IoU
    val_iou.append(val_iou_score/len(val_loader))
    val_acc.append(test_accuracy/len(val_loader))
    train_iou.append(iou_score/len(train_loader))
    train_acc.append(accuracy/len(train_loader))

    end_time = time.perf_counter()

    print(f"""
          Epoch: {e+1}/{epochs},
          Train Loss: {running_loss/len(train_loader):.3f},
          Val Loss: {test_loss/len(val_loader):.3f},
          Train mIoU: {iou_score/len(train_loader):.3f},
          Val mIoU: {val_iou_score/len(val_loader):.3f},
          Train Acc: {accuracy/len(train_loader):.3f},
          Val Acc: {test_accuracy/len(val_loader):.3f},
          Time: {end_time - since},
          Next: {e+1}/{epochs} epoch
          """)
  
  history = {'train_loss': train_losses, 'val_loss': test_losses,
             'train_miou': train_iou, 'val_miou': val_iou,
             'train_acc': train_acc, 'val_acc': val_acc,
             'lrs': lrs
            }

  exec_end = time.perf_counter()
  print(f'Total time: {(exec_end - fit_time)/60:.2f} m')

  return history