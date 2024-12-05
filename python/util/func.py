import numpy as np

import torch
from torchvision import transforms

def segment_image(model, image):
    # モデルをGPUに移動（もし利用可能なら）
    device = next(model.parameters()).device
    model.eval()  # モデルを評価モードにする

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
    ])

    image_tensor = transform(image).unsqueeze(0).to(device)  # 入力をデバイスに移動

    with torch.no_grad():
        output = model(image_tensor)
    
    mask = output.argmax(dim=1).squeeze().cpu().numpy()
    return mask


def apply_color_map(mask, color_map):
    colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_idx, color in enumerate(color_map):
        colored_mask[mask == class_idx] = color
    return colored_mask