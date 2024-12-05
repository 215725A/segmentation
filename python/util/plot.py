import matplotlib.pyplot as plt
from .setup import create_save_dir  

def plot_loss(history, output_dir):
  fig_path = f'../images/{output_dir}/loss.png'
  create_save_dir(fig_path)
  plt.plot(history['val_loss'], label='val', marker='o')
  plt.plot( history['train_loss'], label='train', marker='o')
  plt.title('Loss per epoch'); plt.ylabel('loss');
  plt.xlabel('epoch')
  plt.legend(), plt.grid()
  plt.savefig(fig_path, bbox_inches='tight')
  plt.show()


def plot_score(history, output_dir):
  fig_path = f'../images/{output_dir}/score.png'
  create_save_dir(fig_path)
  plt.plot(history['train_miou'], label='train_mIoU', marker='*')
  plt.plot(history['val_miou'], label='val_mIoU',  marker='*')
  plt.title('Score per epoch'); plt.ylabel('mean IoU')
  plt.xlabel('epoch')
  plt.legend(), plt.grid()
  plt.savefig(fig_path, bbox_inches='tight')
  plt.show()


def plot_acc(history, output_dir):
  fig_path = f'../images/{output_dir}/accuracy.png'
  create_save_dir(fig_path)
  plt.plot(history['train_acc'], label='train_accuracy', marker='*')
  plt.plot(history['val_acc'], label='val_accuracy',  marker='*')
  plt.title('Accuracy per epoch'); plt.ylabel('Accuracy')
  plt.xlabel('epoch')
  plt.legend(), plt.grid()
  plt.savefig(fig_path, bbox_inches='tight')
  plt.show()