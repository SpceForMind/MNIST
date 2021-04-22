import torch
from datetime import datetime
import os

from model.mnist import MnistNN


class MnistNNModelManager:
    def __init__(self):
        self.mnist_nn = MnistNN()

    def load_model(self, path_to_model):
        self.mnist_nn.load_state_dict(torch.load(path_to_model))

        # Нужно для замедления переобучения (Dropout)
        self.mnist_nn.eval()
        print('Model: %s loaded!' % path_to_model)

    def save_trained_model(self, dir):
        timestamp = datetime.now()
        model_name = 'train_%i_%i_%i_%i_%i_%i' % (
            timestamp.year,
            timestamp.month,
            timestamp.day,
            timestamp.hour,
            timestamp.minute,
            timestamp.second
        )
        print('Trained model saved as:', os.path.join(dir, model_name))
        torch.save(self.mnist_nn.state_dict(), os.path.join(dir, model_name))