import torch
from torch import nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms

from model.manager import MnistNNModelManager

class Trainer:
    def __init__(self):
        self.__mnist_manager = MnistNNModelManager()

    def train_loop(self, batch_size, learning_rate, epochs, log_interval,
                   path_to_model=None, dir_to_save_model=None):
        # Загружаем обученную модель, если передан путь
        if path_to_model is not None:
            self.__mnist_manager.load_model(path_to_model=path_to_model)

        # Загружаем датасеты для обуения
        self.__load_mnist_datasets(batch_size=batch_size)

        # Создаем оптимайзер Стох. Градиентного Спуска
        optimizer = optim.SGD(self.__mnist_manager.mnist_nn.parameters(), lr=learning_rate, momentum=0.9)
        # Создаем loss-функцию
        criterion = nn.NLLLoss()

        # train loop
        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(self.__train_loader):
                data, target = Variable(data), Variable(target)

                # resize data from (batch_size, 1, 28, 28) - (dims of data from dataset)
                # to (batch_size, 28*28) - input for NN
                # Функция .view() работает с переменными PyTorch и преобразует их форму.
                # Если мы точно не знаем размерность данного измерения, можно использовать ‘-1’ нотацию
                # в определении размера. Поэтому при использование data.view(-1,28*28) можно сказать, что
                # второе измерение должно быть равно 28 x 28, а первое измерение должно быть вычислено из
                # размера переменной оригинальных данных. На практике это означает, что данные теперь будут
                # размера (batch_size, 784)
                data = data.view(-1, 28 * 28)

                # В следующей строке запускаем optimizer.zero_grad(), который обнуляет или перезапускает градиенты в
                # модели так, что они готовы для дальнейшего обратного распространения
                optimizer.zero_grad()

                # Вызывает метод forward() в классе MnistNN.
                # После запуска строки переменная net_out будет иметь
                # логарифмический softmax-выход из нашей нейронной сети для заданной партии данны
                net_out = self.__mnist_manager.mnist_nn(data)

                # Инициализация функции (COST) потери отрицательного логарифмического правдоподобия между выходом
                # нашей нейросети и истинными метками заданной партии данных.
                loss = criterion(net_out, target)

                # Запускает операцию обратного распространения ошибки из переменной потери
                # в обратном направлении через нейросеть
                loss.backward()

                # Выполнение градиентного спуска по шагам
                # на основе вычисленных во время операции .backward() градиентов.
                optimizer.step()

                # Вывод данных об обучении
                if batch_idx % log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(self.__train_loader.dataset),
                               100. * batch_idx / len(self.__train_loader), loss.data))

        # Сохраняем обученную модель, если передан путь
        if dir_to_save_model is not None:
            self.__mnist_manager.save_trained_model(dir=dir_to_save_model)

    def __load_mnist_datasets(self, batch_size):
        self.__train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=batch_size, shuffle=True)


def main(args):
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    epochs = args.epochs
    log_interval = args.log_interval
    path_to_model = args.path_to_model
    dir_to_save_model = args.dir_to_save_model

    trainer = Trainer()

    trainer.train_loop(batch_size=batch_size,
                       learning_rate=learning_rate,
                       epochs=epochs,
                       log_interval=log_interval,
                       path_to_model=path_to_model,
                       dir_to_save_model=dir_to_save_model)