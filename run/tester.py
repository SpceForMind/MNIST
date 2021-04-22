import torch
from torch import nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms

from model.manager import MnistNNModelManager


class Tester:
    def __init__(self):
        self.__mnist_manager = MnistNNModelManager()

    def test_loop(self, batch_size, path_to_model):
        # Загружаем обученную модель
        self.__mnist_manager.load_model(path_to_model=path_to_model)

        # Загружаем датасеты для прогона тестов
        self.__load_test_datasets(batch_size=batch_size)

        # create a loss function
        criterion = nn.NLLLoss()

        # Среднее значение ошибки
        test_loss = 0

        # Количество правильных предиктов
        correct = 0

        for data, target in self.__test_loader:
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

            # Вызывает метод forward() в классе MnistNN.
            # После запуска строки переменная net_out будет иметь
            # логарифмический softmax-выход из нашей нейронной сети для заданной партии данны
            net_out = self.__mnist_manager.mnist_nn(data)

            # sum up batch loss
            test_loss += criterion(net_out, target).data

            # Предсказание нейронной сети - максимальная вероятность классификации выходного слова
            # После вызова метода forward -> Softmax class-probabilities
            pred = net_out.data.max(1)[1]  # get the index of the max log-probability

            # Суммируя выходы функции .eq(), получаем счетчик количества раз,
            # когда нейронная сеть выдает правильный ответ
            correct += pred.eq(target.data).sum()

        # Проходя по каждой партии входных данных,
        # выводим среднее значение функции потери и точность модели:
        test_loss /= len(self.__test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(self.__test_loader.dataset),
            100. * correct / len(self.__test_loader.dataset)))

    def __load_test_datasets(self, batch_size):
        self.__test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
            batch_size=batch_size, shuffle=True)


def main(args):
    batch_size = args.batch_size
    path_to_model = args.path_to_model

    trainer = Tester()
    trainer.test_loop(batch_size=batch_size, path_to_model=path_to_model)