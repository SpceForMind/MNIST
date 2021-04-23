import cv2
from torchvision import transforms
from torch.autograd import Variable

from model.manager import MnistNNModelManager


class NumByPhotoRecognizer:
    def __init__(self):
        pass

    def recognize(self, nn_model_path, img_path):
        self.__load_nn_model(nn_model_path=nn_model_path)
        self.load_and_thumbnail_image(img_path=img_path)
        out = self.__mnist_manager.mnist_nn(self.__tensor_img)
        pred = out.data.max(1)[1] # Берем максимальную вероятность SoftMax
        self.recognized_num = pred.numpy()[0] # Преобразуем Tensor (tensor(7))  -> numpy массив и берем индекс

    def load_and_thumbnail_image(self, img_path):
        self.__img = cv2.imread(img_path)

        # Переводим в оттенки серого
        self.__img = cv2.cvtColor(self.__img, cv2.COLOR_BGR2GRAY)

        # Сжимаем до 28х28
        self.__img = cv2.resize(self.__img, (28, 28))

        # Инвертируем цвета (т.к. НС училась на белых цифрах на черном фоне)
        self.__img = cv2.bitwise_not(self.__img)
        cv2.imwrite(img_path, self.__img)

        # Нормализация
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.__tensor_img = normalize(self.__img) # unsqueeze to add artificial first dimension
        self.__tensor_img = Variable(self.__tensor_img)
        self.__tensor_img = self.__tensor_img.view(-1, 28 * 28)

    def __load_nn_model(self, nn_model_path):
        self.__mnist_manager = MnistNNModelManager()
        self.__mnist_manager.load_model(path_to_model=nn_model_path)


def main(args):
    recognizer = NumByPhotoRecognizer()
    recognizer.recognize(nn_model_path=args.nn_model_path, img_path=args.img_path)
    print('Img path:', args.img_path)
    print('NN-predict:', recognizer.recognized_num)