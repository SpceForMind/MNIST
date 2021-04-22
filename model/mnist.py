from torch import nn
import torch.nn.functional as F


class MnistNN(nn.Module):
    def __init__(self):
        '''
            Construct 4-layer NN

            : 1 layer get 28x28 img of numeral -> count of neurons is 28 * 28
            : 2 layer has 200 neurons and we see relation between 1 layer and 2 layer
                 nn.Linear(28 * 28, 200 - /this parameter define relation between #1 and #2 layers/)
            : 3 layer as well has 200 neurons and its output pass to OUT layer
            : 4 - OUT layer has 10 neurons
        '''
        super(MnistNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 10)

    def forward(self, x):
        '''

        :param x: input
        :return: Softmax act. function with output from #3 layer

        Note:
            We use RELU act. function for evaluation between #1 - #3 layers
            Call Softmax with output of #3 layer - its final stage of evaluations
        '''
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x)