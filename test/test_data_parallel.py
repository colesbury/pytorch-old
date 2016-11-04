import torch
import torch.nn as nn
from torch.autograd import Variable
import unittest
from common import TestCase

n_gpu = torch.cuda.device_count() < 2

class Net(nn.Container):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = nn.Linear(10, 20)
        self.l2 = nn.Linear(20, 30)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        return x


class TestDataParallel(TestCase):
    # @unittest.skipIf(n_gpu, "only one GPU detected")
    # def test_forward(self):
    #     x = Variable(torch.Tensor(16, 10).uniform_(), requires_grad=True).cuda()
    #     net1 = Net().cuda()
    #     output1 = net1(x)
    #
    #     dp = nn.DataParallel(Net().cuda(), range(n_gpu))
    #     output2 = dp(x)
    #     self.assertEqual(output1, output2)

    def test_backward(self):
        dp = nn.DataParallel(Net().cuda(), [0, 1, 2, 3])
        input = Variable(torch.Tensor(16, 10).uniform_(), requires_grad=True)
        output = dp(input)
        output.sum().backward()
        print('DONE test_backward')


if __name__ == '__main__':
    # TestDataParallel().test_backward()
    unittest.main()
