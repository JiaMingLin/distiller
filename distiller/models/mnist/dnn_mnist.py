#
# Copyright (c) 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""An implementation of a trivial MNIST model.
 
The original network definition is sourced here: https://github.com/pytorch/examples/blob/master/mnist/main.py
"""

import torch.nn as nn
import torch.nn.functional as F


__all__ = ['dnn_mnist']


class DNN_MNIST(nn.Module):
    def __init__(self):
        super().__init__()
        # self.conv1 = nn.Conv2d(1, 20, 5, 1)
        # self.relu1 = nn.ReLU(inplace=False)
        # self.pool1 = nn.MaxPool2d(2, 2)
        # self.conv2 = nn.Conv2d(20, 50, 5, 1)
        # self.relu2 = nn.ReLU(inplace=False)
        # self.pool2 = nn.MaxPool2d(2, 2)
        # self.fc1 = nn.Linear(4*4*50, 500)
        # self.relu3 = nn.ReLU(inplace=False)
        # self.fc2 = nn.Linear(500, 10)
        self.dense11 = nn.Linear(784, 512)
        self.relu1 = nn.ReLU(inplace=False)
        self.dense12 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU(inplace=False)
        self.dense13 = nn.Linear(256, 128)
        self.relu3 = nn.ReLU(inplace=False)
        self.dense14 = nn.Linear(128, 10)
        self.output = nn.LogSoftmax()


        
    def forward(self, x):
        # x = self.pool1(self.relu1(self.conv1(x)))
        # x = self.pool2(self.relu2(self.conv2(x)))
        # x = x.view(x.size(0), -1)
        # x = self.relu3(self.fc1(x))
        # x = self.fc2(x)
        x = x.view(x.size(0), -1)
        x = self.relu1(self.dense11(x))
        x = self.relu2(self.dense12(x))
        x = self.relu3(self.dense13(x))
        x = self.output(self.dense14(x))

        return x


def dnn_mnist():
    model = DNN_MNIST()
    return model