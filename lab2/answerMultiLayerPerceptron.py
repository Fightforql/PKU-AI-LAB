import mnist
from copy import deepcopy
from typing import List
from autograd.BaseGraph import Graph
from autograd.utils import buildgraph
from autograd.BaseNode import *

# 超参数
# TODO: You can change the hyperparameters here
lr = 2e-3 # 学习率
wd1 = 3e-5  # L1正则化
wd2 = 1e-2  # L2正则化
batchsize = 128

def buildGraph(Y):
    """
    建图
    @param Y: n 样本的label
    @return: Graph类的实例, 建好的图
    """
    # TODO: YOUR CODE HERE
    nodes = [StdScaler(mnist.mean_X, mnist.std_X),
            Linear(784,196),
            relu(),
            BatchNorm(196),
            Dropout(0.3),

            Dropout(),
            BatchNorm(196),
            Linear(196, mnist.num_class),
            Softmax(),
            CrossEntropyLoss(Y)]
    graph=Graph(nodes)
    return graph