import numpy as np

# 超参数
# TODO: You can change the hyperparameters here
lr = 0.9 # 学习率
wd = 1e-3 # l2正则化项系数


def predict(X, weight, bias):
    """
    使用输入的weight和bias，预测样本X是否为数字0。
    @param X: (n, d) 每行是一个输入样本。n: 样本数量, d: 样本的维度
    @param weight: (d,)
    @param bias: (1,)
    @return: (n,) 线性模型的输出，即wx+b
    """
    # TODO: YOUR CODE HERE
    return X@weight+bias
    raise NotImplementedError

def sigmoid(x):
    x_ravel = x.ravel()
    y = []
    for i in range(len(x_ravel)):
        if x_ravel[i] >= 0:
            y.append(1.0 / (1 + np.exp(-x_ravel[i])))
        else:
            y.append(np.exp(x_ravel[i]) / (np.exp(x_ravel[i]) + 1))
    return np.array(y).reshape(x.shape)

def step(X, weight, bias, Y):
    """
    单步训练, 进行一次forward、backward和参数更新
    @param X: (n, d) 每行是一个训练样本。 n: 样本数量， d: 样本的维度
    @param weight: (d,)
    @param bias: (1,)
    @param Y: (n,) 样本的label, 1表示为数字0, -1表示不为数字0
    @return:
        haty: (n,) 模型的输出, 为正表示数字为0, 为负表示数字不为0
        loss: (1,) 由交叉熵损失函数计算得到
        weight: (d,) 更新后的weight参数
        bias: (1,) 更新后的bias参数
    """
    # TODO: YOUR CODE HERE
    haty=predict(X,weight,bias)
    loss=-np.average(np.log(sigmoid(Y * haty) + 1e-6))
    s1=-np.average(((1.0 - sigmoid(Y * haty)) * Y).reshape(X.shape[0], -1) * X,axis=0) + 2 * wd * weight 
    s2= -np.average((1.0 - sigmoid(Y * haty)) * Y)
    weight=weight-lr*s1
    bias=bias-lr*s2
    return (haty,loss,weight,bias)
    raise NotImplementedError
