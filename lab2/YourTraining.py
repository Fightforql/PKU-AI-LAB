import numpy as np
import random
import mnist
import pickle
from autograd.utils import PermIterator
from util import setseed
from autograd.BaseGraph import Graph
from autograd.BaseNode import *
import scipy.ndimage as ndimage

setseed(0)

_save_path_header = 'model/mymodel_'
lr = 2e-3
wd1 = 1e-4
wd2=0.02
epoch=30
batchsize=1024

def buildGraph(X,Y):
    """
    建图
    @param Y: n 样本的label
    @return: Graph类的实例, 建好的图
    """
    # TODO: YOUR CODE HERE
    std_X, mean_X = np.std(X, axis=0, keepdims=True)+1e-4, np.mean(X, axis=0, keepdims=True)
    nodes = [BatchNorm(784),
             Linear(784, 256),
             relu(),
             Dropout(0.2),
            

            Linear(256, 128),
            relu(),
            BatchNorm(128),
            Dropout(),
            Linear(128, 64),

            relu(),
            BatchNorm(64),
            Dropout(),
            
            Linear(64, mnist.num_class),
            Softmax(),
            CrossEntropyLoss(Y)]
    graph=Graph(nodes)
    return graph

X=mnist.val_X#(1000,784)
Y=mnist.val_Y #这是测试集1000
min_X=mnist.val_X#(69000, 784)
min_Y=mnist.val_Y#这是验证集的图片与数据(69000,)
#要进行数据的划分
#对图片进行 0 到 15 度的随机旋转
#每张图⽚是28*28的矩阵。在训练集和验证集中，图⽚已经展平成⼀个784维的向量。在测试集中没
#有展平。
#不旋转不平移的
s=(28,28)
X=min_X.reshape(-1,*s)
'''
for i in range(10000):
    print(1)
    photo=min_X[i+30000,:]
    photo=np.reshape(photo,(1,784))#改变形状
    X = np.concatenate((X,photo), axis=0)
    Y=np.concatenate((Y,Y[i]),axis=None)
'''
#只旋转不平移的
for i in range(10000):
    #print(2)
    angle=random.uniform(-25,25)
    #photo=min_X[i,:]#一张图片已经加载进来了,以数组的形式
        #对这张图片做数据增广，然后加入到训练集中去，要记得把y也加进去
    #photo=np.reshape(photo,(28,28))#改变形状
    photo=np.copy(X[i])
    photo=ndimage.rotate(photo,angle,reshape=False)
    X[i]=photo

#只平移不旋转的
for i in range(10000,20000):
    #print(3)
    yy1=random.uniform(-4,4)
    yy2=random.uniform(-4,4)
    photo=np.copy(X[i])
    ndimage.shift(photo,[yy1,yy2], output=photo, order=3, mode='nearest')#平移操作
    X[i]=photo

#又旋转又平移的
for i in range(20000,30000):
    #print(4)
    photo=np.copy(X[i])
    angle=random.uniform(-25,25)
    yy1=random.uniform(-4,4)
    yy2=random.uniform(-4,4)
    photo=ndimage.rotate(photo,angle,reshape=False)
    ndimage.shift(photo,[yy1,yy2], output=photo, order=3, mode='nearest')#平移操作
    X[i]=photo

X=X.reshape(X.shape[0],-1)
X=X[:40000]
'''
fig, ax = plt.subplots(
    nrows=3,
    ncols=4,
    sharex=True,
    sharey=True, )

ax = ax.flatten()

for i in range(12):
    # 只查看了前面12张图片
    img =X.data[i]
    ax[i].imshow(img, cmap='Greys', interpolation='nearest')

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()
'''

if __name__ == "__main__":
    graph = buildGraph(X,Y)
    # 训练
    for model in range(1):
        best_train_acc = 0
        dataloader = PermIterator(X.shape[0], batchsize)
        for i in range(1, epoch+1):
            hatys = []
            ys = []
            losss = []
            graph.train()
            for perm in dataloader:
                tX = X[perm]
                tY = Y[perm]
                graph[-1].y = tY
                graph.flush()
                pred, loss = graph.forward(tX)[-2:]
                hatys.append(np.argmax(pred, axis=1))
                ys.append(tY)
                graph.backward()
                graph.optimstep(lr, wd1, wd2)
                losss.append(loss)
            loss = np.average(losss)
            acc = np.average(np.concatenate(hatys)==np.concatenate(ys))
            print(f"epoch {i} loss {loss:.3e} acc {acc:.4f}")
            if acc > best_train_acc:
                best_train_acc = acc
                with open(_save_path_header + str(model) + '.npy', 'wb') as f:
                    pickle.dump(graph, f)

    # 测试
    for model in range(1):
        with open(_save_path_header + str(model) + '.npy', 'rb') as f:
            graph = pickle.load(f)
        graph.eval()
        graph.flush()
        pred = graph.forward(mnist.val_X, removelossnode=1)[-1]
        haty = np.argmax(pred, axis=1)
        print("valid acc", np.average(haty==mnist.val_Y))

