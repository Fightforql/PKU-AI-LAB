from answerMultiLayerPerceptron import buildGraph, lr, wd1, wd2, batchsize
import mnist
import numpy as np
import pickle
from autograd.utils import PermIterator
from util import setseed
from numpy.random import randn
from scipy import ndimage
from ha import printhaha

setseed(0) # 固定随机数种子以提高可复现性
lr=2e-3
wd2=1e-4
batchsize=128
save_path = "model/mymodel_"

# 从验证集和训练集中随机抽取指定数量的样本
X_val=mnist.val_X
X_trn=mnist.trn_X
Y_trn=mnist.trn_Y
Y_val=mnist.val_Y

X=np.concatenate((X_trn,X_val),axis=0)
Y=np.concatenate((Y_trn,Y_val),axis=0)
# 设置训练集和验证集的比例
train_ratio = 0.8  # 训练集比例
validation_ratio = 0.2  # 验证集比例

num_train_samples = int(len(X) * train_ratio)
num_validation_samples = len(X) - num_train_samples

X_trn = X[:num_train_samples]
Y_trn = Y[:num_train_samples]
X_val = X[num_train_samples:]
Y_val = Y[num_train_samples:]
image_shape =(28, 28)
X_images = X_trn.reshape(-1, *image_shape)

shift_range = 4
shift_x = np.random.randint(-shift_range, shift_range + 1)
shift_y = np.random.randint(-shift_range, shift_range + 1)
rotation_range = 25  # 旋转角度范围为 [-10, 10] 度
rotation_angle = np.random.randint(-rotation_range, rotation_range + 1)
for i in range(len(X_images)):
    image = np.copy(X_images[i]) 
    #rotated_image =ndimage.rotate(image.astype(int), angle=rotation_angle, reshape=False)
    shifted_image = np.roll(image, shift_x, axis=0)
    shifted_image = np.roll(shifted_image, shift_y, axis=1) 
    X_images[i] = shifted_image
X_trn= X_images.reshape(len(X_images), -1)
if __name__ == "__main__":
    #graph = buildGraph(Y)
    # 训练
    avg_acc=0
    #dataloader = PermIterator(X.shape[0], batchsize)
    for model in range(1):
        '''
        X=np.concatenate((X_trn,X_val),axis=0)
        Y=np.concatenate((Y_trn,Y_val),axis=0)
        val_indices = np.random.choice(len(X), size=num_samples, replace=False)
        X_val=X[val_indices]
        Y_val=Y[val_indices]
        X=X_val
        Y=Y_val
        '''
        '''
        val_indices = np.random.choice(len(X_val), size=num_samples, replace=False)
        X_val_sampled = X_val[val_indices]
        Y_val_sampled = Y_val[val_indices]

        X= np.concatenate((X_val_sampled, X_trn), axis=0)
        Y= np.concatenate((Y_val_sampled, Y_trn), axis=0)

        '''
        '''
        image_shape =(28, 28)
        X_images = X.reshape(-1, *image_shape)
        shift_range = 4
        shift_x = np.random.randint(-shift_range, shift_range + 1)
        shift_y = np.random.randint(-shift_range, shift_range + 1)
        rotation_range = 25# 旋转角度范围为 [-15, 15] 度
        rotation_angle = np.random.randint(-rotation_range, rotation_range + 1)
        for i in range(len(X_images)):
            image = np.copy(X_images[i]) 
            shifted_image = np.roll(image, shift_x, axis=0)
            shifted_image = np.roll(shifted_image, shift_y, axis=1) 
            rotated_image = rotate(shifted_image, angle=rotation_angle, reshape=False) 
            X_images[i] = shifted_image
        X= X_images.reshape(len(X_images), -1)
        '''
        graph = buildGraph(Y_trn)
        dataloader = PermIterator(X_trn.shape[0], batchsize)
        best_train_acc = 0
        for i in range(1, 20+1):
            hatys = []
            ys = []
            losss = []
            graph.train()
            for perm in dataloader:
                tX = X_trn[perm]
                tY = Y_trn[perm]
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
                with open(save_path+str(model)+'.npy', "wb") as f:
                    pickle.dump(graph, f)
#测试
        with open(save_path + str(model) + '.npy', 'rb') as f:
            graph = pickle.load(f)
        graph.eval()
        graph.flush()
        pred = graph.forward(X_val, removelossnode=1)[-1]
        haty = np.argmax(pred, axis=1)  
        valid_acc = np.average(haty ==Y_val)
        avg_acc += valid_acc
        print(f'valid acc {valid_acc:.4f}')
    print(f'avg acc {avg_acc / 10:.4f}')



class MyModel:
    def __init__(self) -> None:
        self.models = []
        for model_id in range(3):
            with open(My.save_path + str(model_id) + '.npy', 'rb') as f:
                graph = pickle.load(f)
            graph.eval()
            self.models.append(graph)

    def __call__(self, figure):
        predictions = []
        for model in self.models:
            model.flush()
            pred = model.forward(figure, removelossnode=True)[-1]
            predictions.append(pred)
        avg_pred = np.mean(predictions, axis=0)
        return np.argmax(avg_pred, axis=-1)

modeldict = {
    "Null": NullModel,
    "LR": LRModel,
    "Tree": TreeModel,
    "Forest": ForestModel,
    "SR": SRModel,
    "MLP": MLPModel,
    "Your": MyModel
}
