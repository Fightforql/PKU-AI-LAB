'''
import numpy as np
from numpy.random import randn
import mnist
import pickle
from util import setseed
from answerLogisticRegression import step,predict,sigmoid,lr,wd
from scipy import ndimage

setseed(0)

save_path= 'model/mymodel_'
lr = 2e-3
wd = 5e-4
epoch=50


X_val=mnist.val_X
X_trn=mnist.trn_X
Y_trn=mnist.trn_Y
Y_val=mnist.val_Y

X=np.concatenate((X_trn,X_val),axis=0)
Y=np.concatenate((Y_trn,Y_val),axis=0)



train = 0.8  
valid = 0.2

trainnum = int(len(X) * train)
validnum = len(X) - trainnum

X_trn = X[:trainnum]
Y_trn = Y[:trainnum]
X_val = X[trainnum:]
Y_val = Y[trainnum:]


X=X_trn
Y=Y_trn
X=X[:10000]
Y=Y[:10000]
image_shape =(28, 28)
X_images = X.reshape(-1, *image_shape)
X1=np.zeros_like(X_images)


shift_range = 4
shift_x = np.random.randint(-shift_range, shift_range + 1)
shift_y = np.random.randint(-shift_range, shift_range + 1)
rotation_angle = np.random.randint(-30, -20)
for i in range(len(X_images)):
    image = np.copy(X_images[i]) 
    rotated_image =ndimage.rotate(image, angle=rotation_angle, reshape=False)
    shifted_image = np.roll(rotated_image, 4, axis=0)
    shifted_image = np.roll(shifted_image, 4, axis=1) 
    X1[i] = shifted_image

X1=X1.reshape(len(X1), -1)

X=np.concatenate((X,X1),axis=0)
Y=np.concatenate((Y,Y),axis=0)
if __name__ == '__main__':
    avg_acc = 0.0
    for model in range(10):
        best_train_acc = 0
        tmpY = 2 * (Y== model) - 1
        weight = randn(mnist.num_feat)
        bias = 0
        for i in range(1, epoch + 1):
            haty, loss, weight, bias = step( X, weight, bias, tmpY)
            acc = np.average(haty * tmpY>0)
            print(f"epoch {i} loss {loss:.3e} acc {acc:.4f}")
            if acc > best_train_acc:
                best_train_acc = acc
                with open(save_path + str(model) + '.npy', 'wb') as f:
                    pickle.dump((weight, bias), f)

        with open(save_path + str(model) + '.npy', 'rb') as f:
            weight, bias = pickle.load(f)
        haty = predict(X_val, weight, bias)
        haty = (haty > 0)
        y = (Y_val == model)    
        print(f'confusion matrix: TP {np.sum((haty > 0) * (y > 0))} FN {np.sum((y > 0) * (haty <= 0))} FP {np.sum((y <= 0) * (haty > 0))} TN {np.sum((y <= 0) * (haty <= 0))}')
        valid_acc = np.average(haty == y)
        avg_acc += valid_acc
        print(f'valid acc {valid_acc:.4f}')

class MyModel:
    def __init__(self) -> None:
        self.weight = []
        self.bias = []
        for model in range(10):
            with open(My.save_path + str(model) + '.npy', 'rb') as f:
                _weight, _bias = pickle.load(f)
                self.weight.append(_weight)
                self.bias.append(_bias)

    def __call__(self, figure):
        preds =[]
        for modelid in range(10):
            pred = figure @ self.weight[modelid] + self.bias[modelid]
            preds.append(pred)
        return np.argmax(preds)


modeldict = {
    "Null": NullModel,
    "LR": LRModel,
    "Tree": TreeModel,
    "Forest": ForestModel,
    "SR": SRModel,
    "MLP": MLPModel,
    "Your": MyModel
}
'''
def printhaha():
    print('ha')