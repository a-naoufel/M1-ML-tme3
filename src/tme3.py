import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mltools import plot_data, plot_frontiere, make_grid, gen_arti


def perceptron_loss(w, x, y):

    x = np.asarray(x)
    y = np.asarray(y)

    margins = y * (x @ w)
    losses = np.maximum(0.0, -margins)
    return float(np.mean(losses))


def perceptron_grad(w, x, y):

    x = np.asarray(x)
    y = np.asarray(y)

    margins = y * (x @ w)
    viol = margins < 0
    if not np.any(viol):
        return np.zeros_like(w)
    grad = (-y[viol, None] * x[viol]).mean(axis=0)
    return grad


# class Lineaire(object):
#     def __init__(self,loss=perceptron_loss,loss_g=perceptron_grad,max_iter=100,eps=0.01):
#         self.max_iter, self.eps = max_iter,eps
#         self.w = None
#         self.loss,self.loss_g = loss,loss_g
        
#     def fit(self,datax,datay):
#         X = np.asarray(datax, dtype=float)
#         y = np.asarray(datay, dtype=float)

#         n, d = X.shape


#         rng = np.random.default_rng(None)
#         self.w = rng.normal(0.0, 0.01, size=d)

#         self.cost_history = []

#         for _ in range(self.max_iter):
#             # coût (moyen) sur le dataset
#             c = self.loss(self.w, X, y)
#             self.cost_history.append(c)

#             # gradient (moyen) sur le dataset
#             g = self.loss_g(self.w, X, y)

#             # mise à jour
#             self.w = self.w - self.eps * g

#         return self
    
#     def predict(self,datax):
#         X = np.asarray(datax)
#         scores = X @ self.w
#         yhat = np.where(scores >= 0, 1, -1)
#         return yhat

#     def score(self,datax,datay):
#         y = np.asarray(datay)
#         yhat = self.predict(datax)
#         return float(np.mean(yhat == y))

class Lineaire(object):
    def __init__(self, loss=perceptron_loss, loss_g=perceptron_grad, max_iter=100, eps=0.01):
        self.max_iter, self.eps = max_iter, eps
        self.w = None
        self.loss, self.loss_g = loss, loss_g
        self.cost_history = []

    def _encode_y(self, y):
        """Convertit des labels {a,b} vers {-1,+1}."""
        y = np.asarray(y)
        pos = y[0]
        return np.where(y == pos, 1.0, -1.0)

    def fit(self, datatrainx, datatrainy,datatestx=None, datatesty=None):
        X = np.asarray(datatrainx, dtype=float)
        y = self._encode_y(datatrainy)

        n, d = X.shape
        rng = np.random.default_rng(None)
        self.w = rng.normal(0.0, 0.01, size=d)

        self.cost_history = []
        self.train_err_history = []
        self.test_err_history = []

        has_test = (datatestx is not None) and (datatesty is not None)
        if has_test:
            Xte = np.asarray(datatestx, dtype=float)
            yte = np.asarray(datatesty, dtype=float)

        for _ in range(self.max_iter):
            # coût (moyen) sur train
            c = self.loss(self.w, X, y)
            self.cost_history.append(c)

            # erreur de classification
            train_err = 1.0 - self.score(X, y)
            self.train_err_history.append(train_err)

            if has_test:
                test_err = 1.0 - self.score(Xte, yte)
                self.test_err_history.append(test_err)

            # gradient + update
            g = self.loss_g(self.w, X, y)
            self.w = self.w - self.eps * g

        return self

    def predict(self, datax):
        X = np.asarray(datax, dtype=float)
        scores = X @ self.w
        yhat_pm = np.where(scores >= 0, 1, -1)
        return yhat_pm 

    def score(self, datax, datay):
        y = np.asarray(datay)
        y = self._encode_y(y)
        yhat = self.predict(datax)
        return float(np.mean(yhat == y))


def load_usps(fn):
    with open(fn,"r") as f:
        f.readline()
        data = [[float(x) for x in l.split()] for l in f if len(l.split())>2]
    tmp=np.array(data)
    return tmp[:,1:],tmp[:,0].astype(int)

def get_usps(l,datax,datay):
    if type(l)!=list:
        resx = datax[datay==l,:]
        resy = datay[datay==l]
        return resx,resy
    tmp =   list(zip(*[get_usps(i,datax,datay) for i in l]))
    tmpx,tmpy = np.vstack(tmp[0]),np.hstack(tmp[1])
    return tmpx,tmpy

def show_usps(data):
    plt.imshow(data.reshape((16,16)),interpolation="nearest",cmap="gray")



if __name__ =="__main__":
    uspsdatatrain = "../data/USPS_train.txt"
    uspsdatatest = "../data/USPS_test.txt"
    alltrainx,alltrainy = load_usps(uspsdatatrain)
    alltestx,alltesty = load_usps(uspsdatatest)
    neg = 5
    pos = 6
    datax,datay = get_usps([neg,pos],alltrainx,alltrainy)
    testx,testy = get_usps([neg,pos],alltestx,alltesty)
