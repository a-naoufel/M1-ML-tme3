import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mltools import *


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

def hinge_loss(w, x, y, alpha=1.0, lam=0.0):
    """
    x: (d,) ou (n,d)
    y: scalaire ou (n,)
    Retourne la moyenne sur n si batch.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.ndim == 1:
        margin = y * np.dot(x, w)
        return max(0.0, alpha - margin) + lam * np.dot(w, w)

    margins = y * (x @ w)
    losses = np.maximum(0.0, alpha - margins)
    return float(np.mean(losses) + lam * np.dot(w, w))


def hinge_loss_grad(w, x, y, alpha=1.0, lam=0.0):
    """
    Gradient du hinge + L2.
    Pour un point:
      si alpha - y<w,x> <= 0 -> grad = 2 lam w
      sinon grad = -y x + 2 lam w
    Batch: moyenne des contributions hinge + 2 lam w
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    reg = 2.0 * lam * w

    if x.ndim == 1:
        margin = y * np.dot(x, w)
        if alpha - margin <= 0:
            return reg
        return (-y * x) + reg
    
    margins = y * (x @ w)
    viol = (alpha - margins) > 0
    if np.any(viol):
        grad_hinge = (-y[viol, None] * x[viol]).mean(axis=0)
    else:
        grad_hinge = np.zeros_like(w)

    return grad_hinge + reg


def proj_biais(datax):

    X = np.asarray(datax, dtype=float)
    return np.c_[np.ones(X.shape[0]), X]


def proj_poly(datax):

    X = np.asarray(datax, dtype=float)
    n, d = X.shape

    # 1 et termes linéaires
    feats = [np.ones((n, 1)), X]

    # termes quadratiques et croisés (i <= j) : x_i * x_j
    quad = []
    for i in range(d):
        for j in range(i, d):
            quad.append((X[:, i] * X[:, j]).reshape(n, 1))

    feats.append(np.hstack(quad) if quad else np.empty((n, 0)))
    return np.hstack(feats)

def proj_gauss(datax, base, sigma):

    X = np.asarray(datax, dtype=float)
    B = np.asarray(base, dtype=float)
    sigma = float(sigma)
    if sigma <= 0:
        raise ValueError("sigma doit être > 0")

    # distances au carré: (n,b)
    # (x - b)^2 = x^2 + b^2 - 2 x.b
    X2 = np.sum(X**2, axis=1, keepdims=True)      # (n,1)
    B2 = np.sum(B**2, axis=1, keepdims=True).T    # (1,b)
    D2 = X2 + B2 - 2 * (X @ B.T)                  # (n,b)

    Phi = np.exp(-D2 / (2.0 * sigma**2))
    return Phi

class Lineaire:
    def __init__(self, loss, loss_g, max_iter=50, eps=1e-2, w=None, projection=None):
        """
        projection: None ou fonction (X)->Xproj
        """
        self.loss = loss
        self.loss_g = loss_g
        self.max_iter = int(max_iter)   # nb d'époques
        self.eps = float(eps)
        self.random_state = 0

        self.projection = projection

        self.w = None if w is None else np.asarray(w, dtype=float)

        self.cost_history = []
        self.train_err_history = []
        self.test_err_history = []

    def _proj(self, X):
        X = np.asarray(X, dtype=float)
        if self.projection is None:
            return X
        return self.projection(X)

    def predict(self, datax):
        X = self._proj(datax)
        scores = X @ self.w
        return np.where(scores >= 0, 1, -1)

    def score(self, datax, datay):
        y = np.asarray(datay)
        yhat = self.predict(datax)
        return float(np.mean(yhat == y))

    def fit(self, datax, datay, testx=None, testy=None, batch_size=None, shuffle=True):
        # Projeter avant apprentissage
        X = self._proj(datax)
        y = np.asarray(datay, dtype=float)
        n, d = X.shape

        has_test = (testx is not None) and (testy is not None)
        if has_test:
            Xte = self._proj(testx)
            yte = np.asarray(testy, dtype=float)

        if self.w is None:
            rng = np.random.default_rng(self.random_state)
            self.w = rng.normal(0.0, 0.01, size=d)

        if batch_size is None:
            batch_size = n
        batch_size = int(batch_size)
        batch_size = max(1, min(batch_size, n))

        rng = np.random.default_rng(self.random_state)

        self.cost_history = []
        self.train_err_history = []
        self.test_err_history = []

        for _ in range(self.max_iter):
            # métriques par époque
            self.cost_history.append(self.loss(self.w, X, y))
            self.train_err_history.append(1.0 - self.score(datax, y))  # score repasse par _proj
            if has_test:
                self.test_err_history.append(1.0 - self.score(testx, yte))

            # shuffle
            if shuffle:
                perm = rng.permutation(n)
                Xs, ys = X[perm], y[perm]
            else:
                Xs, ys = X, y

            # mini-batches
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                Xb = Xs[start:end]
                yb = ys[start:end]
                g = self.loss_g(self.w, Xb, yb)
                self.w = self.w - self.eps * g

        return self

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

