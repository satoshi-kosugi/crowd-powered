import math
import scipy
import numpy
import sklearn.kernel_ridge

import sys
import os
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)),os.pardir))
from config import activeLearning_config

class RegressorPrototype:

    def __init__(self, sigmaN=0.01, gamma=None, kernel='rbf', verbose=True):

        self.sigmaN = activeLearning_config["sigmaN"]
        self.kernel = activeLearning_config["kernel"]
        self.gamma = activeLearning_config["gamma"]
        self.numKernelCores = activeLearning_config["numKernelCores"]
        self.verbose = activeLearning_config["verbose"]

        self.K = []
        self.alpha = []
        self.X = numpy.asmatrix(numpy.empty([0,1], dtype=numpy.float))
        self.Xcounts = numpy.asmatrix(numpy.empty([0,1], dtype=numpy.float))
        self.y = []

        self.allowedKernels = ['exponential', 'rbf', 'sigmoid', 'polynomial', 'poly', 'linear', 'cosine']

        if self.kernel not in self.allowedKernels:
            raise Exception('Unknown kernel %s!'%self.kernel)


    def kernelFunc(self, x1, x2=None, gamma=None):

        if gamma is not None:
            self.gamma = gamma

        if self.kernel == "exponential":
            if x2 is None:
                x1_ = numpy.asarray(x1)
                x1_ = numpy.repeat(x1_[:,None], x1_.shape[0], axis=1)
                x2_ = x1_.transpose((1,0,2))
                kernels = numpy.exp(-((x1_ - x2_) ** 2).sum(axis=2) ** 0.5 * 10)
                return numpy.asmatrix(kernels)
            else:
                x1_ = numpy.asarray(x1)
                x2_ = numpy.asarray(x2)
                x1_ = numpy.repeat(x1_[:,None], x2_.shape[0], axis=1)
                x2_ = numpy.repeat(x2_[None], x1_.shape[0], axis=0)
                kernels = numpy.exp(-((x1_ - x2_) ** 2).sum(axis=2) ** 0.5 * 10)
                return numpy.asmatrix(kernels)
        elif self.kernel in ['rbf', "sigmoid"]:
            return numpy.asmatrix(sklearn.kernel_ridge.pairwise_kernels(x1, x2, metric=self.kernel, gamma=self.gamma, n_jobs=self.numKernelCores), dtype=numpy.float)
        else:
            return numpy.asmatrix(sklearn.kernel_ridge.pairwise_kernels(x1, x2, metric=self.kernel, n_jobs=self.numKernelCores), dtype=numpy.float)


    def train(self, X, Xcounts, y, sigmaN=None, gamma=None, kernel=None, numKernelCores=None):
        if gamma is None:
            gamma = 1000
        if sigmaN is not None:
            if self.verbose:
                print('regressor - switched sigmaN from {} to {}'.format(self.sigmaN, sigmaN))
            self.sigmaN = sigmaN

        if gamma is not None:
            if self.verbose:
                print('regressor - switched gamma from {} to {}'.format(self.gamma, gamma))
            self.gamma = gamma

        if kernel is not None and kernel in self.allowedKernels:
            if self.verbose:
                print('regressor - switched kernel from {} to {}'.format(self.kernel, kernel))
            self.kernel = kernel

        if numKernelCores is not None:
            if self.verbose:
                print('regressor - switched numKernelCores from {} to {}'.format(self.numKernelCores, numKernelCores))
            self.numKernelCores = numKernelCores

        if X.shape[0] != 0:
            self.X = X
            self.Xcounts = Xcounts
            self.y = y
            self.K = self.kernelFunc(X, gamma=self.gamma)
            self.alpha = numpy.linalg.solve(self.K + numpy.identity(self.X.shape[0], dtype=numpy.float)*self.sigmaN, self.y)

            self.checkVars()


    def update(self, x, xcounts, y):
        if self.X.shape[0] == 0:
            self.train(x, xcounts, y, sigmaN=self.sigmaN, gamma=self.gamma, kernel=self.kernel, numKernelCores=self.numKernelCores)
        else:
            k = self.kernelFunc(self.X, x)
            selfKdiag = self.getSelfKdiag(x)

            term1 = 1.0 / (self.calcSigmaF(x, k, selfKdiag) + self.sigmaN)

            term2 = numpy.asmatrix(numpy.ones((self.X.shape[0] + 1,x.shape[0])), dtype=numpy.float) * -1.0
            term2[0:self.X.shape[0],:] = numpy.linalg.solve(self.K + numpy.identity(self.X.shape[0], dtype=numpy.float)*self.sigmaN, k)

            term3 = self.predict(x) - y

            self.alpha = numpy.append(self.alpha, numpy.zeros((1,self.alpha.shape[1])), axis=0) + numpy.dot(numpy.dot(term1,term2.T).T,term3)
            self.K = numpy.append(numpy.append(self.K, k, axis=1), numpy.append(k.T, selfKdiag, axis=1), axis=0)
            self.X = numpy.append(self.X, x, axis=0)
            self.Xcounts = numpy.append(self.Xcounts, xcounts, axis=0)
            self.y = numpy.append(self.y, y, axis=0)

            self.checkVars()


    def checkVars(self):

        if not numpy.all(numpy.isfinite(self.X)):
            raise Exception('not numpy.all(numpy.isfinite(self.X))')

        if not numpy.all(numpy.isfinite(self.y)):
            raise Exception('not numpy.all(numpy.isfinite(self.y))')

        if not numpy.all(numpy.isfinite(self.K)):
            raise Exception('not numpy.all(numpy.isfinite(self.K))')

        if not numpy.all(numpy.isfinite(self.alpha)):
            raise Exception('not numpy.all(numpy.isfinite(self.alpha))')


    def predict(self, x):
        if self.X.shape[0] == 0:
            return x[:,0] * 0
        k = self.kernelFunc(self.X, x)
        return numpy.dot(k.T, self.alpha)


    def calcSigmaF(self, x, k=None, selfKdiag=None):
        if self.X.shape[0] == 0:
            return x[:,0] * 0 + 1

        if k is None:
            k = self.kernelFunc(self.X, x)

        if selfKdiag is None:
            selfKdiag = self.getSelfKdiag(x)

        return selfKdiag - numpy.sum(numpy.multiply(k, numpy.linalg.solve(self.K + numpy.identity(self.K.shape[0], dtype=numpy.float)*self.sigmaN, k)), axis=0).T


    def getSelfKdiag(self, x):

        selfKdiag = numpy.asmatrix(numpy.empty([x.shape[0],1], dtype=numpy.float))
        for idx in range(x.shape[0]):
            selfKdiag[idx,:] = self.kernelFunc(x[idx,:])

        return selfKdiag


    def calcAlScores(self, x, xcounts):

        return None


    def chooseSample(self, x):

        alScores = self.calcAlScores(x)

        if alScores.shape[0] != x.shape[0] or alScores.shape[1] != 1:
            raise Exception('alScores.shape[0] != x.shape[0] or alScores.shape[1] != 1')

        if not numpy.all(numpy.isfinite(alScores)):
            raise Exception('not numpy.all(numpy.isfinite(alScores))')

        return numpy.argmax(alScores, axis=0).item(0)
