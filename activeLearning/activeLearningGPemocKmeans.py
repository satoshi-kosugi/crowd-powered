import math
import scipy
import numpy
import sklearn.kernel_ridge

import sys
import os
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)),os.pardir))
from config import activeLearning_config

import activeLearning.activeLearningGPprototypeKmeans

class Regressor(activeLearning.activeLearningGPprototypeKmeans.RegressorPrototype):

    def __init__(self, sigmaN = 0.01, gamma = None, kernel = 'rbf', norm = 1, verbose=True):

        activeLearning.activeLearningGPprototypeKmeans.RegressorPrototype.__init__(self, sigmaN=sigmaN, gamma=gamma, kernel=kernel, verbose=verbose)
        self.norm = activeLearning_config["norm"]

    def gaussianAbsoluteMoment(self, muTilde, predVar):

        f11 = scipy.special.hyp1f1(-0.5*self.norm, 0.5, -0.5*numpy.divide(muTilde**2,predVar))
        prefactors = ((2 * predVar**2)**(self.norm/2.0) * math.gamma((1 + self.norm)/2.0)) / numpy.sqrt(numpy.pi)

        return numpy.multiply(prefactors,f11)


    def calcEMOC(self, x, xcounts):

        emocScores = numpy.asmatrix(numpy.empty([x.shape[0],1], dtype=numpy.float))
        muTilde =numpy.asmatrix(numpy.zeros([x.shape[0],1], dtype=numpy.float))
        if self.X.shape[0] == 0:
            kAll = self.kernelFunc(x)
        else:
            kAll = self.kernelFunc(numpy.vstack([self.X, x]))
        k = kAll[0:self.X.shape[0],self.X.shape[0]:]
        selfKdiag = numpy.asmatrix(numpy.diag(kAll[self.X.shape[0]:,self.X.shape[0]:])).T

        sigmaF = self.calcSigmaF(x, k, selfKdiag)
        moments = numpy.asmatrix(self.gaussianAbsoluteMoment(numpy.asarray(muTilde), numpy.asarray(sigmaF)))

        term1 = 1.0 / (sigmaF + self.sigmaN)

        term2 = numpy.asmatrix(numpy.ones((self.X.shape[0] + 1,x.shape[0])), dtype=numpy.float)*(-1.0)
        term2[0:self.X.shape[0],:] = numpy.linalg.solve(self.K + numpy.identity(self.X.shape[0], dtype=numpy.float)*self.sigmaN, k)

        preCalcMult = numpy.dot(term2[:-1,:].T, kAll[0:self.X.shape[0],:])

        xCountsAll = numpy.vstack([self.Xcounts, xcounts]).reshape(1, -1)
        for idx in range(x.shape[0]):
            vAll = term1[idx,:]*(preCalcMult[idx,:] + numpy.dot(term2[-1,idx].T, kAll[self.X.shape[0] + idx,:]))
            vAll = numpy.multiply(vAll, xCountsAll)
            emocScores[idx,:] = numpy.mean(numpy.power(numpy.abs(vAll),self.norm))
        return numpy.multiply(emocScores,moments)


    def calcAlScores(self, x, xcounts):

        return self.calcEMOC(x, xcounts)
