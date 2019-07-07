'''============================ myGaborFilter.py file ============================= %
Description                         : 
Input parameter                     :
Output parameter                    :
Subroutine  called                  : NA
Called by                           : NA
Reference                           :
Author of the code                  : Kundan Kumar
Date of creation                    : 2nd July 2019
------------------------------------------------------------------------------------------------------- %
Modified on                         :
Modification details                :
Modified By                         : Kundan Kumar
===================================================================== %
   Copy righted by ECE Department, ITER, SOA University India.
===================================================================== %'''

import numpy as np
from matplotlib import pyplot as plt
import cv2

class gaborFilter:
    def __init__(self, thetaInc=15, omegaInc = 0.4, K=np.pi,filterSize = (32,32),func = 'cos'):
        self.thetaInc = thetaInc
        self.omegaStart = 0.7
        self.omegaEnd = 1.5
        self.omegaInc = omegaInc
        self.K = K
        self.func= func
        self.nRows = filterSize[0]
        self.nCols = filterSize[1]

    def genGaborFilter(self,omega,theta):
        radius = (int(self.nRows/2.0), int(self.nCols/2.0))
        [x, y] = np.meshgrid(range(-radius[0], radius[0]+1), range(-radius[1], radius[1]+1))
        x1 = x * np.cos(theta) + y * np.sin(theta)
        y1 = -x * np.sin(theta) + y * np.cos(theta)
        gauss = omega**2 / (4*np.pi * self.K**2) * np.exp(- omega**2 / (8*self.K**2) * ( 4 * x1**2 + y1**2))
    #     myimshow(gauss)
        if self.func == 'sin':
            sinusoid = np.sin(omega * x1) * np.exp(self.K**2 / 2)
        else:
            sinusoid = np.cos(omega * x1) * np.exp(self.K**2 / 2)
        # sinusoid = func(omega * x1) * np.exp(K**2 / 2)
    #     myimshow(sinusoid)
        gabor = gauss * sinusoid
        return gabor

    def genFilterBanks(self,filterSize = (31,31)):
    # Here angle is in degree so convert to radian
        thetaInc = self.thetaInc*np.pi/180
        theta = np.arange(0, np.pi, thetaInc) # range of theta
        omega = np.arange(self.omegaStart, self.omegaEnd, self.omegaInc) # range of omega
        params = [(t,o) for o in omega for t in theta]
        gaborParams = []
        gaborParam = {'omega':omega, 'theta':theta}
        if self.func == 'sin':
            sinFilterBank = []
            for (theta, omega) in params:
                sinGabor = self.genGaborFilter(theta=theta,omega=omega)
                #sinGabor = myNormalize(sinGabor,0,1)
                sinFilterBank.append(sinGabor)
                gaborParams.append(gaborParam)
            return sinFilterBank
        else:
            cosFilterBank = []
            for (theta, omega) in params:
                cosGabor = self.genGaborFilter(theta=theta,omega=omega)
                #cosGabor = myNormalize(cosGabor,0,1)
                cosFilterBank.append(cosGabor)
                gaborParams.append(gaborParam)
            return cosFilterBank

    def plotImages(self,filterBanks,nRows,nCols):
        nFilters = len(filterBanks)
        plt.figure()
        for i in range(nFilters):
            plt.subplot(nRows,nCols,i+1)
            # title(r'$\theta$={theta:.2f}$\omega$={omega}'.format(**gaborParams[i]))
            plt.axis('off');
            plt.imshow(filterBanks[i],cmap='gray')

    def applyFilters(self,im, kernels):
        #Given a filter bank, apply them and record maximum response
        images = np.array([cv2.filter2D(im, cv2.CV_32F, k) for k in kernels])
        #return np.max(images, 0)
        return images

    def batchNormalization(self,imgs,minVal,maxVal):
        out = cv2.normalize(imgs, None, minVal, maxVal, cv2.NORM_MINMAX)
        return out

# A small example to use the filter
model = gaborFilter(K=2.2,thetaInc=45)
filterBanks = model.genFilterBanks((31,31))
model.plotImages(filterBanks,2,4)
img = cv2.imread('lena.bmp',0)
gaborResponse = model.applyFilters(img,filterBanks)
gaborResponse = model.batchNormalization(gaborResponse,0,255)
model.plotImages(gaborResponse,2,4)


