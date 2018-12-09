import numpy as np
import numpy.matlib as ml
#import cv2
from matplotlib import pyplot as plt

def boxfilter(imSrc,r):
    [hei,wid]=imSrc.shape
    imDst=np.zeros(imSrc.shape)
    
    # Cumulative sum over Y axis
    imCum=np.cumsum(imSrc,0)
    
    # Difference over Y axis
    imDst[range(0, r),:]=imCum[(range(r, 2*r)),:]
    imDst[range(r,hei-r-1),...]=imCum[(range(2*r,hei-1)),...]-imCum[(range(0,hei-2*r-1)),...]
    imDst[(range(hei-r-1,hei-1)),...]=ml.repmat(imCum[hei-1,...], r,1)-imCum[(range(hei-2*r-2,hei-r-2)),...]
    
    # Cumulative sum over X axis
    imCum=np.cumsum(imDst,1)
    
    # Difference over X axis
    imDst[:,range(0, r)]=imCum[:,(range(r, 2*r))]
    imDst[...,range(r,wid-r-1)]=imCum[:,(range(2*r,wid-1))]-imCum[...,(range(0,wid-2*r-1))]
    imDst[...,(range(wid-r-1,wid-1))]=ml.repmat(imCum[...,wid-1], r, 1).T-imCum[...,(range(wid-2*r-1,wid-r-1))]
    
    return imDst
    
def guidedFilt(I,p,r,eps):
    [hei,wid]=I.shape
    sizing=np.ones((hei,wid))
    N=boxfilter(sizing,r) 
    mean_I=boxfilter(I,r)/N
    mean_p=boxfilter(p,r)/N
    mean_Ip=boxfilter(I*p,r)/N
    cov_Ip=mean_Ip-mean_I*mean_p
    
    mean_II=boxfilter(I*I,r)/N
    var_I=mean_II-mean_I*mean_I
    
    a=cov_Ip/(var_I+eps)
    b=mean_p-a*mean_I
    
    mean_a=boxfilter(a,r)/N
    mean_b=boxfilter(b,r)/N
    
    q=mean_a*I+mean_b
    return q
    
