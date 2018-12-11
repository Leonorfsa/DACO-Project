import numpy as np
import numpy.matlib as ml

def boxfilter(imSrc,r):
    
#   Box filtering using cumulative sum (based on the work by: 
#   He Kaiming, Jian Sun, and Xiaoou Tang. "Guided image filtering.")
#
#   - Definition imDst(x, y)=sum(sum(imSrc(x-r:x+r,y-r:y+r)));
#   - Running time independent of r; 
#   - Equivalent to the function colfilt() on Matlab but much faster.
    
    [hei,wid]=imSrc.shape
    imDst=np.zeros(imSrc.shape)

    # Cumulative sum over Y axis
    imCum=np.cumsum(imSrc,0)
    
    # Difference over Y axis
    imDst[0:r+1,...]=imCum[r: 2*r+1,...]
    imDst[r+1:hei-r,...]=imCum[2*r+1:hei,...]-imCum[1:hei-2*r,...]
    imDst[hei-r:hei,...]=imCum[hei-1,...]-imCum[hei-2*r:hei-r,...]
    
    # Cumulative sum over X axis
    imCum=np.cumsum(imDst,1)
    
    # Difference over X axis
    imDst[...,0:r+1]=imCum[...,r, 2*r+1]
    imDst[...,r+1,wid-r]=imCum[:,2*r+1,wid]-imCum[...,1,wid-2*r]
    imDst[...,wid-r,wid]=imCum[...,wid-1]-imCum[...,wid-2*r,wid-r]
    
    return imDst
    
def guidedFilt(I,p,r,eps):
    [hei,wid]=I.shape
    sizing=np.ones((hei,wid))
    N=boxfilter(sizing,r) 
    mean_I=boxfilter(I,r)/N
    mean_p=boxfilter(p,r)/N
    mean_Ip=boxfilter(I*p,r)/N
    cov_Ip=mean_Ip-mean_I*mean_p # this is the covariance of (I, p) in each local patch.
    
    mean_II=boxfilter(I*I,r)/N
    var_I=mean_II-mean_I*mean_I
    
    a=cov_Ip/(var_I+eps)
    b=mean_p-a*mean_I
    
    mean_a=boxfilter(a,r)/N
    mean_b=boxfilter(b,r)/N
    
    q=mean_a*I+mean_b
    return q
    
