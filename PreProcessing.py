## Pre-processing
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from skimage.filters.rank import entropy
from skimage.morphology import disk
from sklearn.model_selection import train_test_split
from skimage.filters import gaussian



# Nota: o gaussian tmb funciona para 3D
def gaussFiltering(nodule, sigma=1, output=None, mode='nearest', 
                   cval=0, multichannel=None, preserve_range=False, truncate=4.0):
    gaussImage=gaussian(nodule, sigma, output,mode,cval,multichannel,preserve_range,truncate)
    return gaussImage







