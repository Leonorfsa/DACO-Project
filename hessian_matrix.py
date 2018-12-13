#================================================
#            HESSIAN MATRIX FORMULAS
#================================================
# Functions here:
#   - hessian_matrix_eigvals: used on eigenValues (remove if skimage.feature.hessian_matrix_eigvals works)
#   - _image_orthogonal_matrix22_eigvals: used on hessian_matrix_eigvals (same as previous)
#   - _hessian_matrix_image: (same as previous)
#
#   - eigenValuesShapeIndexCurveness: returns hessian matrix eigenValues, shape index and curvedness

#%% IMPORTS

from skimage.feature import hessian_matrix#, hessian_matrix_eigvals
import numpy as np
from itertools import combinations_with_replacement
import math
from warnings import warn

#%% FUNCTIONS

def hessian_matrix_eigvals(H_elems, Hxy=None, Hyy=None, Hxx=None):
    """Compute Eigenvalues of Hessian matrix.
    Parameters
    ----------
    H_elems : list of ndarray
        The upper-diagonal elements of the Hessian matrix, as returned
        by `hessian_matrix`.
    Hxy : ndarray, deprecated
        Element of the Hessian matrix for each pixel in the input image.
    Hyy : ndarray, deprecated
        Element of the Hessian matrix for each pixel in the input image.
    Hxx : ndarray, deprecated
        Element of the Hessian matrix for each pixel in the input image.
    Returns
    -------
    eigs : ndarray
        The eigenvalues of the Hessian matrix, in decreasing order. The
        eigenvalues are the leading dimension. That is, ``eigs[i, j, k]``
        contains the ith-largest eigenvalue at position (j, k).
    Examples
    --------
    >>> from skimage.feature import hessian_matrix, hessian_matrix_eigvals
    >>> square = np.zeros((5, 5))
    >>> square[2, 2] = 4
    >>> H_elems = hessian_matrix(square, sigma=0.1, order='rc')
    >>> hessian_matrix_eigvals(H_elems)[0]
    array([[ 0.,  0.,  2.,  0.,  0.],
           [ 0.,  1.,  0.,  1.,  0.],
           [ 2.,  0., -2.,  0.,  2.],
           [ 0.,  1.,  0.,  1.,  0.],
           [ 0.,  0.,  2.,  0.,  0.]])
    """
    if Hxy is not None:
        if Hxx is None:
            Hxx = H_elems
        H_elems = [Hxx, Hxy, Hyy]
        warn('The API of `hessian_matrix_eigvals` has changed. Use a list of '
             'elements instead of separate arguments. The old version of the '
             'API will be removed in version 0.16.')
    if len(H_elems) == 3:  # Use fast Cython code for 2D
        eigvals = np.array(_image_orthogonal_matrix22_eigvals(*H_elems))
    else:
        matrices = _hessian_matrix_image(H_elems)
        # eigvalsh returns eigenvalues in increasing order. We want decreasing
        eigvals = np.linalg.eigvalsh(matrices)[..., ::-1]
        leading_axes = tuple(range(eigvals.ndim - 1))
        eigvals = np.transpose(eigvals, (eigvals.ndim - 1,) + leading_axes)
    return eigvals

# Used on hessian_matrix_eigvals
def _image_orthogonal_matrix22_eigvals(M00, M01, M11):
    l1 = (M00 + M11) / 2 + np.sqrt(4 * M01 ** 2 + (M00 - M11) ** 2) / 2
    l2 = (M00 + M11) / 2 - np.sqrt(4 * M01 ** 2 + (M00 - M11) ** 2) / 2
    return l1, l2


def _hessian_matrix_image(H_elems):
    """Convert the upper-diagonal elements of the Hessian matrix to a matrix.
    Parameters
    ----------
    H_elems : list of array
        The upper-diagonal elements of the Hessian matrix, as returned by
        `hessian_matrix`.
    Returns
    -------
    hessian_image : array
        An array of shape ``(M, N[, ...], image.ndim, image.ndim)``,
        containing the Hessian matrix corresponding to each coordinate.
    """

    image = H_elems[0]
    hessian_image = np.zeros(image.shape + (image.ndim, image.ndim))
    for idx, (row, col) in \
            enumerate(combinations_with_replacement(range(image.ndim), 2)):
        hessian_image[..., row, col] = H_elems[idx]
        hessian_image[..., col, row] = H_elems[idx]
    return hessian_image


def eigenValuesShapeIndexCurveness(sigmas, flat_nodule):
    #já não é preciso fazer o filtro gaussiano porque esta função faz 
    
    shapeindex = np.zeros((51,51,len(sigmas)))
    cv = np.zeros((51,51,len(sigmas)))
    eigValues = []
    
    for (i,s) in enumerate(sigmas):
        #já não é preciso fazer o filtro gaussiano porque esta função faz 
        #para os vários sigmas usados vamos guardar apenas o tuplo com os maiores valores de hrr, hrc, hcc
        h_elem = hessian_matrix(flat_nodule, sigma = s, order='rc')
        eigValues = hessian_matrix_eigvals(h_elem)
        shapeindex[:,:,i] = ((2/math.pi)*np.arctan((eigValues[0]+eigValues[1])/(eigValues[0])-eigValues[1]))
        aux = np.sqrt((np.power(eigValues[1],2)+(np.power(eigValues[0],2))))
        cv[:,:,i] = aux
        
    cv = np.max(cv, axis = -1)
    shapeindex = np.max(shapeindex, axis = -1) 
 
    return shapeindex,cv,eigValues
