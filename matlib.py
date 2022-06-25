import numbers
import numpy as np
from collections import Sequence
def nancat(X, axis=0, pad_val=np.nan):
  """ concatenate a list of numpy arrays, with padding 
      This function concatenates nd-arrays into a larger array, like concatenate.
      If the arrays have different sizes, pad the missing values with NaN.
      It should work for any number of dimensions. 
      Inputs:
        X = a list or sequence or numpy arrays. 
        axis = which axis to concatenate along (default 0)
        pad_val = the numeric scalar value to fill missing data (default np.nan)
      
      e.g.
      >>> nancat( ( np.array( [[ 0, 1 ],
                               [ 2, 3 ]] ), 
                    np.array( [[ 4, 5, 6]] ) ), axis=0
                    
      array([[ 0.,  1., nan],
             [ 2.,  3., nan],
             [ 4.,  5.,  6.]])
             
      You can nest lists. Inner lists of depth i are treated as being
      along the higher dimensions axis + i.
      e.g.
      >>> nancat(  [ np.array([1,2,3]),
                     [ np.array([4,5]), 
                       np.array([6])   ]
                   ], axis=1 )
          
      array([[[ 1., nan],
              [ 4.,  6.]],

             [[ 2., nan],
              [ 5., nan]],

             [[ 3., nan],
              [nan, nan]]])
      Sanjay Manohar 2022
  """
  def cat_pair( X,Y, axis, pad_val):
    """helper function to concatenate two arrays"""
    sX = X.shape
    sY = Y.shape
    if len(sX)<len(sY): # ensure same number of dimensions
        X=X[...,np.newaxis]
        return cat_pair(X,Y,axis=axis, pad_val=pad_val)
    elif len(sX)>len(sY):
        Y=Y[...,np.newaxis]
        return cat_pair(X,Y,axis=axis,pad_val=pad_val)
    nD = len(sX)  # number of dimensions.
    if axis>=nD:  # ensure that we have enough to do the concatenation!
        X=X[...,np.newaxis]
        Y=Y[...,np.newaxis]
        return cat_pair(X,Y,axis=axis, pad_val=pad_val)
        
    # compute new size
    sZ = np.maximum(np.array(sX),np.array(sY))
    sZ[axis] = sX[axis] + sY[axis]
    # now add padding 
    for i in range(nD): # for each dimension
        if i==axis:    # don't pad dimension that we are concatenating along
            continue
        if i>=len(sX) or i>=len(sY):
            breakpoint()
        if sX[i] < sY[i]: # if Y is bigger
            sX = X.shape
            padsize = np.array(sX)
            padsize[i] = sY[i]-sX[i]  # add padding to X
            pad = pad_val * np.ones( padsize )
            X=np.concatenate((X, pad ),axis=i)
        elif sX[i] > sY[i]: # if X is bigger
            sY = Y.shape
            padsize = np.array(sY)
            padsize[i] = sX[i]-sY[i]  # add padding to Y
            pad = pad_val * np.ones( padsize )
            Y=np.concatenate((Y, pad ), axis=i)
    # now we should be in a position to concatenate as usual...
    Z = np.concatenate( (X,Y), axis=axis )
    return Z
  def ensure_np(i):
      """if the argument is a sequence, call nancat recursively on it.
         if it isn't an array or number, raise an error."""
      if not isinstance(i,np.ndarray):
        if isinstance(i,Sequence):
            # I am unclear whether this should be axis + 1, or something else + 1.
            i = nancat(i,axis=axis+1,pad_val=pad_val)  
        elif isinstance(i,numbers.Number): # promote scalars to np array
            i = np.array(i)
        else:
            raise ValueError("first arg must contain sequence or np array")
      return i
  # Now the main bit
  if len(X)==0:
    return np.array([])
  Y = ensure_np(X[0])  # first element of sequence
  for i in X[1:]: # for each subsequent element,
     i = ensure_np(i) # convert to np array
     Y=cat_pair(Y,i,axis=axis,pad_val=pad_val) # and concatenate 
  return Y 

# test case
nancat(  [np.array([1,2,3])[:,None].T,[np.array([4,5]),np.array([6])]], 
          axis=1 )
