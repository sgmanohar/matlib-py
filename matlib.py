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


def errorBarPlot(Y,x=None, within_subject=True, plot_individuals=False):
    """plot mean and standard error along dimension 0.
       Y [ subjects, x_levels, lines ]   = the data to plot
       within_subject: if True, subtract subject means before calculating s.e.m.
       x : x values corresponding to the columns (axis 1) of Y """
    
    print(x.shape,Y.shape)
    if x is None:       # default X values are integers
        x = np.arange(Y.shape[1])
    if x.shape==Y.shape : # do they have a value per bin?
        x = np.nanmean(x,axis=0)
    if len(Y.shape)<3:
        Y=Y[:,:,None]   # ensure Y is at least 3D
    mY = np.nanmean(Y,axis=0) # calculate mean of Y across subjects
    if within_subject: # use within subject errors?
        dY = Y-np.nanmean(Y,axis=(1,2))[:,None,None] # subtract subject mean
    else: # use total error
        dY = Y
    sY = np.nanstd(Y,axis=0) / np.sqrt(Y.shape[0]) # calculate standard error
    hplot = plt.plot(x, mY) # draw lines
    nlines = mY.shape[1]
    for i in range(nlines): # draw error areas
      if len(x.shape)>1:
          plt.fill_between(x[:,i], mY[:,i]+sY[:,i], mY[:,i]-sY[:,i],alpha=0.3)
      else:
          plt.fill_between(x, mY[:,i]+sY[:,i], mY[:,i]-sY[:,i],alpha=0.3)
    if plot_individuals:
        for sub in range(Y.shape[0]):
            plt.gca().set_prop_cycle(None)
            plt.plot(x, Y[sub], linewidth=0.5)

            
def bool2nan(b):
    """ convert boolean True --> nan, False --> 0 """
    x = np.zeros(b.shape)
    x[b] = np.nan
    return x

def conditionalPlot(x,y,
        bin_width = 0.2,
        n_bins = 20,
        x_mean_function = np.nanmean,
        y_mean_function = np.nanmean,
        plot = True
                     ):
    """
    Bin and plot the y values against the x values. 
    
    Parameters
    ----------
    x : ndarray datapoints x subjects x conditions
        values corresponding to those in y, that will be used to bin y
    y : ndarray datapoints x subjects x conditions
        values to be averaged and plotted within the bins
        
    n_bins : number of bins i.e. step of sliding window (default 20)
    bin_width : proportion of datapoints (0-1) contained in each window 
             i.e. window width as a percentile (default 0.2)
    y_mean_function, x_mean_function : function to apply to the values in each
            window to get the plotted / returned value (default np.nanmean)
    plot : True/False whether to plot (default True)
            
    Returns
    -------
    x_bin : bin centres for each bin x subject x condition
    y_bin : bin means   for each bin x subject x conditoin

    """
    # if shape is too small, create an interface to a 3D array
    if len(x.shape)==2: # no conditions?
        x = np.array(x, ndmin=3, copy=False).transpose([1,2,0])  # add one dimension at the end
        y = np.array(y, ndmin=3, copy=False).transpose([1,2,0]) 
    elif len(x.shape)==1: # no subjects or conditions?
        x = np.array(x, ndmin=3, copy=False).transpose([2,0,1])  # add two dimensions at the end
        y = np.array(y, ndmin=3, copy=False).transpose([2,0,1]) 
    
    quantiles_left  = np.linspace( 0, 1-bin_width,n_bins+1 )
    quantiles_right = np.linspace( bin_width, 1,  n_bins+1 )
    qx_l = np.nanquantile(x, quantiles_left, axis=0, interpolation='linear')
    qx_r = np.nanquantile(x, quantiles_right, axis=0, interpolation='linear')
    qx_r[-1] += 1  # ensure that 'less than' will include the rightmost datapoint
    mx = np.zeros( (n_bins, *y.shape[1:]) )
    my = np.zeros( (n_bins, *y.shape[1:]) )
    ey = np.zeros( (n_bins, *y.shape[1:]) )
    n  = np.zeros( (n_bins, *y.shape[1:]) ) # keep count of how many in each bin
    for i in range(n_bins):
        edge_left  = qx_l[i]  # in range 0 to 1
        edge_right = qx_r[i] if i<n_bins else inf
        select = ( (x >= edge_left)
                 & (x <  edge_right) )
        # now take means of the selected values to give                    
        # bin x subject x condition
        mx[i,:,:] = x_mean_function( x + bool2nan(~select), axis = 0 ) # mean x
        my[i,:,:] = y_mean_function( y + bool2nan(~select), axis = 0 ) # mean y
        n[i,:,:]  = np.sum(select,axis=0) # count of datapoints
        ey[i,:,:] = np.nanstd( y[select] , axis=0) / np.sqrt(n[i,:,:]) # standard error
        
    if plot:
        if mx.shape[1]>1:  # are there subjects?
            errorBarPlot(my.transpose((1,0,2)), x=mx.transpose((1,0,2)))
        else: # just single line
            sns.lineplot(x=mx[:,0,0],y=my[:,0,0])
            plt.fill_between( mx[:,0,0], my[:,0,0]-ey[:,0,0] , my[:,0,0]+ey[:,0,0] ,
                             alpha = 0.2) # error area
    return mx,my
  


def smoothn(X,width=5, kernel='uniform'):
    """ 
    apply a smoothing kernel along axis 0.
    
    """ 
    KERNELS = {
      'uniform':   lambda w: np.ones((int(w)))/w,
      'gauss':  lambda w: gauss_kern( w )
    }
    k = KERNELS[kernel](width)
    shp = X.shape
    x2 = X.copy().reshape(X.shape[0],-1)
    Y  = x2.copy()
    g  = x2.copy()
    goodness = ~np.isnan(x2)
    x2[np.isnan(x2)] = 0 # trick the algorithm into ignoring nans
    for i in range(x2.shape[1]):
      # the trick here is to do a second convolution on a 0/1 array to 
      # find out how many valid data points there were, and normalise by this
      Y[:,i] = np.convolve( x2[:,i],       k, mode='same' )
      g[:,i] = np.convolve( goodness[:,i], k,  mode='same')
    #Y[0:width,:] = np.nan
    #Y[-width:,:] = np.nan
    Y = Y / g
    Y[ g==0 ] = np.nan
    Y = Y.reshape(shp)
    return Y
  
def gauss_kern(N=10,S=0.4):
  """
  n-D Gaussian kernel, 
   * N = width of kernel - either a single integer, or a list / tuple
         with one value per dimension. e.g.  N=(10,10) gives a 10 x 10 kernel.
   * S = standard deviation S (as a proportion of the width).
         defaults to 0.4. Can be specified for each dimension separately.
  Values are normalised to add up to 1
  if S defaults to 1/3 of the width.
  """
  if not hasattr(N,"__len__"): # check if N is a scalar
    N=(N,)
  if not hasattr(S,"__len__"): # if S is scalar, use it for every dimension
    S = [S] * len(N)
  from scipy.stats import norm
  xs = []
  for i,n in enumerate(N):
    dims = [ 1 ] * len(N)
    dims[i] = n
    support = np.linspace(-1,1,n) / S[i] # 1-dimensional support array
    pdf = norm.pdf( support ) # convert to probabliity
    xs.append(  pdf.reshape(dims) ) #
  import math
  p = math.prod(xs)  # not the numpy prod, which will multiply every inner element!
  p = p / np.sum(p)
  return p
  
  

  

  
  
  
  

def hampel( X,
           half_window : int   = 3, 
           sigma       : float = 0.5,   
           k           : float = 1.4826  # assumption of distribution being gaussian
           ):
    """ 
    run hampel_filter on each column of the data X. X is an n-dimensional array.
    Return: an n-dimensional array.
    """
    if len(X.shape)>1:
      Xn = [ hampel(xx) for xx in X.reshape(X.shape[0],-1) ]
      Y = np.array(Xn).reshape(X.shape)
      return Y
    Y = X.copy()
    
    for i in range(X.shape[1]):
      Y[:,i] = hampel_filter(X[:,i], 1, window_size=half_window, sigma=sigma)
      
    return Y
  
  
def hampel_filter(x, ratio, window_size=6, sigma=0.5):
    """
    Hampel filter removes outliers as the deviation from the median 
    in a sliding window.
    Outliers are replaced by the median of this window.
    Input:
        * x - values to filter  - must be a 1-dimensional np.array or list.
        * window_size - int with size of the window
        * sigma - float defining whats counts as an outlier
    Output:
        * trace with removed outliers
    """
    x = np.array(x)
    cleaned_trace = x.copy()

    window_size = window_size*ratio
    HW = int(window_size) # half-window


    for i in range(HW, len(x)-HW):

        median_window = np.nanmedian(x[i-HW:i+HW])
        median_absolute_deviation = k * \
            np.nanmedian(
                np.abs(x[i-HW:i+HW]-median_window))

        if (np.abs(x[i]-median_window) > sigma*median_absolute_deviation):
            cleaned_trace[i] = median_window

    # first and last half of the window is replaced by median of this half of window
    # this is used to remove potential outliers at these parts of the time trace
    cleaned_trace[:HW] = np.nanmedian(cleaned_trace[:HW])
    cleaned_trace[-HW:] = np.nanmedian(cleaned_trace[-HW:])

    return cleaned_trace



def lmplot_stats(x, y, data, **kwargs):
  """ 
  calls seaborn lmplot, and adds p values
  requires data=, x='name', and y='name' as kwargs.
  also can use row, col
  """
  import seaborn as sns
  import pandas as pd
  import statsmodels.formula.api as smf
  import statsmodels.api as sm
  import scipy
  bad = np.isnan(data[x]) | np.isnan(data[y])
  g = sns.lmplot(x=x, y=y,data=data.loc[~bad,:], **kwargs)
  
  def annotate(data, **kws):
      r, p = scipy.stats.pearsonr(data[x], data[y])
      ax = plt.gca()
      ax.text(.05, .8, 'r={:.2f}, p={:.2g}'.format(r, p),
              transform=ax.transAxes)     
  g.map_dataframe(annotate)
  
