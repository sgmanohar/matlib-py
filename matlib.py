import numbers
import numpy as np
from collections import Sequence
import matplotlib.pyplot as plt
import scipy

"""
Sanjay Manohar's matlib (2018-)
https://github.com/sgmanohar/matlib-py/blob/main/matlib.py

based on Matlab library (2007-) for n-dimensional and time-series workflow.
  nancat:          concatenate arrays with nan-padding
  errorBarPlot:    plot nd-array calculating mean & SEM on axis 0,
                   putting axis 1 as the x-axis, and axis 2 as different lines.
  conditionalPlot: plot (X,Y) using sliding windowed binning of Y.
  bool2nan:        map True:NaN, False:0
  smoothn          smooth array along axis=0
  gauss_kern       construct n-dimensional gaussian kernel
  interpnan        interpolate over NaNs along axis=0
  hampel           hampel filtering along axis=0
  lmplot_stats     scatter plot with linear regression and p-values
  
"""

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
        plot = True,
        plot_individuals = False,
        smoothing = 0,
        stats = False, **kwargs ):
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
    stats: True/False whether to include statistics. 
            
    Returns
    -------
    x_bin : bin centres for each bin x subject x condition
    y_bin : bin means   for each bin x subject x conditoin
    model : if stats==True, then return the linear model fitted to the data.
    
    stats: requires statsmodels package

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
        if smoothing > 0:
            # apply smoothing here
            my = smoothn(my, width=smoothing)
        if mx.shape[1]>1:  # are there subjects?
            errorBarPlot(my.transpose((1,0,2)), x=mx.transpose((1,0,2)), 
                         plot_individuals=plot_individuals,**kwargs)
        else: # just single line
            plt.plot(mx[:,0,0],my[:,0,0],**kwargs)
            plt.fill_between( mx[:,0,0], my[:,0,0]-ey[:,0,0] , my[:,0,0]+ey[:,0,0] ,
                             alpha = 0.2) # error area
            if plot_individuals:
                plt.scatter( x, y )
          
    if stats:
      import statsmodels.formula.api as smf
      import statsmodels.api as sm
      import pandas as pd
      # make subject and condition arrays by singleton expansion
      subj = np.arange(x.shape[1])[None,:,None] + 0*x
      cond = np.arange(x.shape[2])[None,None,:] + 0*x
      tmp_df = pd.DataFrame({
            'x':x.ravel(), 
            'y':y.ravel(), 
            's':subj.ravel(),
            'c':cond.ravel(),
            'ones': np.ones(x.ravel().shape)
      })
      tmp_df.s = tmp_df.s.astype('category') # convert subject and condition
      tmp_df.c = tmp_df.c.astype('category') # to categorical
      
      if mx.shape[1]>1: # several subjects?
        m=smf.mixedlm('y~x*c',data=tmp_df, groups=tmp_df.s, re_formula='~1').fit()
        print(sm.stats.anova_lm(m))
        print(m.summary())
        signif = m.pvalues[1:] < 0.05
      else: # just single subject
        m = smf.ols("y~x",data=tmp_df).fit()
        print(sm.stats.anova_lm(m))  
        signif = m.pvalues[1:2] < 0.05
      if any(signif):
        print("*")
      return mx, my, m
    return mx,my
  



def smoothn(X,width=5, kernel='uniform',
            remove_edges = False):
    """ 
    apply a smoothing kernel along axis 0.
    width: integer specifying number of samples in smoothing window
    kernel: 'uniform', 'gauss', or 1D-array
    remove_edges: replace NaN for edge values (makes no assumptions)
    """ 
    KERNELS = {
      'uniform':   lambda w: np.ones((int(w)))/w,
      'gauss':  lambda w: gauss_kern( w )
    }
    if type(kernel) == str:
      k = KERNELS[kernel](width)
    else:
      k = kernel
      width = len(k) # ignore width parameter 
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
    if remove_edges:
      Y[0:width,:] = np.nan
      Y[-width:,:] = np.nan
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
  
  

  
def epoch_1d(x, edges, left=0, right=0):
  """ 
  Divide a vector into regions, and stack the regions into a matrix.
   x:  the vector to epoch (1D np array)
   edges: boolean array same size as x. 
          =True at points to epoch on.
          These determine the starts and ends of the segments.
   left / right: move the epoch's left / right edges to include more/less datapoints. 
          e.g. left = -10    ==>  include 10 datapoints before each edge marker
               right = +20   ==>  include 20 datapoints after the end of each epoch
   result 2D array with nan padding with epoched data as columns. 
   """
  assert isinstance(x, np.ndarray), "x should be an numpy nd-array"
  original_shape = x.shape # store original shape in case we want to restore it
  # convert 1-d matrix to vector if needed
  if len(original_shape)==2 and original_shape[0]==1:
    x = x[0,:]
  elif original_shape[1]==1:
    x = x[:,0]
  assert len(x.shape)==1, "x should be a 1d vector"
  if ~any(edges):
    return x
  # Get list of edges
  edge = np.where(edges)[0]
  if len(edge)==0: # situation with no edges at all
    edge = np.array([0]) # add start point
  if not edge[0]==0: # Make sure first epoch starts at time zero
    edge = np.r_[ 0, edge ]
  if not edge[-1]==x.shape[0]:  # Make sure last epoch finishes at end
    edge = np.r_[ edge, x.shape[0] ]
  # Now do epoching
  out = [ ]   # accumulate epochs in here
  for e in range(len(edge)-1): # for each edge,
    le = np.max( edge[e] + left, 0)  # left edge
    re = np.min( edge[e+1] + right, x.shape[0] ) # right edge
    out.append(x[ le:re ] )  # extract the required datapoints
  return nancat(out,axis=1) # combine into new array
  
  
  
  
  
  

def interpnan(X, interpolator = np.interp):
  """
  Intepolate across regions of NaN, along axis = 0
  Parameters
  ----------
  X : nd array
    the input values, treated as an equally spaced series, 
    with missing values coded as NaN
    
  intepolator : an interpolator function 
    e.g. scipy.interpolate.CubicSpline, PchipInterpolator, Akima1DInterpolator,
    or make_interp_spline

  Returns
  -------
  nd array
    copy of input array, with nan-values interpolated.
  """
  if len(X.shape)>2: # if more than 1D, convert to 2D
    # call myself with the last dimensions flattened
    Xn = [ interpnan(xx) for xx in X.reshape(X.shape[0],-1).T  ]
    Y = np.array(Xn).reshape(X.shape) # then coerce back to original shape
    return Y # done!
  elif len(X.shape)<2:
    X=X[:,None]
  # run on 2D data:
  Y = X.copy()
  for i in range(Y.shape[1]):
    yo = Y[:,i] # get vector
    xo = np.arange(len(x)) # create time indices
    bad = np.isnan(yo) # remove nans from both
    yo = yo[~bad]
    xo = xo[~bad]
    xx = np.where(bad) # list of points to interpolate
    # there are different syntaxes for linear and other interpolators
    if interpolator is np.interp:
      Y[xx,i] = np.interp( xx, xo, yo )
    else:
      Y[xx,i] = interpolator( xo,yo )   ( xx )
  return Y


  
  
  


def hampel( X,
           half_window : int   = 3, 
           sigma       : float = 0.5,   
           k           : float = 1.4826  # assumption of distribution being gaussian
           ):
    """
    Run hampel filter on n-dimensional array, using axis=0 as the time dimension.  
    Hampel filter removes outliers as the deviation from the median 
    in a sliding window.
    Outliers are replaced by the median of this window.

    Parameters
    ----------
    X : nd array
    half_window : int. The default is 3.
    sigma : float, optional. The standard deviation threshold. The default is 0.5.
       higher value means less filtering
    k : float, optional
      DESCRIPTION. The default is 1.4826  # assumption of distribution being gaussian.
  
    Returns
    -------
    nd array -  filtered data
    """
    if len(X.shape)>2: # 3 or more dimensions?
      # condense to 2 D
      Xn = [ hampel(xx) for xx in X.reshape(X.shape[0],-1).T ]
      Y = np.array(Xn).T.reshape(X.shape) # restore N-dimensions
      return Y # finished!
    elif len(X.shape)<2: # one-dimensional? add a dimension.
      X = X[:,None]
    Y = X.copy()
    HW = int(half_window) # half-window
      
    def hampel_filter(x, HW=6, sigma=0.5):
        """
        Input:
            * x - values to filter  - must be a 1-dimensional np.array or list.
            * HW - int half-with size of the window
            * sigma - float defining whats counts as an outlier
        Output:
            * trace with removed outliers
        """
        x = np.array(x)
        cleaned_trace = x.copy()
        for i in range(HW, len(x)-HW):
            median_window = np.nanmedian(x[i-HW:i+HW])
            median_absolute_deviation = k * np.nanmedian(
                    np.abs(x[i-HW:i+HW]-median_window)
                )
    
            if (np.abs(x[i]-median_window) > sigma*median_absolute_deviation):
                cleaned_trace[i] = median_window
    
        # first and last half of the window is replaced by median of this half of window
        # this is used to remove potential outliers at these parts of the time trace
        cleaned_trace[:HW]  = np.nanmedian(cleaned_trace[:HW])
        cleaned_trace[-HW:] = np.nanmedian(cleaned_trace[-HW:])
    
        return cleaned_trace
      
    for i in range(Y.shape[1]): # for each time trace, 
      # run the hampel filter.
      Y[:,i] = hampel_filter(Y[:,i], HW=HW, sigma=sigma)
      
    return Y
  



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
  
