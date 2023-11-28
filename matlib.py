import numbers
import numpy as np
try:
  from collections import Sequence
except ImportError:
  from collections.abc import Sequence
import matplotlib.pyplot as plt
import scipy
import warnings
import datetime
"""
Sanjay Manohar's matlib (2018-)
https://github.com/sgmanohar/matlib-py/blob/main/matlib.py

based on Matlab library (2007-) for n-dimensional and time-series workflow.
  nancat:          concatenate arrays with nan-padding. Handles nested lists.
  errorBarPlot:    plot nd-array calculating mean & SEM on axis 0,
                   putting axis 1 as the x-axis, and axis 2 as different lines.
  conditionalPlot: plot (X,Y) using sliding windowed binning of Y. 
                   Treats axis 0 as samples, axis 1 as subjects, axis 3 as conditions.
                   Also fit linear mixed effects model. 
  bool2nan:        map True:NaN, False:0
  smoothn          Smooth array along axis=0. Nan-friendly.
  gauss_kern       construct n-dimensional gaussian kernel
  interpnan        interpolate over NaNs along axis=0
  hampel           hampel filtering along axis=0
  lmplot_stats     scatter plot with linear regression and p-values
  scatter_regress  
  slide_match      find where two arrays match by sliding one over another
  simple_ols       ordinary least squares regression, for large data matrices
  epoch_1d         divide a vector into regions, and stack the regions into a matrix.
  sprint           structured print - drill down into nested lists and dicts
  permutation_ols  permutation test to correct for multiple comparisons in OLS regression
  show_principal
  crossval_logistic_classifier      - logistic regression with LOO-CV and AUROC
  show_feature_principal_components - simple PCA visualisation with optimal leaf order and scree
"""
#### CODE SNIPPETS YOU MIGHT NEED
# To re-import matlib, you might like to do something like:
if False:
  import importlib; import matlib; importlib.reload(matlib); from matlib import *
# To get SVG graphics you might want to do something like:
if False: ####   imports I will need for basic data science
  from pylab import *
  import pandas as pd
  import seaborn as sns
  import numbers
  import statsmodels.formula.api as smf
  import statsmodels.api as sm
  import scipy.stats
  import scipy.linalg
  import pickle
  import os
  import sys


def setup_for_science():
  """
  This function sets up some useful defaults for numpy and pandas.
  """
  import matplotlib_inline 
  import pandas as pd
  matplotlib_inline.backend_inline.set_matplotlib_formats('svg') 
  import matplotlib.pyplot as plt
  plt.rcParams['svg.fonttype'] = 'none' # to allow editing in inkscape
  np.set_printoptions(precision=3, suppress=True, linewidth=200, edgeitems=10)
  pd.set_option('display.max_columns', 100)
  pd.set_option('display.max_rows', 100)
  pd.set_option('display.width', 256)
  pd.set_option('display.max_colwidth', 10)
  pd.set_option('display.expand_frame_repr', False)
  pd.set_option('display.precision', 3)
def install_package(package):
  """ install with pip using the current python interpreter"""
  import sys
  import subprocess
  subprocess.check_call([sys.executable, "-m", "pip", "install", package])


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
        if i>=len(sX) or i>=len(sY): # enough dimensions?
            breakpoint() # this should never happen!
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


def errorBarPlot(Y,x=None, within_subject=True, plot_individuals=False,
                 alpha=0.3, quantiles = None, **kwargs, ):
    """plot mean and standard error along dimension 0.
       Y [ subjects, x_levels, lines ]   = the data to plot
       within_subject: if True, subtract subject means before calculating s.e.m.
       x : x values corresponding to the columns (axis 1) of Y """
    
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
          xvals = x[:,i]
      else:
          xvals = x
      if quantiles is None: # use standard error
        plt.fill_between(xvals, mY[:,i]+sY[:,i], mY[:,i]-sY[:,i],alpha=alpha)
      else: # use quantiles
        if quantiles == True:
           quantiles = [0.2,0.3, 0.4]
        for j in quantiles:
          q_low  = np.nanquantile(dY,   quantiles[j], axis=0)
          q_high = np.nanquantile(dY, 1-quantiles[j], axis=0)
          plt.fill_between(xvals, q_low, q_high,alpha = alpha/(len(quantiles)) )
         
    if plot_individuals:
        for sub in range(Y.shape[0]):
            plt.gca().set_prop_cycle(None)
            plt.plot(x, Y[sub], linewidth=0.5)

            
def bool2nan(b):
    """ convert boolean True --> nan, False --> 0 """
    x = np.zeros(b.shape)
    x[b] = np.nan
    return x

def conditionalPlot_df(data, x, y, group=None, do_stats=True, **kwargs):
  """
  Plot x against y, with different lines for each group.
    data: dataframe
    x: column name for x-axis
    y: column name for y-axis
    group: column name for grouping variable
  """
  if group is None:
    output = conditionalPlot(data[x], data[y], **kwargs )
  else:
    grp = data[group]
    unique_grp = np.unique(grp)
    output = []
    for g in unique_grp:
        output.append( conditionalPlot(data[x][grp==g], data[y][grp==g], **kwargs ) )
    # double up the list for shading
    plt.legend([y for z in  [ [x,''] for x in unique_grp] for y in z ])
  plt.xlabel(x)
  plt.ylabel(y)
  if do_stats: # display linear model result
    from  statsmodels.formula.api import ols
    if group is None:
      print(ols(f'{y} ~ {x}', data).fit().summary().tables[1])
    else:
      print(ols(f'{y} ~ {x} * C({group},Sum)', data).fit().summary().tables[1])
  return output

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
    
    extra keyword arguments get sent to errorBarPlot().
            
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
    quantileargs = {'interpolation':'linear'} # old versions of numpy
    quantileargs = {'method':'interpolated_inverted_cdf'} # new versions
    qx_l = np.nanquantile(x, quantiles_left , axis=0, **quantileargs)
    qx_r = np.nanquantile(x, quantiles_right, axis=0, **quantileargs)
    qx_r[-1] += 1  # ensure that 'less than' will include the rightmost datapoint
    mx = np.zeros( (n_bins, *y.shape[1:]) )
    my = np.zeros( (n_bins, *y.shape[1:]) )
    ey = np.zeros( (n_bins, *y.shape[1:]) )
    n  = np.zeros( (n_bins, *y.shape[1:]) ) # keep count of how many in each bin
    if len(qx_l)==1:
      # Unexpectedly, nanquantile returns a single [nan], instead of an array of nan,
      # when there is not enough data... so sadly we have to deal with it.
      warnings.warn('not enough data for conditional plot')
      return [],[]
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
            my = smoothn(my, width=smoothing )
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
    apply a 1D smoothing kernel along axis 0.
    width:     integer specifying number of samples in smoothing window
    kernel:    'uniform', 'gauss', or 1D-array
    remove_edges: replace NaN for edge values (makes no assumptions)
    """ 
    KERNELS = {
      'uniform':   lambda w: np.ones((int(w)))/w,
      'gauss':     lambda w: gauss_kern( w )
    }
    if type(kernel) == str:
      k = KERNELS[kernel](width)
    else:
      k = kernel
      width = len(k) # ignore width parameter 
    shp = X.shape
    x2 = X.copy().reshape(X.shape[0],-1) # collapse last dimensions to give 2D
    if X.dtype == int:
      warnings.warn('smoothn: integer being converted fo float')
      x2.dtype = float
    Y  = x2.copy()
    g  = x2.copy()
    # @TODO next pair of lines can be modified to handle confidence / uncertainty.
    # eg. : goodness = confidence ;  x2 = x2 * confidence
    # and we can also infer confidence on the output trace eg. 1/c_out = sqrt(sum((1/c_in)^2))
    goodness = ~np.isnan(x2)  # points that are good
    x2[np.isnan(x2)] = 0 # trick the algorithm into ignoring nans
    for i in range(x2.shape[1]): # for each column
      # the trick here is to do a second convolution on a 0/1 array to 
      # find out how many valid data points there were, and normalise by this
      Y[:,i] = np.convolve( x2[:,i],       k, mode='same' )
      g[:,i] = np.convolve( goodness[:,i], k,  mode='same')
    #import pdb; pdb.set_trace()
    if remove_edges:
      Y[0:width,:] = np.nan
      Y[-width:,:] = np.nan
    Y = Y / g          # normalise by number of good points
    Y[ g==0 ] = np.nan # remove points where no data available
    Y = Y.reshape(shp) # convert back to original shape
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
    Y = np.array(Xn).T.reshape(X.shape) # then coerce back to original shape
    return Y # done!
  elif len(X.shape)<2: # if 1D, make 2D
    X=X[:,None]
  # run on 2D data:
  Y = X.copy()
  for i in range(Y.shape[1]): # for each column
    yo = Y[:,i] # get vector
    xo = np.arange(len(x)) # create time indices
    bad = isnan(yo) # remove nans from both
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
  

def lmplot_stats(x, y, data=None, **kwargs):
  """ 
  calls seaborn lmplot, and adds p values
  requires data=, x='name', and y='name' as kwargs.
  also can use x,y as separate columns of numbers (pd.series or np.array)

  """
  import seaborn as sns
  import pandas as pd
  import statsmodels.formula.api as smf
  import statsmodels.api as sm
  import scipy
  
  if data is None: 
    # convert from x,y into a dataframe if needed
    if isinstance(x, pd.Series) and isinstance(y, pd.Series):
      # if series as input, make a dataframe by concatenation
      data = pd.concat((x,y), axis=1)
      x=x.name
      y=y.name
    else: # if x/y is numeric,
      data = pd.DataFrame(data=np.stack((x,y),axis=1), columns=['x','y'])
      x='x' # create new dataframe
      y='y'
  bad = np.isnan(data[x]) | np.isnan(data[y])
  g = sns.lmplot(x=x, y=y,data=data.loc[~bad,:], **kwargs)
  
  def annotate(data, **kws):
      r, p = scipy.stats.pearsonr(data[x], data[y])
      ax = plt.gca()
      ax.text(.05, .8, 'r={:.2f}, p={:.2g}'.format(r, p),
              transform=ax.transAxes)     
  g.map_dataframe(annotate)


def regplot_stats(x,y,data=None, **kwargs):
  import seaborn as sns
  import pandas as pd
  import scipy.stats
  g=sns.regplot(x=x,y=y,data=data, **kwargs)
  bad = np.isnan(data[x]) | np.isnan(data[y])
  def annotate(data, **kws):
    r, p = scipy.stats.pearsonr(data[x], data[y])
    ax = plt.gca()
    ax.text(.05, .8, 'r={:.2f}, p={:.2g}'.format(r, p),
            transform=ax.transAxes)     
  annotate(data.loc[~bad,:])

def scatter_regress(x=None, y=None, data=None, jitter = 0, **kwargs):
  """ 
  calls seaborn scatterplot, ols to get fit line, and pearson for stats.
  syntax: data=DataFrame, x='name', and y='name' 
  or:     x=Series, y=Series 
  or:     x=column array, y = column array
  or:     data=DataFrame -- plots all pairs of columns as a PairGrid
  
  """
  import seaborn as sns
  import pandas as pd
  import statsmodels.formula.api as smf
  import statsmodels.api as sm
  import scipy
  import pylab
  def looks_discrete(x):
    """ does a column of values look discrete? 
    what proportion of items are unique? 
    if all unique, then return 0, as it looks continuous.
    if more than 50% are repeated values, then return +1, as it looks discrete """
    f_different = len(unique(x))/len(x)
    return 1-minimum(1, 2*f_different)
  
  if data is None: 
    # convert from x,y into a dataframe if needed
    if isinstance(x, pd.Series) and isinstance(y, pd.Series):
      # if series as input, make a dataframe by concatenation
      if x.name is None: # can happen if series has no name
        x=x.copy()
        x.name = 'x'
      if y.name is None:
        y = y.copy()
        y.name = 'y'
      data = pd.concat((x,y), axis=1)
      x=x.name
      y=y.name
    else: # if x/y is numeric,
      data = pd.DataFrame(data=np.stack((x,y),axis=1), columns=['x','y'])
      x='x' # create new dataframe
      y='y'
  else:
    if isinstance(x,str) and isinstance(y,str): # already got column labels in a dataframe
      data = data.copy()
    elif x is None and y is None: # we have a dataframe, but not column labels
      grid=sns.PairGrid( data )
      grid.map_upper( scatter_regress , **kwargs) # recurse to plot single scatter plot in upper triangle
      def lower(x,y, **kwargs):
         conditionalPlot(x,y) # line 
         sns.kdeplot(x=x,y=y, fill=True, alpha=0.3, **kwargs) # area
      grid.map_lower( lower  , **kwargs ) 
      grid.map_diag( sns.kdeplot, fill=True, alpha=0.3 , common_norm=False) # histograms
      return
    else:
      raise ValueError("scatter_regress: x/y should be strings or None.")
  bad = np.isnan(data[x]) | np.isnan(data[y])
  # best fit line - use least squares to predict
  model = smf.ols(y+'~'+x,data).fit()
  # calculate r and p from pearson 
  if sum(~bad)>2:
    r, p = scipy.stats.pearsonr(data[x][~bad], data[y][~bad])
  else:
    r = np.nan
  if jitter: # add jittering?
    data[x] += pylab.rand(*data[x].shape) * jitter
    data[y] += pylab.rand(*data[y].shape) * jitter
  
  g = sns.scatterplot(x=x, y=y,data=data.loc[~bad,:], alpha=0.3, **kwargs)
  ypred = model.predict( pd.DataFrame({ x:plt.xlim()})  )
  if model.pvalues[1]<0.05:
    sns.lineplot(x=plt.xlim(),y=ypred)  
  if ~np.isnan(r):
    ax = plt.gca()
    ax.text(.05, .8, 'r={:.2f}, p={:.2g}'.format(r, p),
              transform=ax.transAxes)     





def sprint(x, nest=0, id_history = []):
  """
  sprint - structure printer. Goes through the first element of each "listy" item, 
  and shows its size. It currently iterates through lists, tuples and dicts.
  It uses getsizeof to find the largest item in a list, to drill down into it.
    That makes it quite slow... but worth the wait.
    
  Sanjay Manohar 2023
  """
  from sys import getsizeof
  import pandas as pd
  import types

  max_nest = 6 # @todo this should be a parameter 
  new_id = id(x) # ensure we don't recursively enter the same object
  if new_id in id_history: 
       return 
  else:
       id_history.append(new_id)
  if nest > max_nest:
     print((' '*nest) + "...")
     return
  if isinstance(x, (list,tuple)): # list items
    print( (' '*nest ) + f"{len(x)}["  )
    if len(x)>0:
        # get the biggest element in the list
        which_index = np.argmax( np.array( [getsizeof(i) for i in x] ) )
        # get the example item and recursively print it
        next_x = x[which_index]
        #import pdb; pdb.set_trace()
        sprint(next_x,nest+1,id_history)
    print( (' '*nest) + ']' )
  elif isinstance(x,dict): # dictionary contents
    print( (' '*nest) + "{" )
    keylist = list(x.keys()) # keys in the dict
    # if the dict has a lot of keys, or if the keys seem to be numeric,
    if (len(keylist)>8) or any(c.isdigit() for c in keylist[0]):
       # show the range of keys
       which_key = np.argmax( np.array([ getsizeof(x[k]) for k in keylist ]) )
       print((' '*nest)+f"'{keylist[0]}'...{len(keylist)}...'{keylist[-1]}': ")
       # and then show one example value
       sprint(x[keylist[which_key]], nest+1, id_history)
    else: # If there are only a few keys, and the keys don't have numbers in, 
        # go through every key of this item
        for k,v in x.items():
            # recursively print it
            print((' '*nest)+k+": ",  end="")
            sprint(x[k], nest+1, id_history)
    print( (' '*nest) + '}')
  elif isinstance(x, pd.DataFrame): # pandas dataframe
    # display dataframe size and column names
    print( (' '*nest) + f"pd[{x.shape[0]}x{x.shape[1]}] " + ",".join(x.columns) )
  elif isinstance(x, pd.Series): # pandas series
    # display series size and name
    print( (' '*nest) + f"pd[{len(x)}] " + x.name )
  elif isinstance(x, np.ndarray): # numpy object
    # display array size
    print( (' '*nest) + "np[" + ",".join([f"{i}" for i in x.shape]) + "]" )
  elif isinstance(x,str):
    # display string length
    if len(x)<40: # short strings can be displayed
       print( (' '*nest) + str )
    else: # long strings show just the length
      print( (' '*nest) + f"str[{len(x)}]" )
  elif callable(x):
    print( (' '*nest) + x.__doc__.split("\n")[0] ) # only show the first line of the docstring
  elif isinstance(x, (numbers.Number, datetime.datetime)): # a primitive type - number or date
    # don't bother going down the hierarchy!
    print( (' '*nest) + f"{type(x).__name__}={repr(x)}")
  elif isinstance(x,types.ModuleType):
    # don't drill far into modules, but allow functions
    fields = [i for i in dir(x) if not i.startswith("_")  ]
    print((' '*nest) + f"{x} fields[{len(fields)}]")
  elif isinstance(x,object):
    # generic objects - they could have interesting attributes. List them
    # but don't include functions or private attributes
    fields = [i for i in dir(x) if not i.startswith("_") and not callable(getattr(x,i)) ]
    # and for each attribute
    for f in fields:
      print((' '*nest) + f + ": ", end="")
      try:
        sprint(getattr(x,f), nest+1, id_history) # recursively display it
      except:
        #print("err")
        pass # print("")
  else:
    # this should never happen - everything is an object, right? 
    raise RuntimeError(f"Unknown type {type(x)}") 

def slide_match(x,y):
  """ slide two 1-dimensional arrays along each other to find best match.
  x = large array, in which to find y.
  Example: if x = [1,2,3,4,5,6,7,8,9,10] and y = [3,4,5], then the best match
  is at position 2, where x[2:5] = [3,4,5], and the match proportion is 1.0.
  If x = [1,2,3,4,5,6,7,8,9,10] and y = [7,6,5], then the best match is at
  any of positions 6,7,8, e.g. where x[6] = 6, and the match proportion is 1/3.
  """
  x=np.array(x) # ensure numpy arrays
  y=np.array(y)
  assert(len(x.shape)==1) # ensure 1 dimensional
  assert(len(y.shape)==1)
  if len(x)<len(y): # ensure x is at least as large as y
    x = np.r_[x, np.nan * np.ones(len(y)-len(x))]
  # how many different positions are there, if you slide y along x?
  num_slides = len(x)-len(y) + 1 
  # empty array for calculating the proportion of words that match, 
  # for each slide potition.
  match_prop = np.nan * np.zeros(num_slides)
  # for each sliding position
  for i in range( num_slides ):
    # calculate what proportion of the words in y match the words in x
    match_prop[i] = np.mean( x[i:(i+len(y))] == y )
  return match_prop.max()    
    
def show_feature_principal_components(df):
  # remove non numeric columns
  df = df.loc[:, [t.kind in 'biufc' for t in df.dtypes] ]
  # clean data - first remove columns with zero variance
  df = df.loc[:,df.std()>0]
  # now zscore each column  and replace nan with zero (i.e. imputation to mean)
  dfz = (df - df.mean()) / df.std()
  dfz[np.isnan(dfz)] = 0
  # get correlation matrix
  C = dfz.corr()
  # flip columns that are negatively associated with others
  flip = mean(C>0,axis=1) < 0.5
  dfz.loc[:,flip] = -dfz.loc[:,flip]
  C = dfz.corr()
  # calculate optimal leaf ordering
  z = scipy.cluster.hierarchy.ward(dfz.T)
  olo = scipy.cluster.hierarchy.leaves_list(z)
  #olo = scipy.cluster.hierarchy.optimal_leaf_ordering( 
  #      scipy.cluster.hierarchy.ward(dfz.T), dfz.T )
  # reorder rows and columns
  C = C.iloc[olo,olo]
  subplot(2,2,1)
  # show heatmap with a divergent colormap
  sns.heatmap(C, cmap='RdBu_r', center=0)
  try:
    plt.colorbar()
  except:
    pass
  subplot(2,2,2)
  # get eigenvalues and eigenvectors
  w,v = np.linalg.eig(C)
  # scree plot of eigenvalues 
  plt.plot(w,'o-')
  # show significant values with asterisks
  plt.plot(np.where(w>1)[0],w[w>1],'*')
  plt.ylabel("eigenvalue")
  plt.xlabel("component")
  # show principal components as heatmap
  subplot(2,2,3)
  # scale components by their eigenvalues
  v = v * w[None,:]
  sns.heatmap(v , cmap='RdBu_r', center=0)
  # calculate scores 
  scores = dfz.values @ v
  # scatter datapoints in the first two components
  subplot(2,2,4)
  plt.scatter(scores[:,0],scores[:,1])
  plt.xlabel("PC1")
  plt.ylabel("PC2")

    

def crossval_logistic_classifier(df,  # dataframe
          group,     # column with labels to predict
          predictors = None, # list of column names to use as predictors
          classify=None,   # which two groups to classify
          do_plot=True,
          normalise_predictors_axis = None, # subtract predictor means (along one axis) before classifying
          normalise_divisive = False, # divide by mean instead of subtracting
          resample_all = False # if true, both classes are resampled to match the size of the first class.
           # if false, only the second class is resampled, and it is done without replacement.

):
  """
  crossval_logistic_classifier(df, group, predictors = None, classify=None)
  Cross-validated logistic regression classifier.
  Balances the two classes by subsampling items with classify[1] to match the number of items with classify[0].
  Then fits a logistic regression model to predict the group from the predictors.
  Then predicts the group of each item, using leave-one-out cross-validation.
  Then plots the ROC curve, and returns the dataframe with predictions.
  If classify is not specified, it uses the first two labels in the group column.
  If predictors is not specified, it uses all columns except group.
  Return:
    df: dataframe with predictions column as 'pred'
    auc: area under the ROC curve
    acc: accuracy of prediction 
  """
  import pandas as pd
  import statsmodels.formula.api as smf
  import statsmodels.api as sm
  df = df[ df[group].isin(classify) ].copy() # remove irrelevant rows (creating a copy)
  df = df[ [group] + list(predictors) ] # select relevant columns
  if classify == None: # if not specified, use first two labels
     classify = df[group].unique()[0:2] 
  if predictors == None: # if not specified, use all columns except group
    predictors = df.columns[ df.columns!=group ] 
  yvar = 'is'+classify[0]
  df[yvar] = (df[group] == classify[0]).astype(int) # gives 1 if it's the first class.
  if 'pred' in df.columns: # if there is already a prediction column, give an error
    raise RuntimeError("crossval_logistic_classifier: dataframe already has a 'pred' column")
  if normalise_predictors_axis is not None: 
    if   normalise_predictors_axis == 1:
      meanval = df[predictors].mean(axis=1).values[:,None]
    elif normalise_predictors_axis == 0:
      meanval = df[predictors].mean(axis=0).values[None,:]
    if normalise_divisive:
      df[predictors] = df[predictors] / meanval
    else:
      df[predictors] = df[predictors] - meanval
  df = df.dropna(subset=predictors) # drop rows where predictors contain a nan
  df['pred'] = np.nan # create blank column for prediction
  model = yvar+' ~ ' + ' + '.join(predictors) # create model string
  for row in range(df.shape[0]):   # leave one out: 
    # subsample the data so that equal number of each class are present:
    train_df = df.loc[ df[yvar]==0 ]   # 1. select all rows of the other class
    # 2. select the same number of rows of the class we are predicting
    if resample_all:
       train_df = pd.concat( [ df.loc[ df[yvar]==0 ].sample(train_df.shape[0], replace=True), 
                           df.loc[ df[yvar]==1 ].sample(train_df.shape[0], replace=True) ] )
    else:
      train_df = pd.concat( [ train_df,  df.loc[ df[yvar]==1 ].sample(train_df.shape[0], replace=False) ] )
    index_to_predict = df.index[row] # 3. which row are we predicting?
    try: # remove index of row-to-predict from tmp2  (if it is present - it might not be!)
      train_df = train_df.drop(index_to_predict)     
    except:
      pass
    try:
      m = smf.logit(model,train_df).fit(disp=0)
    except:
      m = smf.glm(model,train_df,family=sm.families.Binomial()).fit_regularized( alpha=0.1, L1_wt=0 )
    # predict the value of this row
    pred = m.predict(df.loc[[index_to_predict],:])
    # store the prediction 
    df.loc[index_to_predict,'pred'] = pred.values # should be just one value
  # compute error of prediction
  df['pred_err'] = abs(df['pred'] - df[yvar])
  # calculate roc from pred and yvar
  thresholds = np.linspace(0,1,100)  
  tpr = np.mean( df.pred[ df[yvar]==1 ].values[:,None] > thresholds[None,:] , axis=0)
  fpr = np.mean( df.pred[ df[yvar]==0 ].values[:,None] > thresholds[None,:] , axis=0)
  auc = -np.trapz(tpr,fpr)   # calculate AUC
  if do_plot:
    plt.plot(fpr,tpr)  # plot ROC curve
    plt.gca().set_aspect('equal', adjustable='box')   # ensure axes are square
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title( 'ROC: '+classify[0]+' vs '+classify[1]+"\n" +
              group+" ~ "+"+".join(predictors) )
    plt.plot([0,1],[0,1],'--') # plot diagonal
    # show AUC on plot
    plt.text(0.5,0.1,f'AUC={auc:.3f}',horizontalalignment='center',verticalalignment='center')
  acc = 1-np.mean((df.pred > 0.5) ^ (df[yvar])) # xor to get accuracy
  return df, auc, acc

def simple_ols(Y,X,C=None, var=False, cov=False):
  """
  Fit a linear model, and return the contrast of interest. Ultra fast, operates on matrices.
  returns contrasts of parameter estimates, and optionally their variance, covariance and t-statistic.
    Y = (measurements x repeats)
    X = (measurements x predictors)
    C = (contrasts x predictors)
    var = True/False, whether to return variances and t-statistics
    cov = True/False, whether to return covariances
    Based on TB 2004!
  """
  from numpy.linalg import pinv
  if C is None: # default contrasts = one per predictor
    C = np.eye(X.shape[1])
  Xi = pinv(X)
  b  = Xi @ Y
  cb = C @ b 
  if not var: 
    return cb   
  else: # calculate variance and t-statistic
    prevar = np.diag( C @ Xi @ Xi.T @ C.T ) 
    # multiply the i'th row of X by the i'th column of Xi
    tR = X.shape[0] - sum( ( X[i,:] @ Xi[:,i] for i in range(X.shape[0]) ) )
    # R = np.eye(X.shape[0])  -  X @ Xi  # old version takes up a lot of ram
    # tR = np.trace(R)
    res = Y - X @ b              # residuals
    sigsq = np.sum(res*res / tR,axis=0) # scalar - error variance
    varcb = prevar[:,None] * sigsq[None,:]       # variance of contrast
    tstat = cb / np.sqrt(varcb)[:,None]   # t statistic = contrast / sqrt(variance). t[ contrast, repeat ]
    if cov:
      covcb = ( C @ X.T @ X @ C.T ) * sigsq[None,None,:] 
      return cb, varcb, covcb, tstat
    else:
       return cb, varcb, tstat


def permutation_ols(
      Y, # Y = measurements x repeats,
      X, # X = predictors matrix, C = contrasts
      C, # C = contrasts matrix
      nperm = 1000, # number of permutations
      G=None, # grouping variable (optional), to permute within groups
      ):
  """
  Fit a linear model, and correct for multiple comparisons using permutations.
    Y = (measurements x repeats)
    X = (measurements x predictors)
    C = (contrasts x predictors)
    nperm = number of permutations
    G = grouping variable, to permute within groups
  returns 
    cb = contrasts of parameter estimates
    t  = t-statistic
    p  = p-value, 0-1, = chance that the t statistic is higher than expected 
       from the chance distribution. Values close to 0 or 1 are significant.
  """
  from scipy.linalg import pinv
  # compute unpermuted stats
  cb,_,_,t = simple_ols(Y,X,C,var=True)    # non-permuted statistic t[ contrast, repeat ]
  tmax = np.nan * np.zeros((nperm, C.shape[0])) # store max t statistic for each permutation and contrast
  for i in range(nperm):                   # for each permutation
    if G is None:
      Xp = np.random.permutation(X)        # shuffle rows of design matrix, keeping values in a row together
    else:
      Xp = np.zeros(X.shape)               # create empty matrix
      for g in np.unique(G):               # for each group
        Xp[G==g,:] = np.random.permutation(X[G==g,:])
    _,_,_,tp = simple_ols(Y,Xp,C,var=True) # get t stat for the shuffled data
    tmax[i, :] = tp.max(axis=1)            # store the max t statistic tmax [ permutation, contrast ]
  # calculate p values
  #            [ permutation, contrast, repeat ]
  p = np.mean( tmax[:,:,None] > t[None,:,:], axis=0 )  # proportion of permutations that had a higher t statistic
  return cb, t, p

if False: # unit test code
  from pylab import *
  N=1000  # number of datapoints
  T=100   # number of repeats
  D=3     # number of predictors
  tau = linspace(0,4*pi,T) # time
  X = randn(N,D)           # predictors matrix
  b = ones((D,T)) * array([sin(tau)]) # ground truth parameters for simulation
  Y = X@b + randn(N,T)                  # generate simulated measurements
  cb,t,p=permutation_ols(Y,X,eye(D))    # test


def align_bottom(X, inplace=False):
  """
  Align each column of the array X to the bottom by removing nans from the bottom end, 
  and moving them to the top.
  Expands to 2D if only 1D.
  """
  #if len(X.shape) == 1: # expand to 2D
  #  X = X[:,None]
  shp = X.shape
  if len(shp) != 2: # convert to 2d array, and remember size
    X = X.reshape(X.shape[0],-1)
  if not inplace:
    X = X.copy()
  for i in range(X.shape[1]):     # for each column
    last = np.where(~np.isnan(X[:,i]))[0][-1]  # find the last non-nan value
    nn = X.shape[0] - last - 1        # number of nans at end (after last)
    if nn>0:                        # the nn==0 case doesn't work because [:-0] is empty, not full!
      X[nn:,i] = X[0:-nn,i]         # move values down 
      X[0:nn,i] = np.nan            # set the rest to nan
  if len(shp)>2:                  # restore original shape
    X = X.reshape(shp)
  return X
