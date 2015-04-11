import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import math
import os
import random

from io import BytesIO

from scipy.signal import correlate
from scipy.signal import fftconvolve
import scipy.signal as ss
from scipy.ndimage.filters import gaussian_filter 



def thresh_sample(v):
    """
    Truncates a signal (dataframe) to where the signal is active.
    """
    m,s = v.mean(),v.std().apply(lambda x:.23*math.sqrt(x))
    tm,tM = m-s,m+s
    trunc_m = v[ v.apply(lambda x: x>tM,axis=1).sum(axis=1)!= 0]
    trunc_M = v[ v.apply(lambda x: x<tm,axis=1).sum(axis=1)!= 0]
    m_ind = min(trunc_m.index[0],trunc_M.index[0])
    M_ind = max(trunc_m.index[-1],trunc_M.index[-1])
    return v.ix[m_ind:M_ind,:]

def normalize(v):
    """
    Smooths, z-scales, then pads/truncates to a window of length 512.
    """
    v = v.apply(lambda x:gaussian_filter(x,sigma=4),axis=0)
    m,s = v.mean(),v.std()
    v = v.apply(lambda x:(x-m[x.name])/s[x.name],axis=0)
    if len(v) < 512:
        zeros = pd.DataFrame(np.zeros([512-len(v),3]),columns=['x','y','z'])
        v = v.append( zeros, ignore_index=True )
    else:
        v = v.reindex(range(512),fill_value=0)
    return v

def convolve(s1,s2):
    """ Convolves two 3d signals (dataframes) component-wise. """
    df = pd.DataFrame()
    for x in s1.columns:
        df[x] = fftconvolve(s1[x],s2[x])
    return df


def convolve_stream(stream,trainingD,window_size=50):
    """
    Returns a dictionary of signals, each of which is
    obtained by convolving the stream (window-by-window) with
    the training gestures (and taking the max of the resulting
    signal at each point.
    """
    D = dict()
    for gesture,v in trainingD.items():
        D[gesture] = []
        for k in range(len(stream)-512):
            if k % window_size == 0:
                window = stream[k:k+512]
                a = convolve(window,v).max().max()  # use *where* max occurs / aligned among coords?
                D[gesture].append(a)
            else:
                D[gesture].append(a)
    return D

## NOT USED
def norm_interp(v):
    """
    Linearly scales signals to a uniform window size of 100.
    """
    def toax(a):
        xd = a.index
        xd = xd/float(xd[-1])*100
        return np.interp(range(100),xd,a)
    df = pd.DataFrame([ toax(v[a]) for a in v.columns ] ).T
    df.columns = ['x','y','z']
    m = df.mean()
    return df.apply(lambda x:x-m[x.name],axis=0)

#---------------------------------------------------------------------
# Signal loading / creation helper functions
#---------------------------------------------------------------------

def load_gestures(L,preprocess=True):
    """
    Loads gestures into a dictionary from a list of file paths.
    """
    D = dict()
    for f in L:
        with open(f) as fh:
            s = ''.join([ x for x in fh if (x[0]!='#' and x[0]!='\n')])
            D[f] = pd.read_csv(BytesIO(s),sep=' ',names=['x','y','z','v'],usecols=['x','y','z'])
            if preprocess:
                D[f] = thresh_sample(D[f]).reset_index(drop=True)
            D[f] = normalize(D[f])
    return D

# add noise between gestures in the function below!!!!!

def create_streaming_gesture(D,length=10000):
    """
    Creates a stream of gestures from a dictionary of gestures of length D.
    Returns a dictionary {'data':signal_stream,'ans':(gesture type,index)}.
    """
    outDict = {'ans':[]}
    df = pd.DataFrame(columns=['x','y','z'])
    while len(df) < length:
        m = random.randint(0,1000)
        if m < 300:
            k = D.keys()[random.randint(0,len(D)-1)]
            z = D[k]
            outDict['ans'].append( (len(df),k) )
        else:
            z = pd.DataFrame(np.zeros([m,3]),columns=['x','y','z'])
        df = pd.concat([df,z],axis=0)
    outDict['data'] = df.iloc[:length,:].reset_index(drop=True)
    return outDict


#---------------------------------------------------------------------
# Plotting Helper Functions 
#---------------------------------------------------------------------

def plot_gestures(D):
    """Plots raw signal data from a dictionary of signals"""
    m = len(D)
    nrows = m/4
    fig, axes = plt.subplots(nrows=nrows, ncols=4,figsize=(17,4*nrows))
    if nrows == 1:
        for i,(k,v) in enumerate(sorted(D.items())):
            v.plot(title=k,ax=axes[i],xlim=(0,512),sharey=True)
    else:
        for i,(k,v) in enumerate(sorted(D.items())):
            v.plot(title=k,ax=axes[i/4,i%4],xlim=(0,512),sharey=True)


def plot_convolution(convDict,ans=None):
    """
    Plots convolutions (for a dictionary from convolv_stream) w/
    vertical lines marking where/what the true gestures were.
    """
    fig, axes = plt.subplots(nrows=2, ncols=2,figsize=(14,6))
    i=0
    for k,v in convDict.items():
        ax = axes[i/2,i%2]
        ax.plot(v)
        ax.set_title(k.replace('__2.txt',''))
        if ans:
            for (ind,name) in ans:
                if k[:-6] in name:
                    ax.vlines(ind,0,400,colors='red')
        i+=1

#---------------------------------------------------------------------
# functions for guessing the gesture type
#---------------------------------------------------------------------        

def guess(v,trainingD):
    D = {k:convolve(v,x).max().max() for k,x in trainingD.items() }
    if D['slap__2.txt'] > 350:
        return 'slap'
    elif D['hifive__2.txt'] > 325:
        return 'hi five!'
    elif D['fistbump__2.txt'] > 250:
        return 'fistbump, bro!'
    elif D['handshake__2.txt'] > 225:
        return 'handshake'
    else:
        s = max(D, key=D.get)
        return s.replace('__2.txt','')
    
def guess_the_gesture(testingD,trainingD):
    k = random.randint(1,10)
    gesture,v = testingD.items()[k]
    v.plot(title=gesture)
    ri = raw_input("hit <Enter> for guess + answer\n\n")
    print "guess is: %s\n" %guess(v,trainingD)
    print "input was: %s" % gesture
