import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import interactive
interactive(True)
import numpy as np
import math
import os
from io import BytesIO
from scipy.signal import correlate
from scipy.signal import fftconvolve
import scipy.signal as ss
from scipy.ndimage.filters import gaussian_filter 
import random


def thresh_sample(v):
    m,s = v.mean(),v.std().apply(lambda x:.23*math.sqrt(x))
    tm,tM = m-s,m+s
    trunc_m = v[ v.apply(lambda x: x>tM,axis=1).sum(axis=1)!= 0]
    trunc_M = v[ v.apply(lambda x: x<tm,axis=1).sum(axis=1)!= 0]
    m_ind = min(trunc_m.index[0],trunc_M.index[0])
    M_ind = max(trunc_m.index[-1],trunc_M.index[-1])
    return v.ix[m_ind:M_ind,:]

def normalize(v):
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
    df = pd.DataFrame()
    for x in s1.columns:
        #df[x] = np.fft.ifft(np.fft.fft(s1[x])*np.fft.fft(s2[x])).real
        #df[x] = correlate(s1[x],s2[x])
        df[x] = fftconvolve(s1[x],s2[x])
    return df


# load / preprocess signal

def load_gestures(L,preprocess=True):
    D = dict()
    for f in L:
        with open(f) as fh:
            s = ''.join([ x for x in fh if (x[0]!='#' and x[0]!='\n')])
            D[f] = pd.read_csv(BytesIO(s),sep=' ',names=['x','y','z','v'],usecols=['x','y','z'])
            if preprocess:
                D[f] = thresh_sample(D[f]).reset_index(drop=True)
            D[f] = normalize(D[f])
    return D

def create_streaming_gesture(D,length=10000):
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

def plot_gestures(D):
    m = len(D)
    nrows = m/4 #(m+1)/4+1
    fig, axes = plt.subplots(nrows=nrows, ncols=4,figsize=(17,4*nrows))
    if nrows == 1:
        for i,(k,v) in enumerate(sorted(D.items())):
            v.plot(title=k,ax=axes[i],xlim=(0,512),sharey=True)
    else:
        for i,(k,v) in enumerate(sorted(D.items())):
            v.plot(title=k,ax=axes[i/4,i%4],xlim=(0,512),sharey=True)
        
def convolve_stream(stream,trainingD):
    D = dict()
    for gesture,v in trainingD.items():
        D[gesture] = []
        for k in range(len(stream)-512):
            if k % 50 == 0:
                window = stream[k:k+512]
                a = convolve(window,v).max().max()
                D[gesture].append(a)
            else:
                D[gesture].append(a)
    return D

def plot_convolution(convDict,ans=None):
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
