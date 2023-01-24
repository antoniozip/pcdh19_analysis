#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 15:57:11 2021

@author: Antonio G. Zippo, PhD, Institute of Neuroscience, Consiglio Nazionale delle Ricerche
"""
import numpy as np
from importlib import reload
import matplotlib.pylab as plt
import spikeinterface.toolkit as st
import spikeinterface.extractors as se
import spikeinterface.widgets as sw
import spikeinterface.sorters as ss
import spiketoolkit.postprocessing as sp
import os
import matplotlib.pylab as plt
import elephant.spectral as es
import networkx as nx
from scipy import signal
from scipy import stats
from scipy.signal import butter, lfilter, filtfilt, iirfilter, iirnotch, hilbert
import h5py
from joblib import Parallel, delayed
import multiprocessing

def butter_lowpass(cutOff, fs, order=5):
    nyq = 0.5 * fs
    normalCutoff = cutOff / nyq
    b, a = butter(order, normalCutoff, btype='low', analog = False)
    return b, a

def butter_lowpass_filter(data, cutOff, fs, order = 5):
    b, a = butter_lowpass(cutOff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def butter_highpass(cutoff, fs, order = 5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog = False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def line_filter(data, fs):
    b, a = iirnotch(50, 16, fs)
    return lfilter(b,a, data)

def synch_lfp(lfp_x, lfp_y):
    al1 = np.angle(hilbert(lfp_x),deg=False)
    al2 = np.angle(hilbert(lfp_y),deg=False)
    return np.mean(np.sin(np.abs(al1-al2)/2))

nchannels = 4096
fs = 7000
f = h5py.File('file.brw', 'r')
data = f['3BData']['Raw']
length = data.shape[0]
dim1 = length/4096

matrix = np.reshape(data, (int(dim1), nchannels)).T

num_cores = multiprocessing.cpu_count()
lfp_ps = np.zeros((len(matrix), len(matrix)))
for xx in range(len(matrix)):
    lfp_ps[xx,:] = Parallel(n_jobs=num_cores-1)(delayed(synch_lfp)(matrix[xx], i) for i in matrix)
    xx = xx + 1


