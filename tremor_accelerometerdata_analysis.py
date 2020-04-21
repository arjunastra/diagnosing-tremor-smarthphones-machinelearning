'''
File: tremor_accelerometerdata_analysis.py
Project: 'Machine Learning for Classifying Tremor Using Smartphone Accelerometers and Clinical Features', Balachandar et. al 2020

@author: Arjun Balachandar MD

Please refer to README file for more information.

- This program takes the raw accelerometer data for of tremor in patients who had recordings conducted (both time series and frequency-power spectrum data) and analyzes them to produce metrics that can be compared across tremor groups (i.e. PD, ET, DT and controls)
- This program uses a metadata file containing the list of all patients studied (including non-identifying info e.g. initials, clinician-assigned diagnosis, age, etc) and stores this information
- Using the stored basic ID info, the program then opens corresponding tremor recording files (4 per patient, each corresponding to one of the 4 recording positions) and conducts both time-series and frequency-power spectrum analysis
- These analyses are conducted using the functions 'timeseries_analysis' and 'freq_analysis respectively', which are called in the main body of code and take as input a given tremor recording .csv file
- The analysis metrics obtained from processing all the recording data are then stored in a single .csv file containing the list of all patients (i.e. initials) and all of the analysis metrics obtained for each patient in corresponding columns.
- Statistical analysis can then be done using this final output file to compare the values of metrics across each tremor group as a whole (conducted in SPSS for this project)
- The final output file containing all analysis features for each patient can also be used as input features for machine learning algorithms (see README file for mor details)
'''

#Import all required packages
from __future__ import division, print_function
import os
import subprocess
import math
import statistics
import numpy as np
from numpy.random import randn
import numpy.fft as fft
from scipy import signal
from scipy.signal import butter, lfilter, filtfilt
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import io
import pickle
import pprint

#PATH refers to the folder path where both this file, the required input .csv metadata file (the metadata file containing the list of all patient data (patient initials/labels, age, sex etc)) and the tremor accelerometer data files for each patient (containing raw unprocessed accelerometer data) are stored
#Note: user must set PATH according to where this file and required .csv files are stored on their computer
PATH = "/Users/rambalachandar/Desktop/University of Toronto/Fasano Lab/Actual data files"
os.chdir(PATH)

#The four positions that tremor is recorded in each patient (see methods section of the paper)
tremor_types = ["bat","kin","out","rest"]

#Open metadata file containing all basic non-identifying patient information (i.e. initials, diagnosis, age, etc.)
metadata = open('data_metadata_updated.csv')
heading = metadata.readline().strip("\n").split(",") #first row of the metadata file, containing headings for each column (e.g. name, diagnosis, etc.)

#All patient information from the metadata file is stored in arrays, defined below
dates_all = []
dates = []
names = []
ages = []
MRNs = []
hands = []
file_names = []
diagnoses = []
basic_diagnoses = []
comments = []

#Average and standard-dev functions
def ave(lst):
    if len(lst) == 0:
        return [0.0]
    ans = [0.0]*len(lst[0])
    for i in range(len(lst)):
        for j in range(len(lst[i])):
            ans[j] += float(lst[i][j])
             
    for i in range(len(ans)):
        ans[i] = ans[i]/len(lst)        
    return ans

def average(lst):
    if len(lst) == 0:
        return 0.0
    ans = 0.0
    for i in range(len(lst)):
        ans += lst[i]
    ans = ans/len(lst)
    return ans

def std(lst,ave):
    if len(lst) == 0:
        return [0.0]
    if len(lst) <= 1:
        return [0.0]*len(lst[0])
    ans = [0.0]*len(lst[0])
    for i in range(len(lst)):
        for j in range(len(lst[i])):
            ans[j] += math.pow(float(lst[i][j])-float(ave[j]), 2)
             
    for i in range(len(ans)):
        ans[i] = math.sqrt(ans[i]/(len(lst)-1))
         
    return ans
    
def stddev(lst,ave):
    if len(lst) <= 1:
        return 0.0
    ans = 0.0
    for i in range(len(lst)):
        ans += math.pow(float(lst[i])-float(ave), 2)
    ans = math.sqrt(ans/(len(lst)-1))
    return ans


#Check for number of harmonic frequencies (i.e integer multiples of the characteristic/main-peak frequency) using the following function
def num_harmonics(peaks):
    #info in each peak in peaks: [p1U,p2U,pmaxU,powmaxU,pIntegU]
    n_mult = 15 #number of harmonics to test
    num_harm = 0
    harmPresent = 0 #=1 if harmonics present
    harm_powers = []
    h_lst = []
    for i in range(len(peaks)):
        if i==0:
            main_peak = peaks[i]
            main_peak_fre = main_peak[2]
        else:
            peak = peaks[i]
            peak_fre = peak[2]
            peak_pow = peak[3]
            for h in range(n_mult):
                if h>1 and h_lst.count(h)<1 and abs(h*main_peak_fre - peak_fre) <= 0.25*h:
                    num_harm += 1
                    h_lst.append(h)
                    harm_powers.append(peak_pow)
    if num_harm > 0:
        harmPresent = 1
    return [num_harm,h_lst,harm_powers,harmPresent]
    
#integrate function from f_start to f_end
def integrate(flist,xlist,f_start,f_end): #flist = x-var, xlist = y-var
    ans = 0.0
    for i in range(len(flist)):
        if i>0:
            if flist[i-1] >= f_start and flist[i] <= f_end:
                ans += 0.5*(flist[i] - flist[i-1])*(xlist[i] + xlist[i-1])
    return ans

#find positive zero crossings of a curve (discrete, non-continuous)
def find_zerocross(t,sig_ff):
    cross = []
    
    cur_proc = 0.0#slope of pre-processed signal
    prev_proc = 0.0
    
    for i in range(len(t)-1):
        if i>0:
            
            #using processed signal:
            cur_proc = sig_ff[i]
            prev_proc = sig_ff[i-1]
            
            if cur_proc > 0 and prev_proc <= 0:
                cross.append(t[i])
    return cross

#using positive zero-crossings, find inst. freqs
def find_freqs(peaks):
    freqs = []
    for i in range(len(peaks)-1):
        if (peaks[i+1] - peaks[i]) > 0:
            freqs.append(1.0/(peaks[i+1] - peaks[i]))
    return freqs
    
#using slope, find time and power of peaks in time-series
def find_peakPowers(t,sig_ff):
    powers = []
    peaks = [] #time of Peaks
    
    cur_slope = 0.0
    prev_slope = sig_ff[1] - sig_ff[0]
    
    for i in range(len(t)-1):
        cur_slope = sig_ff[i+1] - sig_ff[i]
        if cur_slope <= 0 and prev_slope >=0:
            peaks.append(t[i])
            powers.append(sig_ff[i])
        prev_slope = cur_slope
    return [peaks,powers]


#The below functions are for butterworth signal processing
def butter_bandpass(lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def butter_highpass(cutoff, fs, order):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def butter_lowpass(cutoff, fs, order):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y


#Analyze time-series to calculate various metrics, such as the TSI (Tremor Stability Index, see paper) using below function
def timeseries_analysis(file, new_path, display_tremor,setFreqRange):
    #Function takes the tremor file name and path as input, and then opens the file in the next lines of code
    os.chdir(new_path)
    f = open(file)
    
    #Ignores the first 10 lines of code, which do not contain any data
    for i in range(10):
        line = f.readline()
    
    sample_rate = 50
    
    line = f.readline()
    line = line.strip()
    line_list = line.strip().split(",")
    #Obtains the intial time, x, y and z of the time-series
    t_pre = line_list[0]
    x_pre = line_list[1]
    y_pre = line_list[2]
    z_pre = line_list[3]
    
    ax_ind = 2 #index of direction to use for slope comparison; 0=x, 1=y, 2=z
    
    #Below are arrays which contain the time and position (i.e. values along each axes x,y,z and composite axis u (calculated below))
    t = []
    x = []
    y = []
    z = []
    u = []
    all_axes = []
    
    zero = [] #list of when peak (slope_cur < 0 & slope-prev > 0) occurs for given direction (axe_ind)
        
    while line != "":
        line = f.readline()
        line = line.strip()
        line_list = line.strip().split(",")
        line_len = len(line_list)
        if line == "" or line_len < 4:
            break
        
        #Obtains the current time and position and stores them in the corresponding arrays
        t_cur = float(line_list[0])
        x_cur = float(line_list[1])
        y_cur = float(line_list[2])
        z_cur = float(line_list[3])
        u_cur = math.sqrt(x_cur**2 + y_cur**2 + z_cur**2) #u = sqrt(x^2 + y^2 + z^2), and hence is a composite of x, y, z and u
        t.append(t_cur)
        x.append(x_cur)
        y.append(y_cur)
        z.append(z_cur)
        u.append(u_cur)
        #all_axes.append([x_cur,y_cur,z_cur,u_cur])
    
    #signal processing of the raw time-series data using butterworth filter
    b, a = signal.butter(3, 0.1, btype='high',analog=False)
    x_tcor = signal.filtfilt(b, a, x) #trend corrected x-series
    y_tcor = signal.filtfilt(b, a, y)
    z_tcor = signal.filtfilt(b, a, z)
    u_tcor = signal.filtfilt(b, a, u)
    
    for i in range(len(x_tcor)):
        all_axes.append([x_tcor[i],y_tcor[i],z_tcor[i],u_tcor[i]])
    
    #do PCA analysis on all_axes to find dominant axis
    pca_on = True #if True, then do PCA and determine which axis to use
    all_axesnp = np.array(all_axes)
    if pca_on == True:
        pca = PCA(n_components=4)
        pca.fit(all_axesnp)
        pca_output = pca.explained_variance_ratio_
        bestAxisInd = np.argmax(pca_output) #which axis was strongest component
        all_axes_trans = pca.transform(all_axesnp)
        if bestAxisInd == 0:
            sig = x
        elif bestAxisInd == 1:
            sig = y
        elif bestAxisInd == 2:
            sig = z
        else:
            sig = u
        
        #convert to list
        sig = []
        for i in range(len(all_axes_trans)):
            sig.append(all_axes_trans[i][0])
    else:
        sig = z
    
    sig_ff1 = sig
    #butterworth filters
    #sig = z #use z-axis time-series for analysis
    #b, a = signal.butter(3, 0.1, btype='high',analog=False)
    #sig_ff1 = signal.filtfilt(b, a, sig)
    
    #FFT to find main freq peaks
    spectrum = fft.fft(sig_ff1)
    freq_fft = abs(fft.fftfreq(len(spectrum)))*sample_rate
    threshold = 0.5 * max(abs(spectrum))
    mask = abs(spectrum) > threshold
    peaks = freq_fft[mask]
    
    spectrum2 = spectrum #duplicate for use later
    freq_fft2 = freq_fft

    c = 0
    freq_max = 0.0
    
    if setFreqRange == True: #i.e. if OT file
        freq_high = 20 #range of freqs allowed
        freq_low = 2
    else: #i.e. if not OT
        freq_high = 9 #range of freqs allowed
        freq_low = 2
        
    while True:
        idx = np.argmax(abs(spectrum2))
        freq_max = (freq_fft2[idx]).item()
        if freq_max <= freq_high and freq_max >= freq_low:
            break
        else:
            spectrum2 = numpy.delete(spectrum2,idx)
            freq_fft2 = numpy.delete(freq_fft2,idx)
            c = c + 1
        
    
    #idx = np.argmax(abs(spectrum))
    #freq_max = freq_fft[idx]
    #print(freq_max)
    
    #Plot the FFT data below
    #plt.figure()
    #plt.plot(freq_fft, color='blue', label='Peaks')
    #print(freq_fft.max())
    #print(freq_fft.min())
    #print(len(freq_fft))
    
    #zero = find_zerocross(t,x,y,z,sig_ff1)
    
    # m_cur = [0.0,0.0,0.0] #slope of x,y,z; i.e. has 3 elements, one for each slope
    # m_prev = [0.0,0.0,0.0]
    # 
    # m_cur_proc = 0.0#slope of pre-processed signal
    # m_prev_proc = 0.0
    # 
    # for i in range(len(t)-1):
    #     if i>0:
    #         m_cur[0] = x[i+1] - x[i]
    #         m_cur[1] = y[i+1] - y[i]
    #         m_cur[2] = z[i+1] - z[i]
    #         
    #         m_prev[0] = x[i] - x[i-1]
    #         m_prev[1] = y[i] - y[i-1]
    #         m_prev[2] = z[i] - z[i-1]
    #         
    #         #using processed signal:
    #         m_cur_proc = sig_ff[i+1] - sig_ff[i-1]
    #         m_prev_proc = sig_ff[i] - sig_ff[i-1]
    #         
    #         #ax_ind determines which axis to use for calculating f and df data
    #         #if m_cur[ax_ind] < 0 and m_prev[ax_ind] > 0:
    #         if m_cur_proc < 0 and m_prev_proc > 0:
    #             peaks.append(t[i])
    
    #determine freq and delta_f using peaks-time list
    # freqs = []
    # delta_fs = []
    # for i in range(len(peaks)-1):
    #     if (peaks[i+1] - peaks[i]) > 0:
    #         freqs.append(1.0/(peaks[i+1] - peaks[i]))
    
    #determine freq and delta_f using peaks-time list
    #freqs = find_freqs(zero)
    
    #freqs_filtered = [] #only take freqs within +/- d_fc of f_median
    #f_mean = sum(freqs)/len(freqs)
    #f_median = statistics.median(freqs) #median frequency
    #f_c = f_median #choose which of mean or median to use
    
    f_c = freq_max #use FFT to find peak freq
    d_fc = 2 #fc +/- d_fc
    
    #high-pass and low-pass filters around f_c +/- 2
    fs = 50.0 #sample rate
    highcut = f_c + d_fc
    lowcut = f_c - d_fc
    
    sig_ff = butter_bandpass_filter(sig_ff1, lowcut, highcut, fs, order=3)
    #sig_ff_high = butter_highpass_filter(sig_ff1, highcut, fs, order=3)
    #sig_ff = butter_lowpass_filter(sig_ff_high, lowcut, fs, order=3)
    
    zero2 = find_zerocross(t,sig_ff) #find_zerocross(t,x,y,z,sig_ff)
    
    freqs2 = [] #new freqs after applying bandpass filter
    freqs_filtered = [] #just in case, again make sure all freqs within fc+/-d_fc
    delta_fs = []
    freqs2 = find_freqs(zero2)
    
    #peak timings and power
    peakInfo = find_peakPowers(t,sig_ff)
    peak_times = peakInfo[0]
    peak_power = peakInfo[1]
    #f_median = statistics.median(freqs2) #update median freq
    #f_c = f_median
    
    # for i in range(len(freqs_filtered)-1):
    #     delta_f = freqs_filtered[i+1] - freqs_filtered[i]
    #     delta_fs.append(delta_f)
    
    filter_freqs = False #whether to filter freqs around f_c +/ d_fc or not

    #Below, the instantaneous change in frequency between each oscillation is calculated and stored in arrays
    for i in range(len(freqs2)-1):
        #only take freqs within +/- 2 of f_median:
        if filter_freqs == True:
            if freqs2[i] < (f_c + d_fc) and freqs2[i] > (f_c - d_fc) and freqs2[i+1] < (f_c + d_fc) and freqs2[i+1] > (f_c - d_fc):
                freqs_filtered.append(freqs2[i])
                delta_f = freqs2[i] - freqs2[i+1]
                delta_fs.append(delta_f)
        else:
            freqs_filtered.append(freqs2[i])
            delta_f = freqs2[i] - freqs2[i+1]
            delta_fs.append(delta_f)
    
    # freqs_filtered = freqs2
    # for i in range(len(freqs2)-1):
    #     delta_f = freqs2[i+1] - freqs2[i]
    #     delta_fs.append(delta_f)
            
    #Calculate the TSI index, defined as the interquartile range of the set of instantaneous changes in frequency
    TSI = 0
    if len(delta_fs) > 0:
        fr = numpy.array(delta_fs)
        TSI = (numpy.percentile(fr, 75) - numpy.percentile(fr, 25)).item()
    
    TSI_peakPow = 0
    fs = numpy.array(peak_power)
    TSI_peakPow = (numpy.percentile(fs, 75) - numpy.percentile(fs, 25)).item()
    
    peakPowAve = average(peak_power)
    peakPowStd = stddev(peak_power,peakPowAve)
    
    maxPow = max(peak_power)
    minPow = min(peak_power)
    
    #display plot
    # if display_tremor == True:
    #     plt.figure()
    #     plt.plot(sig, color='silver', label='Original')
    #     plt.plot(sig_ff, color='#3465a4', label='filtfilt')
    #     
    #     zeros_y = [0.0]*(len(zero2))
    #     #plt.scatter(zero2, zeros_y,color='green', label = "Zeros')
    #     
    #     plt.legend(loc="best")
    
    #if display_plot == True:
        # print(len(zero2))
        # print(len(freqs_filtered))
        # print(len(delta_fs))
        # print(TSI)
        # print(f_c)
        #print(delta_fs)
    
    #return [freqs_filtered,delta_fs,f_mean,f_median,TSI,sig_ff,zero2,t]
    return [freqs_filtered,delta_fs,f_c,TSI,sig_ff,zero2,t,TSI_peakPow,peak_power,peakPowAve,peakPowStd,maxPow,minPow]

#######################
#Function to conduct frequency-power spectrum analysis on each file
def freq_analysis(file, new_path):
    os.chdir(new_path)
    f = open(file)
    line = f.readline()
    
    peak_slope = 0.5
    min_pow = 0.02
    f_low = 2.0 #filter out  all freqs below f_low
    thres = 4.0 #if peak height >= thres * ave, then considered a true peak --> usually 4.0
    
    peakX = False
    peakY = False
    peakZ = False
    peakU = False #Composite peak
    
    fr_ind = [0,0,0,0] #indices of where main peak occurs
    
    freqs = [0.0,0.0,0.0,0.0] #last element is 'u'
    maxpowers = [0.0,0.0,0.0,0.0]
    aves = [0.0,0.0,0.0,0.0] #average power, used as a marker for baseline
    harms = [0.0,0.0,0.0,0.0] #harmonics
    harm_pow = [0.0,0.0,0.0,0.0] #harmonics power
    
    area_0_fend = [0.0,0.0,0.0,0.0] #total area under the curve
    area_f1_fend = [0.0,0.0,0.0,0.0] #area under curve from main peak to end
    
    xmax = -1.0
    ymax = -1.0
    zmax = -1.0
    umax = -1.0
    
    #num of harmonics
    harmX = 0
    harmY = 0
    harmZ = 0
    harmU = 0
    
    flist = []
    xlist = []
    ylist = []
    zlist = []
    ulist = [] #composite value: u = sqrt(x^2 + y^2 + z^2)
    
    peaksX = [] #peaks, but original arrays not yet checked to see if pow > 10*baseline
    peaksY = []
    peaksZ = []
    peaksU = []
    
    peaksX_ref = [] #peaks with pow > 10*baseline
    peaksY_ref = []
    peaksZ_ref = []
    peaksU_ref = []
    
    k = 0
    
    while line!="":
        line = f.readline()
        line = line.strip().strip(",")
        
    
    for i in range(4):
        line = f.readline()
    
    line_len = 4
    num = 0 #number of frequencies analyzed in whole file
    pIntegX = 0.0
    pIntegY = 0.0
    pIntegZ = 0.0
    pIntegU = 0.0
    pmaxX = 0.0
    pmaxY = 0.0
    pmaxZ = 0.0
    pmaxU = 0.0
    powmaxX = 0.0
    powmaxY = 0.0
    powmaxZ = 0.0
    powmaxU = 0.0
    
    while line_len>=4:
        line = f.readline()
        line = line.strip()
        line_list = line.strip().split(",")
        line_len = len(line_list)
        if line_len <4:
            break
        
        fr = float(line_list[0])
        x = float(line_list[1])
        y = float(line_list[2])
        z = float(line_list[3])
        
        u = math.sqrt(x**2 + y**2 + z**2)
        
        xlist.append(x)
        ylist.append(y)
        zlist.append(z)
        ulist.append(u)
        flist.append(fr)
        
        if num>0:
            area_0_fend[0] = area_0_fend[0] + 0.5*(flist[num] - flist[num-1])*(xlist[num] + xlist[num-1])
            area_0_fend[1] = area_0_fend[1] + 0.5*(flist[num] - flist[num-1])*(ylist[num] + ylist[num-1])
            area_0_fend[2] = area_0_fend[2] + 0.5*(flist[num] - flist[num-1])*(zlist[num] + zlist[num-1])
            area_0_fend[3] = area_0_fend[3] + 0.5*(flist[num] - flist[num-1])*(ulist[num] + ulist[num-1])
        
        #Determine peaks, and area under each peak
        if num > 0 and flist[num] > f_low: #Filter all frequencies below f_low
            dp_dfX = (xlist[num] - xlist[num-1])/(flist[num] - flist[num-1])
            if dp_dfX > peak_slope and peakX==False:
                p1X = flist[num]
                peakX = True
                powmaxX = 0.0
                pIntegX = 0.0
                pmaxX = flist[num]
            if peakX==True:
                pIntegX += 0.5*(flist[num] - flist[num-1])*(xlist[num] + xlist[num-1])
                if xlist[num]>powmaxX:
                    powmaxX = xlist[num]
                    pmaxX = flist[num]
                if dp_dfX<-1*peak_slope:
                    p2X = flist[num]
                    peakX = False
                    #if powmaxX >= min_pow:
                    peaksX.append([p1X,p2X,pmaxX,powmaxX,pIntegX,(p2X-p1X)])
                    
            dp_dfY = (ylist[num] - ylist[num-1])/(flist[num] - flist[num-1])
            if dp_dfY > peak_slope and peakY==False:
                p1Y = flist[num]
                peakY = True
                powmaxY = 0.0
                pIntegY = 0.0
                pmaxY = flist[num]
            if peakY==True:
                pIntegY += 0.5*(flist[num] - flist[num-1])*(ylist[num] + ylist[num-1])
                if ylist[num] > powmaxY:
                    powmaxY = ylist[num]
                    pmaxY = flist[num]
                if dp_dfY<-1*peak_slope:
                    p2Y = flist[num]
                    peakY = False
                    #if powmaxY >= min_pow:
                    peaksY.append([p1Y,p2Y,pmaxY,powmaxY,pIntegY,(p2Y-p1Y)])
                    
            dp_dfZ = (zlist[num] - zlist[num-1])/(flist[num] - flist[num-1])
            if dp_dfZ > peak_slope and peakZ==False:
                p1Z = flist[num]
                peakZ = True
                powmaxZ = 0.0
                pIntegZ = 0.0
                pmaxZ = flist[num]
            if peakZ==True:
                pIntegZ += 0.5*(flist[num] - flist[num-1])*(zlist[num] + zlist[num-1])
                if zlist[num] > powmaxZ:
                    powmaxZ = zlist[num]
                    pmaxZ = flist[num]
                if dp_dfZ<-1*peak_slope:
                    p2Z = flist[num]
                    peakZ = False
                    #if powmaxZ >= min_pow:
                    peaksZ.append([p1Z,p2Z,pmaxZ,powmaxZ,pIntegZ,(p2Z-p1Z)])
            
            dp_dfU = (ulist[num] - ulist[num-1])/(flist[num] - flist[num-1])
            if dp_dfU > peak_slope and peakU==False:
                p1U = flist[num]
                peakU = True
                powmaxU = 0.0
                pIntegU = 0.0
                pmaxU = flist[num]
            if peakU==True:
                pIntegU += 0.5*(flist[num] - flist[num-1])*(ulist[num] + ulist[num-1])
                if ulist[num]>powmaxU:
                    powmaxU = ulist[num]
                    pmaxU = flist[num]
                if dp_dfU<-1*peak_slope:
                    p2U = flist[num]
                    peakU = False
                    #if powmaxX >= min_pow:
                    peaksU.append([p1U,p2U,pmaxU,powmaxU,pIntegU,(p2U-p1U)])
        
            
        if num > 0 and flist[num] > f_low:
            if x>xmax:# and peakX==True:
                xmax = x
                maxpowers[0] = xmax
                freqs[0] = fr
                fr_ind[0] = num
            if y>ymax:# and peakY==True:
                ymax = y
                maxpowers[1] = ymax
                freqs[1] = fr
                fr_ind[1] = num
            if z>zmax:# and peakZ==True:
                zmax = z
                maxpowers[2] = zmax
                freqs[2] = fr
                fr_ind[2] = num
            if u>umax:# and peakU==True:
                umax = u
                maxpowers[3] = umax
                freqs[3] = fr
                fr_ind[3] = num
        
            aves[0] = aves[0] + xlist[num]
            aves[1] = aves[1] + ylist[num]
            aves[2] = aves[2] + zlist[num]
            aves[3] = aves[3] + ulist[num]
        
        num += 1
    
    for i in range(len(aves)):
        aves[i] = aves[i]/(num+0.0)
    
    thres_num = 0.0 #0.05
    #Only take peaks > thres*baseline
    for i in range(len(peaksX)):
        if peaksX[i][3] >= thres*aves[0] and peaksX[i][3] >= thres_num:
            a = peaksX[i]
            a.append(aves[0]) #add average to the end of each peak
            peaksX_ref.append(a)
            #peaksX_ref.append(peaksX[i])
    for i in range(len(peaksY)):
        if peaksY[i][3] >= thres*aves[1] and peaksY[i][3] >= thres_num:
            peaksY_ref.append(peaksY[i])
    for i in range(len(peaksZ)):
        if peaksZ[i][3] >= thres*aves[2] and peaksZ[i][3] >= thres_num:
            peaksZ_ref.append(peaksZ[i])
    for i in range(len(peaksU)):
        if peaksU[i][3] >= thres*aves[3] and peaksU[i][3] >= thres_num:
            peaksU_ref.append(peaksU[i])
            
    # if xmax < thres*aves[0]:
    #     freqs[0] = 0.0
    #     maxpowers[0] = 0.0
    # if ymax < thres*aves[1]:
    #     freqs[1] = 0.0
    #     maxpowers[1] = 0.0
    # if zmax < thres*aves[2]:
    #     freqs[2] = 0.0
    #     maxpowers[2] = 0.0
    # if umax < thres*aves[3]:
    #     freqs[3] = 0.0
    #     maxpowers[3] = 0.0
    
    
    #order peaks from least to greatest (greatest to least?)
    peaksX_ref.sort(key=lambda x:x[3], reverse=True)
    peaksY_ref.sort(key=lambda x:x[3], reverse=True)
    peaksZ_ref.sort(key=lambda x:x[3], reverse=True)
    peaksU_ref.sort(key=lambda x:x[3], reverse=True)
    
    RPC = [0,0,0,0] #Relative Power Contribution to the first harmonic (RPC)
    #RPC is calculated from the quotient between the power spectral density of harmonics within the frequency range of f1 (threshold) and 25 Hz and the total normalized power spectral density for a frequency range of 0 to 25 Hz.
    
    #Take end point of first/main peak, used for calculating area_f1_25:
    f_extra = 1 #if no peak seen, take end of main peak to be f_max + f_extra
    
    if len(peaksX_ref) > 0:
        fmain_end_X = peaksX_ref[0][1]
    else:
        fmain_end_X = freqs[0] + f_extra #if no peaks technically seen, add 1 to f_max to get pseudo-end of main peak
    area_f1_fend[0] = integrate(flist,xlist,fmain_end_X,26) #arbitrarily choose f_end=26 since its after end of file f=25
    if area_0_fend[0] > 0:
        RPC[0] = area_f1_fend[0]/area_0_fend[0]
        
    if len(peaksY_ref) > 0:
        fmain_end_Y = peaksY_ref[0][1]
    else:
        fmain_end_Y = freqs[1] + f_extra
    area_f1_fend[1] = integrate(flist,ylist,fmain_end_Y,26)
    if area_0_fend[1] > 0:
        RPC[1] = area_f1_fend[1]/area_0_fend[1]
        
    if len(peaksZ_ref) > 0:
        fmain_end_Z = peaksZ_ref[0][1]
    else:
        fmain_end_Z = freqs[2] + f_extra
    area_f1_fend[2] = integrate(flist,zlist,fmain_end_Z,26)
    if area_0_fend[2] > 0:
        RPC[2] = area_f1_fend[2]/area_0_fend[2]
            
    if len(peaksU_ref) > 0:
        fmain_end_U = peaksU_ref[0][1]
    else:
        fmain_end_U = freqs[3] + f_extra
    area_f1_fend[3] = integrate(flist,ulist,fmain_end_U,26)
    if area_0_fend[3] > 0:
        RPC[3] = area_f1_fend[3]/area_0_fend[3]
    
    
    harm_info_X = num_harmonics(peaksX_ref)
    harms[0] = harm_info_X[3] #index 0 is total # of harmonics in file, index 3 says if harmonic present or not (1 or 0)
    #harm_index[0] = harm_info_X[1] #index of harmonic (see num_harmonics function)
    if len(harm_info_X[2]) > 0:
        #print(harm_info_X)
        harm_pow[0] = 0.0
        for i in range(len(harm_info_X[2])): #take AVERAGE of all harmonic peaks
            harm_pow[0] = harm_pow[0] + harm_info_X[2][i]
        harm_pow[0] = harm_pow[0]/len(harm_info_X[2])
        #harm_pow[0] = harm_info_X[2][0] #power of 1st harmonic, hence [0]
    else:
        harm_pow[0] = 0.0 #harm_info_X[2]
    
    harm_info_Y = num_harmonics(peaksY_ref)
    harms[1] = harm_info_Y[3]
    #harm_index[1] = harm_info_Y[1]
    #harm_pow[1] = harm_info_Y[2]
    if len(harm_info_Y[2]) > 0:
        harm_pow[1] = 0.0
        for i in range(len(harm_info_Y[2])): #take AVERAGE of all harmonic peaks
            harm_pow[1] = harm_pow[1] + harm_info_Y[2][i]
        harm_pow[1] = harm_pow[1]/len(harm_info_Y[2])
        #harm_pow[1] = harm_info_Y[2][0]
    else:
        harm_pow[1] = 0.0
    
    harm_info_Z = num_harmonics(peaksZ_ref)
    harms[2] = harm_info_Z[3]
    #harm_index[2] = harm_info_Z[1]
    #harm_pow[2] = harm_info_Z[2]
    if len(harm_info_Z[2]) > 0:
        #harm_pow[2] = harm_info_Z[2][0]
        harm_pow[2] = 0.0
        for i in range(len(harm_info_Z[2])): #take AVERAGE of all harmonic peaks
            harm_pow[2] = harm_pow[2] + harm_info_Z[2][i]
        harm_pow[2] = harm_pow[2]/len(harm_info_Z[2])
    else:
        harm_pow[2] = 0.0
    
    harm_info_U = num_harmonics(peaksU_ref)
    harms[3] = harm_info_U[3]
    #harm_index[3] = harm_info_U[1]
    harm_pow[3] = harm_info_U[2]
    if len(harm_info_U[2]) > 0:
        #harm_pow[3] = harm_info_U[2][0]
        harm_pow[3] = 0.0
        for i in range(len(harm_info_U[2])): #take AVERAGE of all harmonic peaks
            harm_pow[3] = harm_pow[3] + harm_info_U[2][i]
        harm_pow[3] = harm_pow[3]/len(harm_info_U[2])
    else:
        harm_pow[3] = 0.0
    #print(harm_pow)
    
    return [freqs,peaksX_ref,peaksY_ref,peaksZ_ref,peaksU_ref,aves,harms,maxpowers,harm_pow,RPC,area_0_fend]
    

'''
###########   Main Program: begins parsing files in metadata file #########
'''
for ln in metadata:
    line = ln.strip("\n").split(",")
    
    date = line[0]
    if date.strip()=="":
        dates_all.append(dates[-1].lower())
    else:
        dates.append(line[0].lower())
        dates_all.append(line[0].lower())
    names.append(line[1])
    ages.append(line[2])
    MRNs.append(line[3])
    hands.append(line[4])
    file_names.append(line[5])
    basic_diagnoses.append(line[6])
    diagnoses.append(line[7])
    comments.append(line[8])

#Time-series analysis lists
instAveFreq_list = {}
instAvePow_list = {}
instStdPow_list = {}
TSI_list = {}
TSI_ppow_list = {}
instTSIppow_AvePow_ratio_list = {}
instPow_Std_Ave_ratio_list = {}
    
#Frequency analysis lists
freq_list = {} #Main peaks
freq_list_control = [[],[],[],[]] #Main peaks for control
mpow_list = {} #power of the main peaks
mpow_list_control = [[],[],[],[]]
peakX_list = {}
peakX_list_control = [[],[],[],[]]
peakY_list = {}
peakY_list_control = [[],[],[],[]]
peakZ_list = {}
peakZ_list_control = [[],[],[],[]]
peakU_list = {}
peakU_list_control = [[],[],[],[]]

#split freq into X,Y,Z & U, since sometimes even though peak in some axes, other axes may not have peak; to prevent addign 0 into the average, split into axes
freqX_list = {}
freqY_list = {}
freqZ_list = {}
freqU_list = {}

harm_list = {} #harmonics
harm_pow_list = {} #harmonic power
harm_list_control = [[],[],[],[]] #harmonics for control

RPC_list = {} #Relative Power Contribution to the first harmonic (RPC)

RE_bat_list = {} #for relative energy calculations, using either bat or out
RE_out_list = {}
bat_totArea = [1,1,1,1] #initialize totArea lists used for RE calculations
out_totArea = [1,1,1,1]
rest_totArea = [1,1,1,1]

basic_diagnoses_nums = {}
basic_diagnoses_nums["control"] = 0
numPD = 0

max_outputX = [0,0,0,0] #max num of peaksX seen in any file, for each tremor type
max_outputY = [0,0,0,0] #maximum number of peaksY seen in any file
max_outputZ = [0,0,0,0] #maximum number of peaksZ seen in any file
max_outputU = [0,0,0,0]

finished_one = False #used to make sure only one plot of signal is done

num_not_accounted = 0 #num of patients who don't meet criteria for being analyzed in freq-spectra
names_not_accounted = [] #names of patients who don't meet criteria for being analyzed in freq-spectra

#Cumulative Data File
allDataFile = open("AllAnalysisData.csv","w+")
allDataFile.write("Name,Date,Diagnosis (basic),")
for tremor in tremor_types:
    allDataFile.write("Mean Inst. Freq_"+tremor+",TSI_"+tremor+",TSI_amplitude_"+tremor+",Mean Peak Amplitude_"+tremor+",Stddev Peak Amplitude_+"+tremor+",TSI_amplitude to MeanPeakAmplitude ratio_"+tremor+",StddevPeakAmplitude to MeanPeakAmplitude ratio_"+tremor+",")
    allDataFile.write("Peak Freq_X_"+tremor+",Peak Freq_Y_"+tremor+",Peak Freq_Z_"+tremor+",Peak Freq_U_"+tremor+",Peak Power_X_"+tremor+",Peak Power_Y_"+tremor+",Peak Power_Z_"+tremor+",Peak Power_U_"+tremor+",Harmonic Present_X_"+tremor+",Harmonic Present_Y_"+tremor+",Harmonic Present_Z_"+tremor+",Harmonic Present_U_"+tremor+",Harmonic Power_X_"+tremor+",Harmonic Power_Y_"+tremor+",Harmonic Power_Z_"+tremor+",Harmonic Power_U_"+tremor+",RPC_X_"+tremor+",RPC_Y_"+tremor+",RPC_Z_"+tremor+",RPC_U_"+tremor+",")
allDataFile.write("RE_bat,RE_out\n")

#Data file with only files that match the inclusion criterias
allDataFile_timeseries = open("AllAnalysisData_timeseries.csv","w+")
allDataFile_timeseries.write("Name,Date,Diagnosis (basic),")
for tremor in tremor_types:
    allDataFile_timeseries.write("Mean Inst. Freq_"+tremor+",TSI_"+tremor+",TSI_amplitude_"+tremor+",Mean Peak Amplitude_"+tremor+",Stddev Peak Amplitude_+"+tremor+",TSI_amplitude to MeanPeakAmplitude ratio_"+tremor+",StddevPeakAmplitude to MeanPeakAmplitude ratio_"+tremor+",")
allDataFile_timeseries.write("\n")

allDataFile_freqspec = open("AllAnalysisData_freqspectra.csv","w+")
allDataFile_freqspec.write("Name,Date,Diagnosis (basic),")
for tremor in tremor_types:
    allDataFile_freqspec.write("Peak Freq_X_"+tremor+",Peak Freq_Y_"+tremor+",Peak Freq_Z_"+tremor+",Peak Freq_U_"+tremor+",Peak Power_X_"+tremor+",Peak Power_Y_"+tremor+",Peak Power_Z_"+tremor+",Peak Power_U_"+tremor+",Harmonic Present_X_"+tremor+",Harmonic Present_Y_"+tremor+",Harmonic Present_Z_"+tremor+",Harmonic Present_U_"+tremor+",Harmonic Power_X_"+tremor+",Harmonic Power_Y_"+tremor+",Harmonic Power_Z_"+tremor+",Harmonic Power_U_"+tremor+",RPC_X_"+tremor+",RPC_Y_"+tremor+",RPC_Z_"+tremor+",RPC_U_"+tremor+",")
allDataFile_freqspec.write("RE_bat,RE_out\n")

for i in range(len(names)):
    date = dates_all[i]
    name = names[i]
    basic_diagnosis = basic_diagnoses[i]
    
    MRN = MRNs[i]
    if name=="?":
        name = ""
    
    if MRN == "control":
        basic_diagnoses_nums["control"] = basic_diagnoses_nums["control"] + 1
    
    if basic_diagnosis not in freq_list:
        if MRN != "control":
            #Time analysis
            instAveFreq_list[basic_diagnosis] = [[],[],[],[]]
            instAvePow_list[basic_diagnosis] = [[],[],[],[]] #Ave Peak Pow
            instStdPow_list[basic_diagnosis] = [[],[],[],[]]
            TSI_list[basic_diagnosis] = [[],[],[],[]]
            TSI_ppow_list[basic_diagnosis] = [[],[],[],[]]
            instTSIppow_AvePow_ratio_list[basic_diagnosis] = [[],[],[],[]]
            instPow_Std_Ave_ratio_list[basic_diagnosis] = [[],[],[],[]]
            
            #Freq analysis
            freq_list[basic_diagnosis] = [[],[],[],[]] #main freq
            mpow_list[basic_diagnosis] = [[],[],[],[]] #mpow of main peak
            harm_list[basic_diagnosis] = [[],[],[],[]] #harmonics
            harm_pow_list[basic_diagnosis] = [[],[],[],[]] #harmonics powers
            peakX_list[basic_diagnosis] = [[],[],[],[]]
            peakY_list[basic_diagnosis] = [[],[],[],[]]
            peakZ_list[basic_diagnosis] = [[],[],[],[]]
            peakU_list[basic_diagnosis] = [[],[],[],[]]
            
            RPC_list[basic_diagnosis] = [[],[],[],[]]
            RE_bat_list[basic_diagnosis] = []
            RE_out_list[basic_diagnosis] = []
            
            freqX_list[basic_diagnosis] = [[],[],[],[]]
            freqY_list[basic_diagnosis] = [[],[],[],[]]
            freqZ_list[basic_diagnosis] = [[],[],[],[]]
            freqU_list[basic_diagnosis] = [[],[],[],[]]
            
            basic_diagnoses_nums[basic_diagnosis] = 1 #Track number of patients of each diagnosis
    else:
        if MRN != "control":
            basic_diagnoses_nums[basic_diagnosis] = basic_diagnoses_nums[basic_diagnosis] + 1
    
    noTSIforFile = False #if even one of the tremors for a given patient have TSI or TSI_ppow thats less than threshold values, don't consider this patient
    
    for j in range(len(tremor_types)):
        tremor = tremor_types[j]
        new_path = PATH + "/" + dates_all[i] + "/" + name + "/"+ tremor
        os.chdir(new_path)
        
        cmd = "ls *.csv"
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=None, shell=True)
        output = process.communicate()
        file = (str(output[0])).lstrip("b").strip("'").strip("\\n").strip()
        if file[0] == 'f' and file[1] == '_':
            continue
        
        #Analysis-function outputs:
        #******
        analysis_output = freq_analysis(file, new_path) #Freq analysis
        display_plot = False
        setFreqRange = False #only set to true if OT, since don't want to restrict freq to 2-9Hz in proecessing
        if basic_diagnosis == 'Orthostatic tremor':
            setFreqRange =  True
        
        # if tremor == 'rest' and basic_diagnosis == 'Dystonic tremor':# and finished_one == False:
        #     print(name)
        #     display_plot = True
        #     finished_one = True
        
        timeseries_output = timeseries_analysis(file, new_path, display_plot,setFreqRange) #timeseries analysis
        #******
        
        if j==0:
            bat_RE_present = False
            out_RE_present = False
            rest_RE_present = False
        
        #Time-series analysis
        f_inst = timeseries_output[0]
        df_inst = timeseries_output[1]
        f_c = timeseries_output[2]
        #f_median = timeseries_output[3]
        TSI = timeseries_output[3]
        sig_trans = timeseries_output[4]
        zero_cross = timeseries_output[5]
        tim = timeseries_output[6]
        TSI_ppow = timeseries_output[7]
        peakPowers = timeseries_output[8]
        peakPowAve = timeseries_output[9]
        peakPowStd = timeseries_output[10]
        maxPow = timeseries_output[11]
        minPow = timeseries_output[12]
        
        #to normalize these amplitude-related values to the ave. peak power
        TSIppow_AvePow_ratio = TSI_ppow/peakPowAve
        peakPow_Std_Ave_ratio = peakPowStd/peakPowAve
        
        #if any of the TSI values is less than threshold, don't consider this patient by deleting the TSI values form cumulative list afterwards
        if TSI < 0.1 or TSI_ppow < 0.001:
            noTSIforFile = True
            if MRN!="control":
                if j==0:
                    allDataFile_timeseries.write(name + "," + date + "," + basic_diagnosis + ",")
                allDataFile_timeseries.write(",,,,,,,")
        else:
            #only write data that meets inclusion criteria to 'timeseries' file
            if MRN!="control":
                if j==0:
                    allDataFile_timeseries.write(name + "," + date + "," + basic_diagnosis + ",")
                allDataFile_timeseries.write(str(f_c) + "," + str(TSI) + "," + str(TSI_ppow) + "," + str(peakPowAve) + "," + str(peakPowStd) + "," + str(TSIppow_AvePow_ratio) + "," + str(peakPow_Std_Ave_ratio)+",")
         
        
        if MRN != "control":
            instAveFreq_list[basic_diagnosis][j].append(f_c)
            TSI_list[basic_diagnosis][j].append(TSI)
            TSI_ppow_list[basic_diagnosis][j].append(TSI_ppow)
            instAvePow_list[basic_diagnosis][j].append(peakPowAve)
            instStdPow_list[basic_diagnosis][j].append(peakPowStd)
            instTSIppow_AvePow_ratio_list[basic_diagnosis][j].append(TSIppow_AvePow_ratio)
            instPow_Std_Ave_ratio_list[basic_diagnosis][j].append(peakPow_Std_Ave_ratio)
        
        #write data to AllDataFile
        #Write patient info to cumulative data file
        if j==0:
            if MRN=="control":
                allDataFile.write(name + "," + date + "," + MRN + ",")
            else:
                allDataFile.write(name + "," + date + "," + basic_diagnosis + ",")
        allDataFile.write(str(f_c) + "," + str(TSI) + "," + str(TSI_ppow) + "," + str(peakPowAve) + "," + str(peakPowStd) + "," + str(TSIppow_AvePow_ratio) + "," + str(peakPow_Std_Ave_ratio)+",")
        
        #Save f-df data for all dystonic tremor files
        #if basic_diagnosis == 'Dystonic tremor':
        with open("f_df_"+basic_diagnosis+str(basic_diagnoses_nums[basic_diagnosis])+"_"+tremor+".csv","w") as o:
            o.write("TSI,"+str(TSI)+"\n")
            o.write("f_c,"+str(f_c)+"\n")
            o.write("TSI_amplitude,"+str(TSI_ppow)+"\n")
            o.write("Ave inst. ampl,"+ str(peakPowAve) +"\n")
            o.write("Std of inst. ampl," + str(peakPowStd) +"\n")
            o.write("Max peak pow," + str(maxPow) + "\n")
            o.write("Min peak pow," + str(minPow) + "\n")
            #o.write("f_mean,"+str(f_c)+"\n")
            #o.write("f_median,"+str(f_median)+"\n")
            o.write("f,df\n")
            for k in range(len(df_inst)):
                o.write(str(f_inst[k])+","+str(df_inst[k])+"\n")
            
            #if display_plot == True:
            o.write("\n\n")
            o.write("Signal after Filters\n")
            for k in range(len(sig_trans)):
                o.write(str(tim[k])+","+str(sig_trans[k])+"\n")
            
            o.write("\n")
            o.write("Zero crossings\n")
            for k in range(len(zero_cross)):
                o.write(str(zero_cross[k])+"\n")
            o.close()
            
        
        #Freq analysis:
        th = 0
        #if basic_diagnosis == 'ET':
            # if len(analysis_output[4]) > 0:
            #     print(name,tremor)
            #     print(analysis_output[4][0][5])
            #     print(analysis_output[4][0])
            #     print("\n")
        
        # if name == 'AG' and tremor == 'bat':
        #     print(analysis_output[0])
        #     print(len(analysis_output[3]))
        #     print(analysis_output[3])
        #     print("")
        #     print(analysis_output[9])
        #     print("")
            
        if len(analysis_output[4]) > 0:
            th = analysis_output[4][0][5]
        # if th > 2: #i==18 and j==1:#i==24 and j==1: #JS = 32
        #     print(analysis_output[4][0])
        #     print(name,tremor)
        #     print(analysis_output[6],'\n')
        #     print(analysis_output[1],'\n')
        #     print(analysis_output[2],'\n')
        #     print(analysis_output[3],'\n')
        #     print(analysis_output[4],'\n')
        #     os.chdir(PATH)
        #     with open("file1.csv","w") as o:
        #         o.write("p1X,p2X,pmaxX,powmaxX,pIntegX\n")
        #         tmp = analysis_output[1]
        #         for k in range(len(tmp)):
        #             o.write(str(tmp[k][0])+","+str(tmp[k][1])+","+str(tmp[k][2])+","+str(tmp[k][3])+","+str(tmp[k][4])+"\n")
        
        if MRN == "control":
            freq_list_control[j].append(analysis_output[0])
            harm_list_control[j].append(analysis_output[6])
            peakX_list_control[j].append(analysis_output[1])
            peakY_list_control[j].append(analysis_output[2])
            peakZ_list_control[j].append(analysis_output[3])
            peakU_list_control[j].append(analysis_output[4])
            mpow_list_control[j].append(analysis_output[7])
        if MRN == "control" or MRN != "control": #doing this so have control file stats calculated for AllDataFile
            # if analysis_output[0] != [0.0,0.0,0.0,0.0]:
            #     freq_list[basic_diagnosis][j].append(analysis_output[0])
            #     mpow_list[basic_diagnosis][j].append(analysis_output[7])
            # zero = 0.0
            # if zero in analysis_output[0]:
            #     print(name,basic_diagnosis,analysis_output[0].index(zero))
                
            output_X = analysis_output[1]
            output_Y = analysis_output[2]
            output_Z = analysis_output[3]
            output_U = analysis_output[4]
            
            #Write to AllData file:
            allDataFile.write(','.join(str(e) for e in analysis_output[0]) + "," + ','.join(str(e) for e in analysis_output[7]) + "," + ','.join(str(e) for e in analysis_output[6]) + "," + ','.join(str(e) for e in analysis_output[8]) + "," + ','.join(str(e) for e in analysis_output[9]) + ",")
            
            #switch to 'or' or 'and' for more strict filtering
            #if len(output_X)>0 and len(output_Y)>0 and len(output_Z)>0 and len(output_U)>0:
            pow_thres = 0.05 #max_power above which file must be to be considered
            
            #if last file for given patient, calculated 'relative energy' (RE)
            if j==0:
                bat_totArea = analysis_output[10]
            elif j==2:
                out_totArea = analysis_output[10]
            elif j==3:
                rest_totArea = analysis_output[10]
                #print(str(bat_totArea[2])+"; "+str(out_totArea[2])+"; "+str(rest_totArea[2]))
                #Use z-axis for calculation
                
                allDataFile.write(str(rest_totArea[2]/bat_totArea[2]) + "," + str(rest_totArea[2]/out_totArea[2]) + ",")

            if analysis_output[7][0] > pow_thres or analysis_output[7][1] > pow_thres or analysis_output[7][2] > pow_thres or analysis_output[7][3] > pow_thres:
                #print(basic_diagnosis+", "+tremor+", "+name)
                if MRN!="control":
                    freq_list[basic_diagnosis][j].append(analysis_output[0])
                    mpow_list[basic_diagnosis][j].append(analysis_output[7])
                    harm_list[basic_diagnosis][j].append(analysis_output[6]) #harmonics
                    RPC_list[basic_diagnosis][j].append(analysis_output[9])
                    
                    if analysis_output[8] != [0.0,0.0,0.0,0.0]:
                        harm_pow_list[basic_diagnosis][j].append(analysis_output[8])
                
                    if j==0:
                        allDataFile_freqspec.write(name + "," + date + "," + basic_diagnosis + ",")
                        bat_RE_present = True
                    elif j==2:
                        out_RE_present = True
                    elif j==3:
                        rest_RE_present = True
                    allDataFile_freqspec.write(','.join(str(e) for e in analysis_output[0]) + "," + ','.join(str(e) for e in analysis_output[7]) + "," + ','.join(str(e) for e in analysis_output[6]) + "," + ','.join(str(e) for e in analysis_output[8]) + "," + ','.join(str(e) for e in analysis_output[9]) + ",")
                
                #write RE data to file, RE_bat & RE_out
                if j==3: #only write RE data to AllData_freqspec if on j==3
                    rest_RE_present = True
                    if MRN!="control":
                        RE_bat_list[basic_diagnosis].append(rest_totArea[2]/bat_totArea[2])
                        RE_out_list[basic_diagnosis].append(rest_totArea[2]/out_totArea[2])
                    
                        #allDataFile_freqspec.write(str(rest_totArea[2]/bat_totArea[2]) + "," + str(rest_totArea[2]/out_totArea[2]) + ",")
            else:
                #print("name: "+ name+", date: "+ date+ ", tremor: "+tremor+", diag: "+basic_diagnosis)
                if MRN!="control":
                    if j==0:
                        allDataFile_freqspec.write(name + "," + date + "," + basic_diagnosis + ",")
                    allDataFile_freqspec.write(",,,," + ",,,," + ",,,," + ",,,," + ",,,,")
                if name not in names_not_accounted:
                    names_not_accounted.append((name + ", " + basic_diagnosis))
            if j==3:
                if bat_RE_present == True and out_RE_present == True and rest_RE_present == True and MRN!="control":
                    allDataFile_freqspec.write(str(rest_totArea[2]/bat_totArea[2]) + "," + str(rest_totArea[2]/out_totArea[2]) + ",")
                else:
                    if MRN!="control":
                        allDataFile_freqspec.write(",,")
            
            #for each axis
            if len(output_X)>0:
                freqX_list[basic_diagnosis][j].append(analysis_output[0][0])
                peakX_list[basic_diagnosis][j].append(output_X)
            if len(output_Y)>0:
                freqY_list[basic_diagnosis][j].append(analysis_output[0][1])
                peakY_list[basic_diagnosis][j].append(output_Y)
            if len(output_Z)>0:
                freqZ_list[basic_diagnosis][j].append(analysis_output[0][2])
                peakZ_list[basic_diagnosis][j].append(output_Z)
            if len(output_U)>0:
                if output_U[0][5] < 1:
                    freqU_list[basic_diagnosis][j].append(analysis_output[0][3])
                    peakU_list[basic_diagnosis][j].append(output_U)
            
            #peakX_list[basic_diagnosis][j].append(output_X)
            #peakY_list[basic_diagnosis][j].append(output_Y)
            #peakZ_list[basic_diagnosis][j].append(output_Z)
            #peakU_list[basic_diagnosis][j].append(output_U)
            
            # if len(outputX) > max_outputX[j]:
            #     max_outputX[j] = len(outputX)
            # if len(outputY) > max_outputY[j]:
            #     max_outputY[j] = len(outputY)
            # if len(outputZ) > max_outputZ[j]:
            #     max_outputZ[j] = len(outputZ)
            
    
    #delete this patient from all cumulative TSI lists if even one of the four tremors had a TSI that didn't meet the threshold
    if noTSIforFile == True and MRN != "control":
        for j in range(len(tremor_types)):
            del instAveFreq_list[basic_diagnosis][j][-1]
            del TSI_list[basic_diagnosis][j][-1]
            del TSI_ppow_list[basic_diagnosis][j][-1]
            del instAvePow_list[basic_diagnosis][j][-1]
            del instStdPow_list[basic_diagnosis][j][-1]
            del instTSIppow_AvePow_ratio_list[basic_diagnosis][j][-1]
            del instPow_Std_Ave_ratio_list[basic_diagnosis][j][-1]
    
    allDataFile.write("\n")
    allDataFile_freqspec.write("\n")
    allDataFile_timeseries.write("\n")

num_not_accounted = len(names_not_accounted)
print("num_not_accounted: "+ str(num_not_accounted))
            
#Time-series ave's/std's
TSI_ave = {}
TSI_std = {}
TSI_num = {}
instAveFreq_ave = {}
instAveFreq_std = {}
instAveFreq_num = {}
TSI_ppow_ave = {}
TSI_ppow_std = {}
TSI_ppow_num = {}
instAvePow_ave = {}
instAvePow_std = {}
instAvePow_num = {}
instStdPow_ave = {}
instStdPow_std = {}
instStdPow_num = {}

instTSIppow_AvePow_ratio_ave = {}
instTSIppow_AvePow_ratio_std = {}
instTSIppow_AvePow_ratio_num = {}

instPow_Std_Ave_ratio_ave = {}
instPow_Std_Ave_ratio_std = {}
instPow_Std_Ave_ratio_num = {}

#Freq ave's/std's
freq_ave = {}
freq_std = {}
freq_num = {}

#break freq into individual axes
freqX_ave = {}
freqX_std = {}
freqX_num = {}
freqY_ave = {}
freqY_std = {}
freqY_num = {}
freqZ_ave = {}
freqZ_std = {}
freqZ_num = {}
freqU_ave = {}
freqU_std = {}
freqU_num = {}

mpow_ave = {}
mpow_std = {}
mpow_num = {}

harm_ave = {}
harm_std = {}
harm_num = {}

RPC_ave = {}
RPC_std = {}
RPC_num = {}

#RE
RE_bat_ave = {}
RE_out_ave = {}
RE_bat_std = {}
RE_out_std = {}

harm_pow_ave = {}
harm_pow_std = {}
harm_pow_num = {}

peakX_sorted = {}
peakY_sorted = {}
peakZ_sorted = {}
peakU_sorted = {}

peakX_sorted_ave = {}
peakX_sorted_num = {}
peakX_sorted_std = {}
peakY_sorted_ave = {}
peakY_sorted_num = {}
peakY_sorted_std = {}
peakZ_sorted_ave = {}
peakZ_sorted_num = {}
peakZ_sorted_std = {}
peakU_sorted_ave = {}
peakU_sorted_num = {}
peakU_sorted_std = {}

num_type = 0

#time-series analysis
for diag in TSI_list:
    TSIs = TSI_list[diag]
    instFreqs = instAveFreq_list[diag]
    
    TSI_ave[diag] = [0.0,0.0,0.0,0.0] #element for bat, rest, out, kin respectively
    TSI_std[diag] = [0.0,0.0,0.0,0.0]
    TSI_num[diag] = [0.0,0.0,0.0,0.0]
    instAveFreq_ave[diag] = [0.0,0.0,0.0,0.0]
    instAveFreq_std[diag] = [0.0,0.0,0.0,0.0]
    instAveFreq_num[diag] = [0.0,0.0,0.0,0.0]
    
    for i in range(4):#go through each tremor type
        TSI_ave[diag][i] = average(TSIs[i])
        instAveFreq_ave[diag][i] = average(instFreqs[i])
        TSI_std[diag][i] = stddev(TSIs[i],TSI_ave[diag][i])
        instAveFreq_std[diag][i] = stddev(instFreqs[i],instAveFreq_ave[diag][i])
        TSI_num[diag] = len((TSI_list[diag])[0])
        instAveFreq_num[diag] = len((instAveFreq_list[diag])[0])

#time-series peak-power analysis
for diag in TSI_ppow_list:
    TSI_ppow = TSI_ppow_list[diag]
    instPows = instAvePow_list[diag]
    instStdPows = instStdPow_list[diag]
    instTSIpows_ratios = instTSIppow_AvePow_ratio_list[diag]
    instStdAve_ratios = instPow_Std_Ave_ratio_list[diag]
    
    TSI_ppow_ave[diag] = [0.0,0.0,0.0,0.0] #element for bat, rest, out, kin respectively
    TSI_ppow_std[diag] = [0.0,0.0,0.0,0.0]
    TSI_ppow_num[diag] = [0.0,0.0,0.0,0.0]
    instAvePow_ave[diag] = [0.0,0.0,0.0,0.0]
    instAvePow_std[diag] = [0.0,0.0,0.0,0.0]
    instAvePow_num[diag] = [0.0,0.0,0.0,0.0]
    instStdPow_ave[diag] = [0.0,0.0,0.0,0.0]
    instStdPow_std[diag] = [0.0,0.0,0.0,0.0]
    instStdPow_num[diag] = [0.0,0.0,0.0,0.0]
    
    instTSIppow_AvePow_ratio_ave[diag] = [0.0,0.0,0.0,0.0]
    instTSIppow_AvePow_ratio_std[diag] = [0.0,0.0,0.0,0.0]
    instTSIppow_AvePow_ratio_num[diag] = [0.0,0.0,0.0,0.0]

    instPow_Std_Ave_ratio_ave[diag] = [0.0,0.0,0.0,0.0]
    instPow_Std_Ave_ratio_std[diag] = [0.0,0.0,0.0,0.0]
    instPow_Std_Ave_ratio_num[diag] = [0.0,0.0,0.0,0.0]
    
    for i in range(4):#go through each tremor type
        TSI_ppow_ave[diag][i] = average(TSI_ppow[i])
        instAvePow_ave[diag][i] = average(instPows[i])
        instStdPow_ave[diag][i] = average(instStdPows[i])
        instTSIppow_AvePow_ratio_ave[diag][i] = average(instTSIpows_ratios[i])
        instPow_Std_Ave_ratio_ave[diag][i] = average(instStdAve_ratios[i])
        
        TSI_ppow_std[diag][i] = stddev(TSI_ppow[i],TSI_ppow_ave[diag][i])
        instAvePow_std[diag][i] = stddev(instPows[i],instAvePow_ave[diag][i])
        instStdPow_std[diag][i] = stddev(instStdPows[i],instStdPow_ave[diag][i])
        instTSIppow_AvePow_ratio_std[diag][i] = stddev(instTSIpows_ratios[i],instTSIppow_AvePow_ratio_ave[diag][i])
        instPow_Std_Ave_ratio_std[diag][i] = stddev(instStdAve_ratios[i],instPow_Std_Ave_ratio_ave[diag][i])
        
        TSI_ppow_num[diag] = len((TSI_ppow_list[diag])[i])
        instAvePow_num[diag] = len((instAvePow_list[diag])[i])
        instStdPow_num[diag] = len((instStdPow_list[diag])[i])   
        instTSIppow_AvePow_ratio_num[diag] = len((instTSIppow_AvePow_ratio_list[diag])[i])
        instPow_Std_Ave_ratio_num[diag] = len((instPow_Std_Ave_ratio_list[diag])[i])
    
#save time-series analysis to file
os.chdir(PATH)
with open("timeseries_analysis.csv","w") as o:
    #Write main freqs
    o.write("Average TSI,,,,,,Ave Inst. Freq")
    o.write("\nDiagnosis,Bat,Kin,Out,Rest,,Bat,Kin,Out,Rest\n")
    for diag in TSI_ave:
        o.write(str(diag)+",")
        frs = TSI_ave[diag]
        for i in range(len(frs)):
            o.write(str(frs[i])+",")
        frs = instAveFreq_ave[diag]
        o.write(",")
        for i in range(len(frs)):
            o.write(str(frs[i])+",")
        o.write("\n")
    
    o.write("\n\n")
        
    o.write("Stddev TSI,,,,,,Stddev Inst. Freq")
    o.write("\nDiagnosis,Bat,Kin,Out,Rest,,Bat,Kin,Out,Rest\n")
    for diag in TSI_std:
        o.write(str(diag)+",")
        frs = TSI_std[diag]
        for i in range(len(frs)):
            o.write(str(frs[i])+",")
        frs = instAveFreq_std[diag]
        o.write(",")
        for i in range(len(frs)):
            o.write(str(frs[i])+",")
        o.write("\n")
    
    o.write("\n\n")
    
    o.write("Num TSI,,,,,,Num Inst. Freq")
    o.write("\nDiagnosis,Bat,Kin,Out,Rest,,Bat,Kin,Out,Rest\n")
    for diag in TSI_std:
        o.write(str(diag)+",")
        frs = TSI_num[diag]
        for i in range(4):
            o.write(str(frs)+",")
        frs = instAveFreq_num[diag]
        o.write(",")
        for i in range(4):
            o.write(str(frs)+",")
        o.write("\n")
    
    o.write("\n\n")
    
    o.write("SEM TSI,,,,,,SEM Inst. Freq")
    o.write("\nDiagnosis,Bat,Kin,Out,Rest,,Bat,Kin,Out,Rest\n")
    for diag in TSI_std:
        o.write(str(diag)+",")
        frs1 = TSI_std[diag]
        frs2 = TSI_num[diag]
        for i in range(len(frs1)):
            if frs2 > 0:
                o.write(str(frs1[i]/(math.sqrt(frs2)))+",")
            else:
                o.write(",")
        frs1 = instAveFreq_std[diag]
        frs2 = instAveFreq_num[diag]
        o.write(",")
        for i in range(len(frs1)):
            if frs2 > 0:
                o.write(str(frs1[i]/(math.sqrt(frs2)))+",")
            else:
                o.write(",")
        o.write("\n")
    
    o.write("\n\n")
    
    o.write("Average TSI Peak Power,,,,,,Ave Inst. Peak Power,,,,,Stddev Inst. Peak Power (within file),,,,,Stddev Inst. Peak Power (across files)")
    o.write("\nDiagnosis,Bat,Kin,Out,Rest,,Bat,Kin,Out,Rest,,Bat,Kin,Out,Rest\n")
    for diag in TSI_ppow_ave:
        o.write(str(diag)+",")
        frs = TSI_ppow_ave[diag]
        for i in range(len(frs)):
            o.write(str(frs[i])+",")
        frs = instAvePow_ave[diag]
        o.write(",")
        for i in range(len(frs)):
            o.write(str(frs[i])+",")
        frs = instStdPow_ave[diag]
        o.write(",")
        for i in range(len(frs)):
            o.write(str(frs[i])+",")
        frs = instAvePow_std[diag]
        o.write(",")
        for i in range(len(frs)):
            o.write(str(frs[i])+",")
        o.write("\n")
    
    o.write("\n\n")
    
    o.write("Average TSI:AveragePower ratio,,,,,,Ave Peak Power std:ave ratio")
    o.write("\nDiagnosis,Bat,Kin,Out,Rest,,Bat,Kin,Out,Rest\n")
    for diag in instTSIppow_AvePow_ratio_ave:
        o.write(str(diag)+",")
        frs = instTSIppow_AvePow_ratio_ave[diag]
        for i in range(len(frs)):
            o.write(str(frs[i])+",")
        frs = instPow_Std_Ave_ratio_ave[diag]
        o.write(",")
        for i in range(len(frs)):
            o.write(str(frs[i])+",")
        o.write("\n")
    
    o.write("\n\n")
    
    o.write("Stddev TSI Peak Power,,,,,,Stddev Inst. Peak Power,,,,,Stddev of stddev Inst. Peak Power ")
    o.write("\nDiagnosis,Bat,Kin,Out,Rest,,Bat,Kin,Out,Rest\n")
    for diag in TSI_ppow_std:
        o.write(str(diag)+",")
        frs = TSI_ppow_std[diag]
        for i in range(len(frs)):
            o.write(str(frs[i])+",")
        frs = instAvePow_std[diag]
        o.write(",")
        for i in range(len(frs)):
            o.write(str(frs[i])+",")
        frs = instStdPow_std[diag]
        o.write(",")
        for i in range(len(frs)):
            o.write(str(frs[i])+",")
        o.write("\n")
    
    o.write("\n\n")
    
    o.write("Stddev TSI:AveragePower ratio,,,,,,Stddev Peak Power std:ave ratio")
    o.write("\nDiagnosis,Bat,Kin,Out,Rest,,Bat,Kin,Out,Rest\n")
    for diag in instTSIppow_AvePow_ratio_std:
        o.write(str(diag)+",")
        frs = instTSIppow_AvePow_ratio_std[diag]
        for i in range(len(frs)):
            o.write(str(frs[i])+",")
        frs = instPow_Std_Ave_ratio_std[diag]
        o.write(",")
        for i in range(len(frs)):
            o.write(str(frs[i])+",")
        o.write("\n")
    
    o.write("\n\n")
        
    
#Freq analysis
for diag in freq_list:
    freqs = freq_list[diag]
    #Each list has 4-element array for x,y,z,y respectively
    freq_ave[diag] = [[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0]]
    freq_std[diag] = [[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0]]
    
    #for each axes:
    freqX = freqX_list[diag]
    freqY = freqY_list[diag]
    freqZ = freqZ_list[diag]
    freqU = freqU_list[diag]
    
    freqX_ave[diag] = [0.0,0.0,0.0,0.0]
    freqX_std[diag] = [0.0,0.0,0.0,0.0]
    freqX_num[diag] = [0.0,0.0,0.0,0.0]
    freqY_ave[diag] = [0.0,0.0,0.0,0.0]
    freqY_std[diag] = [0.0,0.0,0.0,0.0]
    freqY_num[diag] = [0.0,0.0,0.0,0.0]
    freqZ_ave[diag] = [0.0,0.0,0.0,0.0]
    freqZ_std[diag] = [0.0,0.0,0.0,0.0]
    freqZ_num[diag] = [0.0,0.0,0.0,0.0]
    freqU_ave[diag] = [0.0,0.0,0.0,0.0]
    freqU_std[diag] = [0.0,0.0,0.0,0.0]
    freqU_num[diag] = [0.0,0.0,0.0,0.0]
    
    
    mpows = mpow_list[diag]
    #Each list has 4-element array for x,y,z,y respectively
    mpow_ave[diag] = [[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0]]
    mpow_std[diag] = [[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0]]
    
    RPC = RPC_list[diag]
    RPC_ave[diag] = [[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0]]
    RPC_std[diag] = [[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0]]
    
    RE_bat = RE_bat_list[diag]
    RE_bat_ave[diag] = 0.0
    RE_out_ave[diag] = 0.0
    RE_bat_std[diag] = 0.0
    RE_out_std[diag] = 0.0
    
    harm = harm_list[diag]
    harm_pow = harm_pow_list[diag]
    #Each list has 4-element array for x,y,z,y respectively
    harm_ave[diag] = [[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0]]
    harm_std[diag] = [[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0]]
    
    harm_pow_ave[diag] = [[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0]]
    harm_pow_std[diag] = [[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0]]
    
    peakX_sorted[diag] = [[],[],[],[]]
    peakY_sorted[diag] = [[],[],[],[]]
    peakZ_sorted[diag] = [[],[],[],[]]
    peakU_sorted[diag] = [[],[],[],[]]
    
    peakX_sorted_ave[diag] = [[],[],[],[]]
    peakX_sorted_num[diag] = [[],[],[],[]] #num of peaks within i-th peak # ave
    peakX_sorted_std[diag] = [[],[],[],[]]
    peakY_sorted_ave[diag] = [[],[],[],[]]
    peakY_sorted_num[diag] = [[],[],[],[]]
    peakY_sorted_std[diag] = [[],[],[],[]]
    peakZ_sorted_ave[diag] = [[],[],[],[]]
    peakZ_sorted_num[diag] = [[],[],[],[]]
    peakZ_sorted_std[diag] = [[],[],[],[]]
    peakU_sorted_ave[diag] = [[],[],[],[]]
    peakU_sorted_num[diag] = [[],[],[],[]]
    peakU_sorted_std[diag] = [[],[],[],[]]
    
    freq_num[diag] = len((freq_list[diag])[0])
    mpow_num[diag] = len((mpow_list[diag])[0])
    
    #for each axis: fix this
    # freqX_num[diag] = len((freqX_list[diag])[0])
    # freqY_num[diag] = len((freqY_list[diag])[0])
    # freqZ_num[diag] = len((freqZ_list[diag])[0])
    # freqU_num[diag] = len((freqU_list[diag])[0])
    
    
    print(diag+": "+str(freq_num[diag]))
    
    RE_bat_ave[diag] = average(RE_bat_list[diag])
    RE_bat_std[diag] = stddev(RE_bat_list[diag],RE_bat_ave[diag])
    RE_out_ave[diag] = average(RE_out_list[diag])
    RE_out_std[diag] = stddev(RE_out_list[diag],RE_out_ave[diag])
    
    for i in range(4):#go through each tremor type
        if [] not in freq_list[diag]:
            freq_ave[diag][i] = ave(freqs[i])
            freq_std[diag][i] = std(freqs[i],freq_ave[diag][i])
            
            mpow_ave[diag][i] = ave(mpows[i])
            mpow_std[diag][i] = std(mpows[i],mpow_ave[diag][i])
            
            RPC_ave[diag][i] = ave(RPC[i])
            RPC_std[diag][i] = std(RPC[i],RPC_ave[diag][i])
            
            harm_ave[diag][i] = ave(harm[i])
            harm_std[diag][i] = std(harm[i],harm_ave[diag][i])
        
            harm_pow_ave[diag][i] = ave(harm_pow[i])
            harm_pow_std[diag][i] = std(harm_pow[i],harm_pow_ave[diag][i])
        if [] not in freqX_list[diag]:
            freqX_ave[diag][i] = average(freqX[i])
            freqY_ave[diag][i] = average(freqY[i])
            freqZ_ave[diag][i] = average(freqZ[i])
            freqU_ave[diag][i] = average(freqU[i])
            freqX_std[diag][i] = stddev(freqX[i],freqX_ave[diag][i])
            freqY_std[diag][i] = stddev(freqY[i],freqY_ave[diag][i])
            freqZ_std[diag][i] = stddev(freqZ[i],freqZ_ave[diag][i])
            freqU_std[diag][i] = stddev(freqU[i],freqU_ave[diag][i])
            freqX_num[diag][i] = len(freqX[i])
            freqY_num[diag][i] = len(freqY[i])
            freqZ_num[diag][i] = len(freqZ[i])
            freqU_num[diag][i] = len(freqU[i])
            
            # #Print for checking
            # if i==2 and diag =='PD':
            #     for j in range(len(freqX[i])):
            #         print(str(freqX[i][j])+"\n")
        
        #Sort peaks   
        num_type = min(len(peakX_list[diag][i]),len(peakY_list[diag][i]),len(peakZ_list[diag][i])) #number of files in each diag
        for j in range(num_type):
            peaksX = peakX_list[diag][i][j] #peaks in the given file
            peaksY = peakY_list[diag][i][j]
            peaksZ = peakZ_list[diag][i][j]
            peaksU = peakZ_list[diag][i][j]
            for k in range(len(peaksX)):
                if (k+1) > len(peakX_sorted[diag][i]):
                    peakX_sorted[diag][i].append([])#add position for kth-peak
                peakX_sorted[diag][i][k].append(peaksX[k])
            for k in range(len(peaksY)):
                if (k+1) > len(peakY_sorted[diag][i]):
                    peakY_sorted[diag][i].append([])#add position for kth-peak
                peakY_sorted[diag][i][k].append(peaksY[k])
            for k in range(len(peaksZ)):
                if (k+1) > len(peakZ_sorted[diag][i]):
                    peakZ_sorted[diag][i].append([])#add position for kth-peak
                peakZ_sorted[diag][i][k].append(peaksZ[k])
            for k in range(len(peaksU)):
                if (k+1) > len(peakU_sorted[diag][i]):
                    peakU_sorted[diag][i].append([])#add position for kth-peak
                peakU_sorted[diag][i][k].append(peaksU[k])
        
        #average the individual peak data
        for j in range(len(peakX_sorted[diag][i])):
            if (j+1) > len(peakX_sorted_ave[diag][i]):
                peakX_sorted_ave[diag][i].append([])
                peakX_sorted_num[diag][i].append([])
                peakX_sorted_std[diag][i].append([])
            peakX_sorted_ave[diag][i][j] = ave(peakX_sorted[diag][i][j])
            peakX_sorted_num[diag][i][j] = len(peakX_sorted[diag][i][j])
            peakX_sorted_std[diag][i][j] = std(peakX_sorted[diag][i][j],peakX_sorted_ave[diag][i][j])
        for j in range(len(peakY_sorted[diag][i])):
            if (j+1) > len(peakY_sorted_ave[diag][i]):
                peakY_sorted_ave[diag][i].append([])
                peakY_sorted_num[diag][i].append([])
                peakY_sorted_std[diag][i].append([])
            peakY_sorted_ave[diag][i][j] = ave(peakY_sorted[diag][i][j])
            peakY_sorted_num[diag][i][j] = len(peakY_sorted[diag][i][j])
            peakY_sorted_std[diag][i][j] = std(peakY_sorted[diag][i][j],peakY_sorted_ave[diag][i][j])
        for j in range(len(peakZ_sorted[diag][i])):
            if (j+1) > len(peakZ_sorted_ave[diag][i]):
                peakZ_sorted_ave[diag][i].append([])
                peakZ_sorted_num[diag][i].append([])
                peakZ_sorted_std[diag][i].append([])
            peakZ_sorted_ave[diag][i][j] = ave(peakZ_sorted[diag][i][j])
            peakZ_sorted_num[diag][i][j] = len(peakZ_sorted[diag][i][j])
            peakZ_sorted_std[diag][i][j] = std(peakZ_sorted[diag][i][j],peakZ_sorted_ave[diag][i][j])
        #Composite peaks:
        for j in range(len(peakU_sorted[diag][i])):
            if (j+1) > len(peakU_sorted_ave[diag][i]):
                peakU_sorted_ave[diag][i].append([])
                peakU_sorted_num[diag][i].append([])
                peakU_sorted_std[diag][i].append([])
            peakU_sorted_ave[diag][i][j] = ave(peakU_sorted[diag][i][j])
            peakU_sorted_num[diag][i][j] = len(peakU_sorted[diag][i][j])
            peakU_sorted_std[diag][i][j] = std(peakU_sorted[diag][i][j],peakU_sorted_ave[diag][i][j])


freq_ave_control = []
freq_std_control = []
freq_num_control = 0
mpow_ave_control = []
mpow_std_control = []
mpow_num_control = 0
harm_ave_control = []
harm_std_control = []
harm_num_control = 0

freqs_control = freq_list_control
#each array in the list has 4 elements, since x,y,z, & u
freq_ave_control = [[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0]]
freq_std_control = [[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0]]
freq_num_control = len(freqs_control[0])
#print(freq_num_control)

for i in range(4):
    freq_ave_control[i] = ave(freqs_control[i])
    freq_std_control[i] = std(freqs_control[i],freq_ave_control[i])

#Num of patients in each diagnosis group
os.chdir(PATH)
with open("num_patients.csv","w") as o:
    diag_keys = basic_diagnoses_nums.keys()
    o.write("Diagnosis,num of patients\n")
    sum = 0
    for i in basic_diagnoses_nums:
        o.write(str(i)+","+str(basic_diagnoses_nums[i])+"\n")
        sum += basic_diagnoses_nums[i]
    o.write("Total,"+str(sum))

os.chdir(PATH)
with open("tremor_analysis.csv","w") as o:
    #Write main freqs
    o.write("Main Freqs,\nDiagnosis,Bat Average X_freq,Bat Average Y_freq,Bat Average Z_freq,Bat Average U_freq,Kin Average X_freq,Kin Average Y_freq,Kin Average Z_freq,Kin Average U_freq,Out Average X_freq,Out Average Y_freq,Out Average Z_freq,Out Average U_freq,Rest Average X_freq,Rest Average Y_freq,Rest Average Z_freq,Rest Average U_freq\n")
    for diag in freq_ave:
        o.write(str(diag)+",")
        # frs = freq_ave[diag]
        # for i in range(len(frs)):
        #     for j in range(len(frs[i])):
        #         o.write(str(frs[i][j])+",")
        frs = freqX_ave[diag]
        for i in range(len(frs)):
            o.write(str(frs[i])+",")
        frs = freqY_ave[diag]
        for i in range(len(frs)):
            o.write(str(frs[i])+",")
        frs = freqZ_ave[diag]
        for i in range(len(frs)):
            o.write(str(frs[i])+",")
        frs = freqU_ave[diag]
        for i in range(len(frs)):
            o.write(str(frs[i])+",")
        o.write("\n")
    
    o.write("\n\n")
        
    o.write("Diagnosis,Bat Std X_freq,Bat Std Y_freq,Bat Std Z_freq,Bat Std U_freq,Kin Std X_freq,Kin Std Y_freq,Kin Std Z_freq,Kin Std U_freq,Out Std X_freq,Out Std Y_freq,Out Std Z_freq,Out Std U_freq,Rest Std X_freq,Rest Std Y_freq,Rest Std Z_freq,Rest Std U_freq\n")
    for diag in freq_std:
        o.write(str(diag)+",")
        # frs = freq_std[diag]
        # for i in range(len(frs)):
        #     for j in range(len(frs[i])):
        #         o.write(str(frs[i][j])+",")
        # o.write("\n")
        # o.write(str(diag)+",")
        frs = freqX_std[diag]
        for i in range(len(frs)):
            o.write(str(frs[i])+",")
        frs = freqY_std[diag]
        for i in range(len(frs)):
            o.write(str(frs[i])+",")
        frs = freqZ_std[diag]
        for i in range(len(frs)):
            o.write(str(frs[i])+",")
        frs = freqU_std[diag]
        for i in range(len(frs)):
            o.write(str(frs[i])+",")
        o.write("\n")
    
    o.write("\n\n")
    
    o.write("Diagnosis,Bat CV X_freq,Bat CV Y_freq,Bat CV Z_freq,Bat CV U_freq,Kin CV X_freq,Kin CV Y_freq,Kin CV Z_freq,Kin CV U_freq,Out CV X_freq,Out CV Y_freq,Out CV Z_freq,Out CV U_freq,Rest CV X_freq,Rest CV Y_freq,Rest CV Z_freq,Rest CV U_freq\n")
    for diag in freq_std:
        o.write(str(diag)+",")
        # frs = freq_std[diag]
        # frs2 = freq_ave[diag]
        # for i in range(len(frs)):
        #     for j in range(len(frs[i])):
        #         if frs2[i][j]>0:
        #             o.write(str(frs[i][j]/frs2[i][j])+",")
        #         else:
        #             o.write("0.0,")
        frs = freqX_std[diag]
        frs2 = freqX_ave[diag]
        for i in range(len(frs)):
            if frs2[i]>0:
                o.write(str(frs[i]/frs2[i])+",")
            else:
                o.write("0.0,")
        frs = freqY_std[diag]
        frs2 = freqY_ave[diag]
        for i in range(len(frs)):
            if frs2[i]>0:
                o.write(str(frs[i]/frs2[i])+",")
            else:
                o.write("0.0,")
        frs = freqZ_std[diag]
        frs2 = freqZ_ave[diag]
        for i in range(len(frs)):
            if frs2[i]>0:
                o.write(str(frs[i]/frs2[i])+",")
            else:
                o.write("0.0,")
        frs = freqU_std[diag]
        frs2 = freqU_ave[diag]
        for i in range(len(frs)):
            if frs2[i]>0:
                o.write(str(frs[i]/frs2[i])+",")
            else:
                o.write("0.0,")
        
        
        o.write("\n")
    
    o.write("\n\n")
    
    o.write("Diagnosis,Bat Num X_freq,Bat Num Y_freq,Bat Num Z_freq,Bat Num U_freq,Kin Num X_freq,Kin Num Y_freq,Kin Num Z_freq,Kin Num U_freq,Out Num X_freq,Out Num Y_freq,Out Num Z_freq,Out Num U_freq,Rest Num X_freq,Rest Num Y_freq,Rest Num Z_freq,Rest Num U_freq\n")
    for diag in freq_ave:
        o.write(str(diag)+",")
        # frs = freq_std[diag]
        # for i in range(len(frs)):
        #     for j in range(len(frs[i])):
        #         o.write(str(frs[i][j])+",")
        # o.write("\n")
        # o.write(str(diag)+",")
        frs = freqX_num[diag]
        for i in range(len(frs)):
            o.write(str(frs[i])+",")
        frs = freqY_num[diag]
        for i in range(len(frs)):
            o.write(str(frs[i])+",")
        frs = freqZ_num[diag]
        for i in range(len(frs)):
            o.write(str(frs[i])+",")
        frs = freqU_num[diag]
        for i in range(len(frs)):
            o.write(str(frs[i])+",")
        o.write("\n")
    
    o.write("\n\n")
    
    #Write max pow of main freqs
    o.write("Max Power of Main Freqs,\nDiagnosis,Bat Average X_pow,Bat Average Y_pow,Bat Average Z_pow,Bat Average U_pow,Kin Average X_pow,Kin Average Y_pow,Kin Average Z_pow,Kin Average U_pow,Out Average X_pow,Out Average Y_pow,Out Average Z_pow,Out Average U_pow,Rest Average X_pow,Rest Average Y_pow,Rest Average Z_pow,Rest Average U_pow\n")
    for diag in mpow_ave:
        o.write(str(diag)+",")
        frs = mpow_ave[diag]
        for i in range(len(frs)):
            for j in range(len(frs[i])):
                o.write(str(frs[i][j])+",")
        o.write("\n")
    
    o.write("\n\n")
        
    o.write("Diagnosis,Bat Std X_pow,Bat Std Y_pow,Bat Std Z_pow,Bat Std U_pow,Kin Std X_pow,Kin Std Y_pow,Kin Std Z_pow,Kin Std U_pow,Out Std X_pow,Out Std Y_pow,Out Std Z_pow,Out Std U_pow,Rest Std X_pow,Rest Std Y_pow,Rest Std Z_pow,Rest Std U_pow\n")
    for diag in mpow_std:
        o.write(str(diag)+",")
        frs = mpow_std[diag]
        for i in range(len(frs)):
            for j in range(len(frs[i])):
                o.write(str(frs[i][j])+",")
        o.write("\n")
    
    o.write("\n\n")
    
    o.write("Diagnosis,Bat CV X_pow,Bat CV Y_pow,Bat CV Z_pow,Bat CV U_pow,Kin CV X_pow,Kin CV Y_pow,Kin CV Z_pow,Kin CV U_pow,Out CV X_pow,Out CV Y_pow,Out CV Z_pow,Out CV U_pow,Rest CV X_pow,Rest CV Y_pow,Rest CV Z_pow,Rest CV U_pow\n")
    for diag in mpow_ave:
        o.write(str(diag)+",")
        frs = mpow_std[diag]
        frs2 = mpow_ave[diag]
        for i in range(len(frs)):
            for j in range(len(frs[i])):
                if frs2[i][j]>0:
                    o.write(str(frs[i][j]/frs2[i][j])+",")
                else:
                    o.write("0.0,")
        o.write("\n")
    
    o.write("\n\n")
    
    #Write harmonics
    o.write("Harmonics,\nDiagnosis,Bat Average X_harm,Bat Average Y_harm,Bat Average Z_harm,Bat Average U_harm,Kin Average X_harm,Kin Average Y_harm,Kin Average Z_harm,Kin Average U_harm,Out Average X_harm,Out Average Y_harm,Out Average Z_harm,Out Average U_harm,Rest Average X_harm,Rest Average Y_harm,Rest Average Z_harm,Rest Average U_harm\n")
    for diag in harm_ave:
        o.write(str(diag)+",")
        frs = harm_ave[diag]
        for i in range(len(frs)):
            for j in range(len(frs[i])):
                o.write(str(frs[i][j])+",")
        o.write("\n")
    
    o.write("\n\n")
        
    o.write("Diagnosis,Bat Std X_harm,Bat Std Y_harm,Bat Std Z_harm,Bat Std U_harm,Kin Std X_harm,Kin Std Y_harm,Kin Std Z_harm,Kin Std U_harm,Out Std X_harm,Out Std Y_harm,Out Std Z_harm,Out Std U_harm,Rest Std X_harm,Rest Std Y_harm,Rest Std Z_harm,Rest Std U_harm\n")
    for diag in harm_std:
        o.write(str(diag)+",")
        frs = harm_std[diag]
        for i in range(len(frs)):
            for j in range(len(frs[i])):
                o.write(str(frs[i][j])+",")
        o.write("\n")
    
    o.write("\n\n")
    
    #Harmonic power
    o.write("1st Harmonic Power,\nDiagnosis,Bat Average X_harm,Bat Average Y_harm,Bat Average Z_harm,Bat Average U_harm,Kin Average X_harm,Kin Average Y_harm,Kin Average Z_harm,Kin Average U_harm,Out Average X_harm,Out Average Y_harm,Out Average Z_harm,Out Average U_harm,Rest Average X_harm,Rest Average Y_harm,Rest Average Z_harm,Rest Average U_harm\n")
    for diag in harm_pow_ave:
        o.write(str(diag)+",")
        frs = harm_pow_ave[diag]
        for i in range(len(frs)):
            for j in range(len(frs[i])):
                o.write(str(frs[i][j])+",")
        o.write("\n")
    
    o.write("\n\n")
        
    o.write("Diagnosis,Bat Std X_harm,Bat Std Y_harm,Bat Std Z_harm,Bat Std U_harm,Kin Std X_harm,Kin Std Y_harm,Kin Std Z_harm,Kin Std U_harm,Out Std X_harm,Out Std Y_harm,Out Std Z_harm,Out Std U_harm,Rest Std X_harm,Rest Std Y_harm,Rest Std Z_harm,Rest Std U_harm\n")
    for diag in harm_std:
        o.write(str(diag)+",")
        frs = harm_pow_std[diag]
        for i in range(len(frs)):
            for j in range(len(frs[i])):
                o.write(str(frs[i][j])+",")
        o.write("\n")
    
    o.write("\n\n")
    
    #RPC
    o.write("Relative Power Contribution,\nDiagnosis,Bat Average X,Bat Average Y,Bat Average Z,Bat Average U,Kin Average X,Kin Average Y,Kin Average Z,Kin Average U,Out Average X,Out Average Y,Out Average Z,Out Average U,Rest Average X,Rest Average Y,Rest Average Z,Rest Average U\n")
    for diag in RPC_ave:
        o.write(str(diag)+",")
        frs = RPC_ave[diag]
        for i in range(len(frs)):
            for j in range(len(frs[i])):
                o.write(str(frs[i][j])+",")
        o.write("\n")
    
    o.write("\n\n")
        
    o.write("Diagnosis,Bat Std X,Bat Std Y,Bat Std Z,Bat Std U,Kin Std X,Kin Std Y,Kin Std Z,Kin Std U,Out Std X,Out Std Y,Out Std Z,Out Std U,Rest Std X,Rest Std Y,Rest Std Z,Rest Std U\n")
    for diag in RPC_std:
        o.write(str(diag)+",")
        frs = RPC_std[diag]
        for i in range(len(frs)):
            for j in range(len(frs[i])):
                o.write(str(frs[i][j])+",")
        o.write("\n")
    
    o.write("\n")
    
    #Relative Energy (RE)
    o.write("Relative Energy,\nDiagnosis,RE_bat,RE_out,RE_bat(std),RE_out(std)\n")
    for diag in RE_bat_ave:
        o.write(str(diag)+",")
        frs = RE_bat_ave[diag]
        o.write(str(frs)+",")
        frs = RE_out_ave[diag]
        o.write(str(frs)+",")
        frs = RE_bat_std[diag]
        o.write(str(frs)+",")
        frs = RE_out_std[diag]
        o.write(str(frs)+",")
        o.write("\n")
    
    o.write("\n\n")
        
    o.write("Diagnosis,Bat Std X,Bat Std Y,Bat Std Z,Bat Std U,Kin Std X,Kin Std Y,Kin Std Z,Kin Std U,Out Std X,Out Std Y,Out Std Z,Out Std U,Rest Std X,Rest Std Y,Rest Std Z,Rest Std U\n")
    for diag in RPC_std:
        o.write(str(diag)+",")
        frs = RPC_std[diag]
        for i in range(len(frs)):
            for j in range(len(frs[i])):
                o.write(str(frs[i][j])+",")
        o.write("\n")
    
    o.write("\n")
    
    #Control
    o.write("\nControl,Bat Average X_freq,Bat Average Y_freq,Bat Average Z_freq,Bat Average U_freq,Kin Average X_freq,Kin Average Y_freq,Kin Average Z_freq,Kin Average U_freq,Out Average X_freq,Out Average Y_freq,Out Average Z_freq,Out Average U_freq,Rest Average X_freq,Rest Average Y_freq,Rest Average Z_freq,Rest Average U_freq\n")
    o.write("Control,")
    frs = freq_ave_control
    for i in range(len(frs)):
        for j in range(len(frs[i])):
            o.write(str(frs[i][j])+",")
    o.write("\n")
    
    o.write("\n")
    
    o.write("\nControl,Bat Std X_freq,Bat Std Y_freq,Bat Std Z_freq,Bat Std U_freq,Kin Std X_freq,Kin Std Y_freq,Kin Std Z_freq,Kin Std U_freq,Out Std X_freq,Out Std Y_freq,Out Std Z_freq,Out Std U_freq,Rest Std X_freq,Rest Std Y_freq,Rest Std Z_freq,Rest Std U_freq\n")
    o.write("Control,")
    frs = freq_std_control
    for i in range(len(frs)):
        for j in range(len(frs[i])):
            o.write(str(frs[i][j])+",")
    o.write("\n")
    
    o.write("\n")
    
    #Write peak data:
    for diag in freq_list:
        peakX_ave = peakX_sorted_ave[diag]
        peakX_std = peakX_sorted_std[diag]
        peakX_num = peakX_sorted_num[diag]
        peakY_ave = peakY_sorted_ave[diag]
        peakY_std = peakY_sorted_std[diag]
        peakY_num = peakY_sorted_num[diag]
        peakZ_ave = peakZ_sorted_ave[diag]
        peakZ_std = peakZ_sorted_std[diag]
        peakZ_num = peakZ_sorted_num[diag]
        peakU_ave = peakU_sorted_ave[diag]
        peakU_std = peakU_sorted_std[diag]
        peakU_num = peakU_sorted_num[diag]
        o.write("\n"+ diag+" Average Peak Data,,,,,,"+diag+" Std Peak Data")
        for i in range(len(peakX_ave)):
            tremor = tremor_types[i]
            o.write("\n"+tremor+":")
            o.write("\nPeak #,num peaks X,p_startX,p_endX,p_maxX,Power at Max X,Peak Area X,p_widthX,p_startX (std),p_endX (std),p_maxX (std),Power at Max X (std),Peak Area X (std),p_widthX (std),")
            o.write("num peaks Y,p_startY,p_endY,p_maxY,Power at Max Y,Peak Area Y,p_widthY,p_startY (std),p_endY (std),p_maxY (std),Power at Max Y (std),Peak Area Y (std),p_widthY (std),")
            o.write("num peaks Z,p_startZ,p_endZ,p_maxZ,Power at Max Z,Peak Area Z,p_widthZ,p_startZ (std),p_endZ (std),p_maxZ (std),Power at Max Z (std),Peak Area Z (std),p_widthZ (std),")
            o.write("num peaks U,p_startU,p_endU,p_maxU,Power at Max U,Peak Area U,p_widthU,p_startU (std),p_endU (std),p_maxU (std),Power at Max U (std),Peak Area U (std),p_widthU (std),\n")
            
            maxNumPeaks = max(len(peakX_ave[i]),len(peakY_ave[i]),len(peakZ_ave[i]),len(peakU_ave[i])) #to determine how many rows needed for max number of peaks
            for j in range(maxNumPeaks):
                o.write(str(j+1)+",");
                
                if j >= len(peakX_ave[i]):
                    for k in range(13):
                        o.write(",")
                else:
                    o.write(str(peakX_num[i][j])+",")
                    for k in range(6):
                        o.write(str(peakX_ave[i][j][k]) + ",")
                    for k in range(6):
                        o.write(str(peakX_std[i][j][k]) + ",")
                
                if j >= len(peakY_ave[i]):
                    for k in range(13):
                        o.write(",")
                else:
                    o.write(str(peakY_num[i][j])+",")
                    for k in range(6):
                        o.write(str(peakY_ave[i][j][k]) + ",")
                    for k in range(6):
                        o.write(str(peakY_std[i][j][k]) + ",")
                
                if j >= len(peakZ_ave[i]):
                    for k in range(13):
                        o.write(",")
                else:
                    o.write(str(peakZ_num[i][j])+",")
                    for k in range(6):
                        o.write(str(peakZ_ave[i][j][k]) + ",")
                    for k in range(6):
                        o.write(str(peakZ_std[i][j][k]) + ",")
                
                #'u' (composite) peak info
                if j >= len(peakU_ave[i]):
                    for k in range(13):
                        o.write(",")
                else:
                    o.write(str(peakU_num[i][j])+",")
                    for k in range(6):
                        o.write(str(peakU_ave[i][j][k]) + ",")
                    for k in range(6):
                        o.write(str(peakU_std[i][j][k]) + ",")
                o.write("\n")
            o.write("\n")
    

#Save data to file (pickle it)
pickle.dump(peakU_sorted, open( "peakU_sorted.p", "wb" ) )
pickle.dump(RPC_list, open( "RPC_list.p", "wb") )


#peakU_delete = pickle.load(open( "peakU_sorted.p", "rb" ))





    