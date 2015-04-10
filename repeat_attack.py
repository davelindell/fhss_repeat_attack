#!/home/lindell/local/miniconda3/envs/py27/bin/python
'''
repeat_attack.py
Tested with Python 2.7
Created on Apr 8, 2015

Version 1.0

2015-04-09 Completed port from MATLAB and initial testing

@author: lindell
'''
import os
import math
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import sys
from itertools import izip
import pickle
from struct import pack
import subprocess

# Function loadWindow
# Analyzes a chunk of measurements from the binary input
# files and determine whether there is a signal present
# If there is a signal present, the function returns
# the fsk frequencies and the start and stop times of the signal
def loadWindow(offset, window_size, begin_time, filename, fc, fs):
    file = open(filename,'r')
    file.seek(offset,0)
    # read file as int16 numpy array
    # use the SC16 Q11 samples [-2048 to 2047] (int16)
    # do window_size/2 because there are N/2 int16 values per byte
    data = np.fromfile(file, dtype=np.dtype(np.int16), count=int(window_size/2))
    data = data.astype(float)
    data = data/2048
    file.close()

    # get inphase/quadrature data
    si = data[0::2]
    sq = data[1::2]
    s = si + 1j* sq;
    
    S,F,T = mlab.specgram(
            s,window=np.hamming(128),noverlap=64,NFFT=128,Fs=fs)
    # get dB power spectral density
    P_log = 10*np.log10(np.square(abs(S)))
    # correct frequency
    F = F + fc;

    # find maximum column indices
    max_points = P_log.argmax(axis=0)
    # max_points = np.amax(P_log,axis=0)
    max_points = F[max_points]
    
    # draw stuff
#     plt.imshow(P_log,interpolation='nearest', aspect='auto',origin='lower', extent=[T[0],T[-1],F[0],F[-1]])
#     plt.plot(T,max_points, linewidth='2.0', color='yellow')
#     plt.show()
    
    # do detection scheme
    start_ind = 0
    stop_ind = 0
    found_start = 0
    subsequent_fcs = 0
    for ii, mp in enumerate(max_points):
        if abs(mp - fc) > 1 and not found_start:
            start_ind = ii+1
            found_start = 1
        elif found_start and mp == fc:
            subsequent_fcs = subsequent_fcs + 1
        elif found_start and mp != fc:
            subsequent_fcs = 0
        if found_start and subsequent_fcs < 100:
            stop_ind = ii+1
    
    stop_ind = stop_ind - subsequent_fcs
            
    if start_ind != 0 and stop_ind != 0:
        ind = range(start_ind,stop_ind)
    else:
        ind = []
    
    center_freq = np.mean(max_points[ind]);
    
    # Calculate return values
    if len(ind) > 20:
        f1 = center_freq - 250000
        f2 = center_freq + 250000
        start_time = T[ind[1]] + begin_time
        stop_time = T[ind[-1]] + begin_time
    else:
        f1 = np.nan
        f2 = np.nan
        start_time =np.nan
        stop_time = np.nan

    return (f1,f2,start_time,stop_time)
    
# Function analyzeFiles
# Handles the reading and analyzing of all measurements 
# from the bladeRF binary files. Splits all the files into
# several windows for easier processing and then saves the 
# start and stop times of each fhss signal as well as the fsk 
# frequencies.
def analyzeFiles(filenames,fs, window_size, num_bytes):
    # initialize data storage
    T = 1/fs
    window_samples = window_size/2;
    repeat_attacks = np.zeros((1000,4))
    r_i = 0;
    loop_i = 0;
    num_hops = np.zeros((len(filenames),1))
    
    for f_i, f in enumerate(filenames):
        # get center frequency from filename
        fc = float(
            f[0] + '.' + f[2] + f[3] + 'E9')
        # f_info = os.stat(f)
        # f_size = f_info.st_size
        
        # get number of samples to read (read .25 sec worth)
        # set up windows of files to read
        offset = range(0,int(num_bytes),int(window_size))
        prev_val = 0

        # loop through windows in each file
        for o_i, o in enumerate(offset):
            # print progress
            # prog_str = str('{0:.2f}%\r'.format(float(o_i)/float(len(offset))*100))
            # print('%s' % prog_str)
            sys.stdout.flush()
            loop_i += 1
            begin_time = loop_i * window_samples * T;            
            f1, f2, start_time, stop_time = \
                loadWindow(o, window_size, begin_time, f, fc, fs)
            # if a window with data in it was found
            if not np.isnan(f1):
                # check to see if we're at the start or starting a window
                if num_hops[f_i] == 0 or r_i == 0:
                    num_hops[f_i] += 1
                    repeat_attacks[r_i][0] = f1
                    repeat_attacks[r_i][1] = f2
                    repeat_attacks[r_i][2] = start_time
                    repeat_attacks[r_i][3] = stop_time
                    r_i += 1
                # if we don't have a signal across multiple windows
                elif np.isnan(prev_val):
                    num_hops[f_i] += 1
                    repeat_attacks[r_i][0] = f1
                    repeat_attacks[r_i][1] = f2
                    repeat_attacks[r_i][2] = start_time
                    repeat_attacks[r_i][3] = stop_time
                    r_i += 1
                # if the signal stretches across windows, give true stop time
                else:
                    repeat_attacks[r_i][3] = stop_time
            prev_val = f1
    # end for(filenames)
    return (repeat_attacks, num_bytes, num_hops)

# Function plotAttacks
# Plots the fhss frequency hops on a frequency vs time
# axis. Matplotlib has some sort of bug where the plotted hops
# won't show up if they are plotted as a line ('-'). So they are
# plotted as the less visually appealing '.'.
def plotAttacks(repeat_attacks, frame_end, color):
    # plot the repeat_attacks
    for r_i, r in enumerate(repeat_attacks):
        # if is all zeros
        if not repeat_attacks[r_i].any():
            break
        tmp = repeat_attacks[r_i]
        f1 = tmp[0]
        f2 = tmp[1]
        start_time = tmp[2]
        stop_time = tmp[3]
        
        plt.plot([start_time, stop_time],[f1,f1],'.',linewidth=2.0, color=color)
        plt.plot([start_time, stop_time],[f2,f2],'.', linewidth=2.0, color=color)
        plt.axis([0, frame_end[-1], 2.4E9, 2.49E9 ])                
    # plot the frame endings
    for f_end in frame_end:
        plt.plot([f_end, f_end],[2.4E9, 2.49E9], linewidth=1.0, color='black')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    # plt.savefig('freq_hops.png',dpi=300)

# Function findPeriod
# Determine the repeat period of the fhss 
# hops based on the hops in the first frequency recorded.
# Note that BladeRF recording on the first frequency (2.41 GHz)
# must span a full repeat period for this program to work
def findPeriod(repeat_attacks, num_hops):
    repeat_period = 0
    num_found_repeats = 0
    hops = repeat_attacks[0:int(num_hops[1])]
    for ii, h_i in enumerate(hops):
        test_f1 = h_i[0]
        test_t1 = h_i[2]
        for jj, h_j in enumerate(hops):
            test_f2 = h_j[0]
            test_t2 = h_j[2]
            if ii != jj and np.abs(test_f2 - test_f1 ) < 1E6:
                repeat_period = repeat_period + abs(test_t2 - test_t1)
                num_found_repeats = num_found_repeats + 1
                break
            
    repeat_period = repeat_period / num_found_repeats
    return repeat_period

# Function correctHops
# This function corrects for the random time gap between measurements
# from different center frequencies. The bladeRF takes some random amount
# of time to switch the center frequency and begin recording again. The switch time
# is non deterministic in part because of scheduling restrictions on the host machine, 
# etc. Basically the frequency hops from each window are corrected so that they occur in 
# the correct time frame relative to the hops recorded on the first channel.
def correctHops(repeat_attacks, repeat_period, num_hops, frame_end):
    stop_indices = np.cumsum(num_hops) - 1
    start_indices = np.hstack([[0], stop_indices+1])
    
    # look for repeated frequencies across windows
    correction = np.zeros((len(start_indices)-1,1))
    for ii, s_i in enumerate(izip(start_indices, stop_indices)):
        if ii == len(start_indices)-2:
            break
        start_i, stop_i = s_i
        hops1 = repeat_attacks[start_i:stop_i+1].copy()
        hops2 = repeat_attacks[start_indices[ii+1]:stop_indices[ii+1]+1].copy()
          
        # get rid of all but last repeat in 1st window
        # find and mark
        for h1_a in hops1:
            for h1_b in hops1:
                if abs(h1_a[0] - h1_b[0]) < 1E6:
                    if h1_a[2] < h1_b[2]:
                        h1_a[2] = 0
        # remove
#         plotAttacks(repeat_attacks, frame_end, 'blue')

        jj = 0;
        while jj <= len(hops1)-1:
            if hops1[jj][2] == 0:
                
                # debug plot
#                 tmp = hops1[jj]
#                 f1 = tmp[0]
#                 f2 = tmp[1]
#                 start_time = tmp[2]
#                 stop_time = tmp[3]
#                 plt.plot([stop_time-.005, stop_time], [f1, f1],'.',linewidth=2,color='red')
#                 plt.plot([stop_time-.005, stop_time], [f2, f2],'.',linewidth=2,color='red')
#                 plt.axis([0, frame_end[-1], 2.4E9, 2.49E9 ])
                # end debug
                hops1 = np.delete(hops1,jj,0)
            else:
                jj = jj + 1
         
#         plt.show()
         
        # get rid of all but first repeat in 2nd window
        # find and mark
        for h2_a in hops2:
            for h2_b in hops2:
                if abs(h2_a[0] - h2_b[0]) < 1E6:
                    if h2_a[2] > h2_b[2]:
                        h2_a[2] = 1E6
                        
#         plotAttacks(repeat_attacks, frame_end, 'blue')
        
        # remove
        jj = 0;
        while jj <= len(hops2)-1:
            if hops2[jj][2] == 1E6:
                
                # debug plot
#                 tmp = hops2[jj]
#                 f1 = tmp[0]
#                 f2 = tmp[1]
#                 start_time = tmp[2]
#                 stop_time = tmp[3]
#                 plt.plot([stop_time-.005, stop_time], [f1, f1],'.',linewidth=2,color='red')
#                 plt.plot([stop_time-.005, stop_time], [f2, f2],'.',linewidth=2,color='red')
#                 plt.axis([0, frame_end[-1], 2.4E9, 2.49E9 ])
                # end debug
                
                hops2 = np.delete(hops2,jj,0)
            else:
                jj = jj + 1
            
#         plt.show()
            
        found = 0
        num_corrections = 0
        correction_candidates = np.zeros((10,2))
        for h1 in hops1:
            for h2 in hops2:
                if abs(h1[0] - h2[0]) < 1E6:
                    # check to see if it's not really the first in the frame
                    if h2[2] - frame_end[ii] < .9*repeat_period:
                        separation = h2[2] - h1[2]
                        if not found:
                            found = 1
                            correction_candidates[0][0] = repeat_period - separation
                            correction_candidates[0][1] = 1
                            num_corrections += 1
                        else:
                            ind = np.where(np.abs(
                              correction_candidates[:,0] - (repeat_period - separation)) < .001)[0]
                            if ind.size > 0:
                                ind = ind[0]                                    
                                correction_candidates[ind][1] += 1
                            else:
                                correction_candidates[num_corrections][0] = repeat_period - separation
                                correction_candidates[num_corrections][1] = 1
                                num_corrections = num_corrections + 1

        # find most popular correction value
        correction[ii] = correction_candidates[correction_candidates[:,1].argsort()][-1][0]
         
        if not found:
            raise RuntimeError('No overlapping frequencies from channel ' + \
               '{0} to channel {1}'.format(ii,ii+1))

#         plotAttacks(repeat_attacks, frame_end, 'red')
        # apply correction
        for jj in range(0,num_hops[ii+1]):
            start_time = repeat_attacks[start_indices[ii+1]+jj][2]
            stop_time = repeat_attacks[start_indices[ii+1]+jj][3]
            start_time = start_time + correction[ii]
            stop_time = stop_time + correction[ii]
            repeat_attacks[start_indices[ii+1]+jj][2] = start_time
            repeat_attacks[start_indices[ii+1]+jj][3] = stop_time
   
#     plotAttacks(repeat_attacks, frame_end, 'blue')
#     plt.show()
        
    # normalize hops so first one starts at t=0
    offset = repeat_attacks[1][3]
    for ra in repeat_attacks:
        if not ra.any():
            break 
        ra[2] = ra[2] - offset
        ra[3] = ra[3] - offset
     
    return repeat_attacks

# Function orderHops
# this function takes hops from all windows and places them in the correct
# positions in the first repeat period of the recorded signals. 
# In this way, a synthetic recording of all hops across the entire frequency
# range is calculated.
def orderHops(repeat_attacks, repeat_period):
    # find first occurrence of all frequencies and then bring them into the
    # correct position in the first period
    seen_frequencies = []
    single_period_hops = np.zeros((len(repeat_attacks),4))
    sph_i = 0
    for ra in repeat_attacks:
        if not ra.any():
            break 
        # check to see if we've seen the frequency already
        test_f1 = ra[1];
        seen = 0;
        for sf in seen_frequencies:
            if np.abs(test_f1 - sf) < 1E6:
                seen = 1;
                break;
        # if not seen, add to seen list, calculate time location, add to list
        # of single period hops
        if not seen:
            seen_frequencies.append(test_f1)
            f1 = ra[0]
            f2 = ra[1]
            start_time = np.mod(ra[2], repeat_period)
            stop_time = np.mod(ra[3], repeat_period)
            single_period_hops[sph_i] = [f1, f2, start_time, stop_time]
            sph_i = sph_i + 1
            
    # get rid of empty entries
    ii = 0;
    while ii <= len(single_period_hops)-1:
        if not all(single_period_hops[ii]):
            single_period_hops = np.delete(single_period_hops,ii,0) 
        else:
            ii = ii + 1;
    return single_period_hops

# Function writeAttack
# Takes the ordered frequency hops and writes a binary file with the
# fsk signal in time on each of the hops. The hop repeat period is divided
# by the number of hops found, and each hop is set to be long enough to 
# fill the entire period so that at least one frequency is sounding at any
# given time. The program records the hop pause length, so the code could
# be modified to pulse each hop for the time recorded initially.
# The function outputs a binary file called 'output.bin'
# Note that the output file is in the S16 Q11 format as required by the bladeRF
def writeAttack(single_period_hops, baud_rate, repeat_period, fs):
    # create bitstream pattern
    pulse_length = np.mean(single_period_hops[:,3] - single_period_hops[:,2])
    channel_length = repeat_period / len(single_period_hops)
    
    # figure out how many samples per channel length
    T = 1/fs
    num_samples = channel_length / T
    bit_period = 1/baud_rate
    num_bits = math.floor(channel_length / bit_period)
    samples_per_bit_period = int(fs*bit_period)
    
    # make repeat signal
    bit_pattern = np.tile([0, 1],math.floor(num_bits/2))
    fid = open('output.bin','w+');

    # write file
    for ii, sph in enumerate(single_period_hops):
        freq1 = single_period_hops[ii][0];
        freq0 = single_period_hops[ii][1];
        w1 = 2*math.pi*freq1
        w0 = 2*math.pi*freq0
        for b in bit_pattern:
            # preamble pattern is 01010101...
            # generate in phase/quadrature signal
            # use the SC16 Q11 samples [-2048 to 2047] (int16) little endian
            if b == 1:
                carrier = w1
            else:
                carrier = w0
                si = (2047.0*np.cos(carrier*np.linspace(0,bit_period,samples_per_bit_period+1))).astype(np.int16)
                sq = (2047.0*np.sin(carrier*np.linspace(0,bit_period,samples_per_bit_period+1))).astype(np.int16)
                s = np.zeros((len(si)+len(sq)))
                s[0::2] = si
                s[1::2] = sq
                fid.write(pack('{}h'.format(len(s)), *s));            
    fid.close()

# Function repeatAttack
# This is the main program driver
# Here are the main assumptions for this program
# 1. The fhss pattern repeats in less than .25 s
#	- This is easily changed in the below constants
# 2. Baud rate = 115200
# 3. The fhss pattern is in the 2.4 GHz ISM band
# 4. There is at least ~.8 ms gap between hops (so that only 1 signal is recorded per window)
# 5. All frequencies in the hop pattern are visited exactly once before the pattern repeats
# 6. The bladeRF scripts to record and transmit are named 'record.blade' and 'tx.blade'
# 7. The output binary files are in the format '2_41.bin'. The center freq. is extracted from the filename
# I've left in most of the debug comments for now, but the program should work as is.
def repeatAttack():
    # take care of some constants
    fs = 38E6
    T = 1/fs
    baud_rate = 115200.0; #230400?
    filenames = ['2_41.bin', '2_42.bin','2_43.bin', \
    '2_44.bin','2_45.bin','2_46.bin', \
    '2_47.bin']
    window_size = int(.025E6);
    num_bytes = math.floor(fs/2)
    frame_end = num_bytes/2 * T * np.arange(1,len(filenames)+1)
    
     # record on bladerf
     bash_command = 'bladeRF-cli -s record.blade'
     process = subprocess.Popen(bash_command.split(), stdout=subprocess.PIPE)
     output = process.communicate()[0]
     process.wait()

    # analyze the input files
     repeat_attacks, num_bytes, num_hops = analyzeFiles(filenames, fs, window_size, num_bytes)

#     # save the data
#     with open('objs.pickle','w') as f:
#         pickle.dump([repeat_attacks, num_bytes, num_hops],f)
#    # load the data
#    with open('objs.pickle') as f:
#        repeat_attacks, num_bytes, num_hops = pickle.load(f)
        
    # plot (if you want-- also saves the image)
    # plotAttacks(repeat_attacks, frame_end, 'blue')
    
    # get repeat period of signal
    repeat_period = findPeriod(repeat_attacks, num_hops)
    
    # apply shift correction to the captures to account for retrieval gap
    repeat_attacks = correctHops(repeat_attacks, repeat_period, num_hops, frame_end)
    
    # put all the hops into one period
    single_period_hops = orderHops(repeat_attacks, repeat_period)

    # sort the hops by time
    single_period_hops = single_period_hops[single_period_hops[:,2].argsort()]
    
#     plotAttacks(single_period_hops, frame_end, 'blue')
#     plt.show()
    
    # write out the binary file
    writeAttack(single_period_hops, baud_rate, repeat_period, fs)

     bash_command = 'bladeRF-cli -s tx.blade'
     process = subprocess.Popen(bash_command.split(), stdout=subprocess.PIPE)
     output = process.communicate()[0]
     process.wait()

if __name__ == '__main__':
    repeatAttack()
    pass
