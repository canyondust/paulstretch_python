#!/usr/bin/env python
#
# Paul's Extreme Sound Stretch (Paulstretch) - Python version
#
# by Nasca Octavian PAUL, Targu Mures, Romania
# http://www.paulnasca.com/
#
# http://hypermammut.sourceforge.net/paulstretch/
#
# this file is released under Public Domain
#
#


import sys
from numpy import *
import scipy.io.wavfile
import wave
import soundfile as sf
from optparse import OptionParser
from datetime import datetime

def print_inputfile_stats(filename):
    try:
        #print('Loading file: {!s}'.format(filename))
        wavedata, samplerate = sf.read(filename)
        print(sf.info(filename), True)
        print('frames: {}'.format(len(wavedata)))
    except:
        print('Error loading file: {!s} {!r}'.format(filename, sys.exc_info()[0]))
        return None

def load_wav(filename, start_frame, end_frame, reverse_input):
    try:
        print('\n#Input\n')
        print(sf.info(filename), True)
        wavedata, samplerate = sf.read(filename, start=start_frame, stop=end_frame)
        print('frames read: {}'.format(len(wavedata)))
        print('start_frame: {}'.format(start_frame))
        print('end_frame: {}'.format(end_frame or len(wavedata)))
        
        #print('Samplerate: {} Hz'.format(samplerate))

        print('Correcting input data layout: {} frames'.format(len(wavedata)))

        ar = [[], []]
        for a in wavedata:
            ar[0].append(a[0])
            ar[1].append(a[1])

        if reverse_input:
            print('Reverse input file')
            ar[0] = ar[0][::-1]
            ar[1] = ar[1][::-1]

        wavedata = array(ar)

        print('{} frames corrected'.format(len(wavedata[0])))

        return (samplerate,wavedata)
    except:
        print('Error loading file: {!s} {!r}'.format(filename, sys.exc_info()[0]))
        return None

def optimize_windowsize(n):
    orig_n = n
    while True:
        n = orig_n
        while (n%2) == 0:
            n /= 2
        while (n%3) == 0:
            n /= 3
        while (n%5) == 0:
            n /= 5

        if n < 2:
            break
        orig_n += 1
    return orig_n

def paulstretch(samplerate, smp, stretch, windowsize_seconds, outfilename):
    nchannels = smp.shape[0]

    print('\n#Streching\n')

    print('output file: {!s}\nsamplerate: {} Hz\nchannels: {}\n'.format(outfilename, samplerate, nchannels))
    outfile = sf.SoundFile(outfilename, 'w', samplerate, nchannels)

    #make sure that windowsize is even and larger than 16
    windowsize = int(windowsize_seconds*samplerate)
    if windowsize < 16:
        windowsize = 16
    windowsize = optimize_windowsize(windowsize)
    windowsize = int(windowsize/2) * 2
    half_windowsize = int(windowsize/2)

    #correct the end of the smp
    nsamples = smp.shape[1]

    end_size = int(samplerate*0.05)
    if end_size < 16:
        end_size = 16

    smp[:,nsamples-end_size:nsamples] *= linspace(1, 0, end_size)

    
    #compute the displacement inside the input file
    start_pos = 0.0
    displace_pos = (windowsize*0.5) / stretch

    #create Window window
#    window=0.5-cos(arange(windowsize,dtype='float')*2.0*pi/(windowsize-1))*0.5

    window = pow(1.0-pow(linspace(-1.0, 1.0, windowsize), 2.0), 1.25)

    old_windowed_buf = zeros((2, windowsize))
#    hinv_sqrt2=(1+sqrt(0.5))*0.5
#    hinv_buf=2.0*(hinv_sqrt2-(1.0-hinv_sqrt2)*cos(arange(half_windowsize,dtype='float')*2.0*pi/half_windowsize))/hinv_sqrt2
    
    start_time = datetime.now()
    
    while True:
        #get the windowed buffer
        istart_pos = int(floor(start_pos))
        buf = smp[:,istart_pos:istart_pos+windowsize]
        if buf.shape[1] < windowsize:
            buf=append(buf, zeros((2, windowsize-buf.shape[1])), 1)
        buf = buf * window
    
        #get the amplitudes of the frequency components and discard the phases
        freqs = abs(fft.rfft(buf))

        #randomize the phases by multiplication with a random complex number with modulus=1
        ph = random.uniform(0,2*pi, (nchannels,freqs.shape[1])) * 1j
        freqs = freqs * exp(ph)

        #do the inverse FFT 
        buf = fft.irfft(freqs)

        #window again the output buffer
        buf *= window

        #overlap-add the output
        output = buf[:,0:half_windowsize] + old_windowed_buf[:,half_windowsize:windowsize]
        old_windowed_buf = buf

        #remove the resulted amplitude modulation
        #update: there is no need to the new windowing function
        #output*=hinv_buf
        
        #clamp the values to -1..1 
        output[output>1.0] = 1.0
        output[output<-1.0] = -1.0

        
        #write the output to wav file
        d = int16(output.ravel(1)*32767.0)
        d = array_split(d, len(d)/2)

        outfile.write(d)

        start_pos += displace_pos
        if start_pos >= nsamples:
            print("100 %")
            break
        
        sys.stdout.write('{} % \r'.format(int(100.0*start_pos/nsamples)))
        sys.stdout.flush()

    outfile.close()
    print('Streched in: {}'.format(datetime.now() - start_time))
    print('\n#Output\n')
    print(sf.info(outfilename), True)

########################################
print("Paul's Extreme Sound Stretch (Paulstretch) - Python version 20141220")
print("by Nasca Octavian PAUL, Targu Mures, Romania\n")
parser = OptionParser(usage="usage: %prog [options] input_wav output_wav")
parser.add_option("-s", "--stretch", dest="stretch", help="stretch amount (1.0 = no stretch)", type="float", default=8.0)
parser.add_option("-w", "--window_size", dest="window_size", help="window size (seconds)", type="float", default=0.25)
parser.add_option("-t", "--start_frame", dest="start_frame", help="Start read on frame", type="int", default=0)
parser.add_option("-e", "--end_frame", dest="end_frame", help="End read on frame", type="int", default=None)
parser.add_option("-r", "--reverse", action="store_true", dest="reverse_input", help="Reverse input file", default=False)
parser.add_option("-i", "--input_file_stat", action="store_true", dest="input_file_stat", help="Print inputfile stat and then exit", default=False)
parser.add_option("-l", "--list_supported_types", action="store_true", dest="list_supported_types", help="List all supported input file types and then exit", default=False)

(options, args) = parser.parse_args()

if options.list_supported_types:
    print('Supported file types:\n')
    for file_type, desc in sf.available_formats().items():
        print('{} - {}'.format(file_type, desc))
    sys.exit(0)

if len(args) > 0 and options.input_file_stat:
    print_inputfile_stats(args[0])
    sys.exit(0) 

if (len(args) < 2) or (options.stretch <= 0.0) or (options.window_size <= 0.001):
    print("Error in command line parameters. Run this program with --help for help.")
    sys.exit(1)

print('stretch amount = {}'.format(options.stretch))
print('window size = {} seconds'.format(options.window_size))
(samplerate, smp) = load_wav(args[0], options.start_frame, options.end_frame, options.reverse_input)

paulstretch(samplerate, smp, options.stretch, options.window_size, args[1])



