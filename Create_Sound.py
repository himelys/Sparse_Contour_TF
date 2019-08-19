
import numpy as np
import math
from scipy import signal
import matplotlib.pyplot as plt

def CreateModelSound(D,fs): # This is just for complex sound example

    Fend = 4000
    SampleLen = int(fs*D)
    sampleindex = np.linspace(0,SampleLen-1,SampleLen)
    t = sampleindex/fs
    y = signal.chirp(t,0,D,Fend) + signal.chirp(t,500,D,Fend+500)
    stdwin = (len(y)-1)/(2*5)
    gwin = signal.gaussian(len(y),stdwin)

    chirpsignal = np.multiply(y,gwin)

    sinusoid = 0.5*(np.sin(2*math.pi*4000*t) + np.sin(2*math.pi*4200*t))
    sound = chirpsignal + sinusoid + 0.00001*np.random.rand(len(sinusoid))
    sound[3000] = 20
    sound[3045] = 20

    return sound, t

def CreateSineWave(D,SNR,fs):

    SampleLen = int(fs*D)
    sampleindex = np.linspace(0,SampleLen-1,SampleLen)
    tt = sampleindex/fs;
    if SNR:
        clean_sound = np.sin(2*math.pi*1000*tt) + np.sin(2*math.pi*2000*tt) + np.sin(2*math.pi*3000*tt)
        amp_cl_sound = np.max(clean_sound)
        target_amp_noise = amp_cl_sound * 10**(-SNR/10)
        rnd_noise = np.random.rand(SampleLen)
        noise = target_amp_noise/np.max(rnd_noise)*rnd_noise
        sound = clean_sound + noise
    else:
        sound = np.sin(2*math.pi*1000*tt) + np.sin(2*math.pi*2000*tt) + np.sin(2*math.pi*3000*tt) + 0.00001*np.random.rand(SampleLen)

    return sound, tt

def CreateWhiteNoise(D,fs):

    SampleLen = int(fs*D)
    sampleindex = np.linspace(0,SampleLen-1,SampleLen)
    tt = sampleindex/fs
    sound = np.random.rand(SampleLen)
    return sound, tt

if __name__ == '__main__':

    y,t = CreateSineWave(0.1,16000)

    plt.figure()
    plt.plot(t,y)
    plt.show()
