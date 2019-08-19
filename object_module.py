
import numpy as np
import math
from scipy import signal
import matplotlib.pyplot as plt
from skimage import measure
import time
import cv2
import Create_Sound as CS
from skimage.filters import gaussian
from skimage.morphology import disk

def local_minima(array2d):
    return ((array2d <= np.roll(array2d,  1, 0)) &
            (array2d <= np.roll(array2d, -1, 0)) &
            (array2d <= np.roll(array2d,  1, 1)) &
            (array2d <= np.roll(array2d, -1, 1)))

def ConvWindow(ts,fs,Nfft,Nshift):

    sigmaF = 1/(ts/1000) # Frequency domain bandwidth
    delF = fs/(Nfft/2+1) # Frequency index
    sigmaFP = sigmaF/delF
    dFP = int(np.ceil(sigmaFP)) # Frequency bandwidth in number of pixels
    tFP = int(np.ceil(((ts/1000)*fs/Nshift)/2))

    Cwin = np.ones([dFP,tFP])

    return Cwin

def GaborWindow(ts,fs,nfft):

    t_g = np.linspace(-nfft/2+0.5,nfft/2-0.5,nfft)

    # Gaussian window
    tScale = (ts/1000)*fs
    w = np.exp(-(t_g ** 2/tScale**2))

    return w

def dGaborWindow(ts,fs,nfft):

    t_g = np.linspace(-nfft/2+0.5,nfft/2-0.5,nfft)

    tScale = (ts/1000)*fs
    w = GaborWindow(ts,fs,nfft)

    # derivative of Gaussian window
    dw = -2*np.multiply(w,-t_g/tScale**2)

    return dw

def GaborTransform(x,fs,ts,nWin,nfft,noverlap):

    w = GaborWindow(ts,fs,nfft)
    # Short-Time Fourier Transform
    # gaussian windowed spectrogram, Eq. 1 in the paper
    f, t, Xs = signal.spectrogram(x, fs, w, nWin, noverlap, nfft, mode='complex')
    Xs = Xs + np.finfo(float).eps
    return Xs, f, t

def dGaborTransform(x,fs,ts,nWin,nfft,noverlap):

    dw = dGaborWindow(ts,fs,nfft)
    # Short-Time Fourier Transform
    # derivative of gaussian windowed spectrogram, Eq. 2 in the paper
    f, t, dXs = signal.spectrogram(x, fs, dw, nWin, noverlap, nfft, mode='complex')
    dXs = dXs + np.finfo(float).eps
    return dXs, f, t

def ReassignmentVec_Calc(Xs,dXs,theta):

    # displacement according to the reassignment algorithm
    dX = np.divide(dXs,Xs)/(2*math.pi)
    Xs = None
    dXs = None

    # A quick approximation to the contour detection
    dv = dX*complex(np.cos(theta),np.sin(theta))
    rsXs = dv.imag

    return rsXs

def CreateContourObj(x,fs,ts,nWin,nfft,noverlap,theta,ARThreshold):

    # Short-Time Fourier Transform
    # gaussian windowed spectrogram, Eq. 1 in the paper
    #t0 = time.time()
    Xs, _, _ = GaborTransform(x,fs,ts,nWin,nfft,noverlap)
    #print "GaborTranform:%1.5f" % (time.time() - t0)
    #abs_X = np.abs(Xs)

    # derivative of gaussian windowed spectrogram, Eq. 2 in the paper
    #t0 = time.time()
    dXs, _, _ = dGaborTransform(x,fs,ts,nWin,nfft,noverlap)
    #print "DGaborTranform:%1.5f" % (time.time() - t0)

    # Caculate Reassignment vector
    # t0 = time.time()
    ReAV = ReassignmentVec_Calc(Xs,dXs,theta)
    # print "Reassignment:%1.5f" % (time.time() - t0)

    LocalMinMask = local_minima(ReAV)
    medMask = gaussian(LocalMinMask,sigma=0.2)

    # plt.figure()
    # ax1 = plt.subplot(121)
    # plt.imshow(np.log10(ReAV+10),cmap='jet',origin='lower',aspect='auto')
    # plt.subplot(122, sharex=ax1, sharey=ax1)
    # plt.imshow(LocalMinMask, aspect='auto', origin='lower', cmap=plt.cm.gray, interpolation='nearest')
    # plt.show()

    # fig, ax = plt.subplots()
    # im = ax.imshow(medMask>0, aspect='auto', origin='lower', cmap=plt.cm.gray, interpolation='nearest')
    # CS = ax.contour(ReAV, 0, origin='lower', linewidths=2)
    # plt.show()

    ConMask = np.zeros(np.shape(ReAV))

    # plt.figure()
    # plt.imshow(medMask>0, aspect='auto', origin='lower', cmap=plt.cm.gray, interpolation='nearest')
    CS = plt.contour(ReAV, 0)
    for path in CS.collections[0].get_paths():
        v = path.vertices
        x = np.rint(v[:,0])
        y = np.rint(v[:,1])
        ConMask[y.astype(int),x.astype(int)] = 1
    #     plt.plot(x,y)
    # plt.show()

    subMask = ConMask-(medMask>0)
    BW = subMask>0
    BW = np.array(BW,dtype=np.uint8)
    ret, ApprConObj = cv2.connectedComponents(BW)

    # plt.figure()
    # ax1 = plt.subplot(131)
    # plt.imshow(ConMask, aspect='auto', origin='lower', cmap=plt.cm.gray, interpolation='nearest')
    # plt.subplot(132,sharex=ax1, sharey=ax1)
    # plt.imshow(subMask>0, aspect='auto', origin='lower', cmap=plt.cm.gray, interpolation='nearest')
    # plt.subplot(133,sharex=ax1, sharey=ax1)
    # plt.imshow(ApprConObj, aspect='auto', origin='lower', cmap='jet', interpolation='nearest')
    # plt.show()

    # t0 = time.time()
    props = measure.regionprops(ApprConObj)
    contour_area = [ele.area for ele in props]
    # print "Contour Measure:%1.5f" % (time.time() - t0)

    # Filtering out short contours
    percentile_area = np.percentile(contour_area,ARThreshold) # Threshold area
    item_index = np.where(contour_area>percentile_area)
    contour_area = None

    # t0 = time.time()
    MaskImg = np.zeros(Xs.shape)
    for index in item_index[0]:
        row_col = props[index].coords
        rows = row_col[:,0]
        cols = row_col[:,1]
        MaskImg[rows,cols] = 1
    # print "Mask:%1.5f" % (time.time() - t0)
    #sparse_Mask = coo_matrix(MaskImg)
    #sparse_label_img = coo_matrix(label_img)

    return MaskImg, ApprConObj, percentile_area

if __name__ == '__main__':

    Simfs = 16000
    Simts = 3.0
    SimTheta = 2.35
    ARThreshold = 80
    nWin = 2048
    nshift = 1
    noverlap = nWin - nshift
    nfft = nWin

    SimSound,tt = CS.CreateSineWave(0.2,50,Simfs)
    Xs, f, t = GaborTransform(SimSound,Simfs,Simts,nWin,nfft,noverlap)
    print(np.shape(Xs))

    t0 = time.time()
    ObjMask, ObjLabel, _ = CreateContourObj(SimSound,Simfs,Simts,nWin,nfft,noverlap,SimTheta,ARThreshold)
    print("Total time took: %1.5f" % (time.time() - t0))

    plt.figure
    plt.subplot(141)
    plt.plot(tt,SimSound)
    ax1 = plt.subplot(142)
    plt.imshow(np.abs(Xs),cmap='jet',origin='lower',aspect='auto')
    plt.subplot(143,sharex=ax1,sharey=ax1)
    plt.imshow(ObjMask,cmap='binary',origin='lower',aspect='auto')
    plt.subplot(144,sharex=ax1,sharey=ax1)
    plt.imshow(ObjLabel,cmap='jet',origin='lower',aspect='auto')
    plt.show()
