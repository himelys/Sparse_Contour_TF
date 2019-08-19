# Object-based time-frequency analysis

This is a python program for object-based time-frequency analysis. Program consists of different modules to extract contours of given sound either using numpy (CPU) packages for fft calculation.

## Requirements
Tested under following packages
- NumPy 1.14.0
- SciPy 1.0.0
- Opencv-python 4.1.0.25
- Skimage 0.14.3

## Example

**1. Define analysis parameter**
```python
fs = 16000 # Sampling frequency
ts = 8.0 # time scale
Theta = 1.5 # angle
ARThreshold = 80 # contour selection threshold
nWin = 2048 # width of windows in samples
nshift = 1 # hop size
noverlap = nWin - nshift # overlap size
nfft = nWin # number of FFT
```

**2. Load or create sound**

```python
import Create_Sound as CS

x,tt = CS.CreateSineWave(0.2,10.0,fs) # create sine wave (0.2 sec, 10dB SNR)
```

**3. Extract contours**

```python
import object_module as om

npObjMask, npConAreas, npLabelImg = om.CreateContourObj(x,fs,ts,nWin,nfft,noverlap,Theta,ARThreshold)
```

## Results for sine wave (0.2 secs long)
- Numpy: 0.288 sec

![Result Image](/example_result.png)

## References
1. Y. Lim, B. Shinn-Cunningham and T. J. Gardner, "Sparse Contour Representations of Sound," in IEEE Signal Processing Letters, vol. 19, no. 10, pp. 684-687, Oct. 2012. [[Link]](http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6256698&isnumber=6249741)
2. Y. Lim, B. Shinn-Cunningham and T. J. Gardner, "Stable time-frequency contours for sparse signal representation," 21st European Signal Processing Conference (EUSIPCO 2013), Marrakech, 2013, pp. 1-5. [[Link]](http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6811462&isnumber=6811387)
3. M. Aoi, K. Lepage, Y. Lim, U. T. Eden and T. J. Gardner, "An Approach to Time-Frequency Analysis With Ridges of the Continuous Chirplet Transform," in IEEE Transactions on Signal Processing, vol. 63, no. 3, pp. 699-710, Feb.1, 2015. [[Link]](http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6937207&isnumber=6994902)
