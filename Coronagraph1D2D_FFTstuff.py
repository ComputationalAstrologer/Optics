# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 09:17:42 2023
@author: rfrazin

This does some simple 1D simulations of coragraphs for the
purposes of understanding 2D simulations in LightTrans

"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import fft 

l = 513  #array length
w = 14  # 1/2 width of aperture
b = 35  # 1/2 width of blocker
s = np.zeros((l,))
s[:w+1] = 1.; s[-w:] = 1.
s1 = fft.fft(s)
s2 = 1.0*s1;  s2[:b+1] = 0.; s2[-b:] = 0.
s3 = fft.fftshift(fft.fft(s2))

plt.figure(); plt.plot(np.abs(s1),'ko-',np.abs(s2),'rx:');
plt.figure(); plt.plot(np.abs(s3),'ko-')

v = np.zeros((l,l))
v[:w+1,:w+1]=1.;  v[-w:,-w:]=1.; v[-w:,:w+1] = 1; v[:w+1,-w:] = 1.
v1 = fft.fft2(v); v2 = 1.0*v1;  
v2[:b+1,:b+1] = 0.; v2[-b:,-b:] = 0.; v2[:b+1,-b:] = 0.; v2[-b:,:b+1] = 0.
v3 = fft.fftshift(fft.fft2(v2))


plt.figure(); plt.imshow(fft.fftshift(np.real(v2)),origin='lower',cmap='seismic'); plt.colorbar();
plt.title('Re[masked field]')
plt.figure(); plt.imshow(fft.fftshift(np.abs(v2)),origin='lower',cmap='seismic'); plt.colorbar();
plt.title('|masked field|')
plt.figure(); plt.imshow(np.real(v3),origin='lower',cmap='seismic'); plt.colorbar();
plt.title('Re[fft(masked field)]')
plt.figure(); plt.imshow(np.abs(v3),origin='lower',cmap='seismic'); plt.colorbar();
plt.title('|fft(masked field)|')


