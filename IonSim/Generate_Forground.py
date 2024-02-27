#~
import numpy as np
from pygdsm import GlobalSkyModel16
from pygdsm import GlobalSkyModel
import h5py
import healpy as hp 
import matplotlib.pyplot as plt
from tqdm import tqdm
import math

def generate_smooth_forground(nu,map230):
    nu_0 = 230 #MHz
    T_cmb = 2.725 #K
    return (map230-T_cmb)*(nu/nu_0)**-2.5 + T_cmb
def generate_gaussian_signal(freq,A0, u, sigma):
#     sigma = 18.7 #MHz
#     u = 78.1     #MHz
#     A0 = -0.53    #K
    y = A0*np.exp(-(freq - u) ** 2 /(2* sigma **2))
    return np.array(y)
def generate_flat_signal(nu,A0, u, sigma,tau):
    B = (4*(nu - u)**2/sigma**2)*(np.log10(-1/tau*(np.log10((1+math.e**(-tau))/2))))
    y = A0*(1-math.e**(-tau*math.e**B))/(1-math.e**-tau)
    return np.array(y)
def show(freq,fmap):
    plt.xlabel("Freq/MHz")
    plt.ylabel("T/K")
    plt.plot(freq,np.mean(fmap,axis=1))
    plt.show()
def GSM_for(filepath, freq=np.arange(30,120,1), nside = 256,A0 = -0.53, u = 78.1, sigma = 18.7,tau = 7, signal = 'gaussian' ):
    print('Generating sky map ......')
    GSM = GlobalSkyModel(freq_unit='MHz')
    # global sky model in healpix format, NSIDE=512, in galactic coord,
    # and in unit of K
    forground = hp.ud_grade(GSM.generate(freq),nside_out = nside)
    #Convert to specified nside map
    print('Adding HI signal ......')
    if signal == 'gaussian':
        g_signal = generate_gaussian_signal(freq,A0, u, sigma)
    elif signal == 'flat':
        g_signal = generate_flat_signal(freq,A0, u, sigma,tau)
    else:
        print('Two kinds of signal: \'gaussian\' or \'flat\'')
    fmap = []
    for i in range(len(freq)):
        fmap.append(forground[i]+g_signal[i])
    hf = h5py.File(filepath, 'w')
    hf.create_dataset('map', data = fmap)
    hf.close()
    show(freq,fmap)
def GSM2016_for(filepath, freq=np.arange(30,120,1), nside = 256,sigma = 18.7,u = 78.1,A0 = -0.53 ):
    print('Generating sky map ......')
    GSM = GlobalSkyModel16(freq_unit='MHz')
    # global sky model in healpix format, NSIDE=512, in galactic coord,
    # and in unit of K
    forground = hp.ud_grade(GSM.generate(freq),nside_out = nside)
    #Convert to specified nside map
    print('Adding HI signal ......')
    g_signal = generate_gaussian_signal(freq,nside,sigma,u,A0)
    fmap = []
    for i in range(len(freq)):
        fmap.append(forground[i]+g_signal[i])
    hf = h5py.File(filepath, 'w')
    hf.create_dataset('map', data = fmap)
    hf.close()
    show(freq,fmap)
def SMOOTH_for(filepath,freq=np.arange(30,120,1),nside = 256,sigma = 18.7,u = 78.1,A0 = -0.53 ):
    print('Generating sky map ......')
    GSM = GlobalSkyModel(freq_unit='MHz')
    map230 = hp.ud_grade(GSM.generate(230),nside_out = nside)
    forground = []
    for ii in freq:
        forground.append(generate_smooth_forground(ii,map230))
    print('\n Adding HI signal ......')
    g_signal = generate_gaussian_signal(freq,nside,sigma,u,A0)
    fmap = []
    for i in range(len(freq)):
        fmap.append(forground[i]+g_signal[i])
    hf = h5py.File(filepath, 'w')
    hf.create_dataset('map', data = fmap)
    hf.close()        
    show(freq,fmap)
