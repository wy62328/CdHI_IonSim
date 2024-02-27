from scipy.optimize import curve_fit
import math
import os,sys
import matplotlib.pyplot as plt
import h5py
import numpy as np
import warnings
import logging
logging.getLogger().setLevel(logging.INFO)
warnings.filterwarnings("ignore")

def func(nu,C,a0,a1,a2,a3,a4,a5,A0,u,sigma):
    alpha = a0 + a1*np.log10(nu)+a2*(np.log10(nu))**2+a3*(np.log10(nu))**3+a4*(np.log10(nu))**4+a5*(np.log10(nu))**5
    gaussian = A0*np.exp(-(nu-u*1e6)**2/(2*(sigma*1e6)**2))
    return  10**alpha + gaussian+C
def func5(nu,C ,a0,a1,a2,a3,a4,a5):
    alpha = a0 + a1*np.log10(nu)+a2*(np.log10(nu))**2+a3*(np.log10(nu))**3+a4*(np.log10(nu))**4+a5*(np.log10(nu))**5
    return  10**alpha+C
def func4(nu,C ,a0,a1,a2,a3,a4):
    alpha = a0 + a1*np.log10(nu)+a2*(np.log10(nu))**2+a3*(np.log10(nu))**3+a4*(np.log10(nu))**4
    return  10**alpha+C
def func3(nu,C ,a0,a1,a2,a3):
    alpha = a0 + a1*np.log10(nu)+a2*(np.log10(nu))**2+a3*(np.log10(nu))**3
    return  10**alpha+C
def func2(nu,C ,a0,a1,a2,):
    alpha = a0 + a1*np.log10(nu)+a2*(np.log10(nu))**2
    return  10**alpha+C
def gaussian_func(nu,A,u,sigma):
    return A*np.exp(-(nu-u*1e6)**2/(2*(sigma*1e6)**2))
def readdata(filepath,nulist):
    data_test = np.loadtxt(filepath, dtype=np.str_, delimiter=' ')
    data_test = np.delete(data_test,len(nulist),axis = 1)
    data_test_f = data_test.astype(float)
    freqdata = np.mean(data_test_f, axis=0)
    print(freqdata.shape)
    return freqdata

#      C,a0,a1,a2,a3,a4,a5,A0,u,sigma
init1 = [ 2.725,12, -1.8,0.,-0.0,0,0,-0.53,78.1,18.7]
pmin1 = [0,0,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-0.6,75,18]
pmax1 = [5,15,np.inf,np.inf,np.inf,np.inf,np.inf,-0.5,80,19]

class Fit_polynomial_gaussian():
    def __init__(self,freq,freqdata,funci = func,initt =init1,pmin = pmin1, pmax = pmax1 ):
        self.freq = freq
        self.initt = initt
        self.pmin = pmin
        self.pmax = pmax
        self.freqdata = freqdata
        self.funci = funci
    def run_fit(self):
        print('Running fitting under parameter: \n init:',self.initt,
                     '\n pmin:',self.pmin,
                     '\n pmax:',self.pmax)
        logging.info(' Fitting using Trust Region Reflective, may takes 5~20min ......')
        freq,freqdata = self.freq,self.freqdata
        sigma=np.array([1e-5]*len(freq))
        popt, pcov = curve_fit(self.funci, freq, freqdata,sigma=[1e-18]*len(freq),
                              absolute_sigma=True,p0 = self.initt,
                               bounds=(self.pmin,self.pmax),
                               maxfev=int(1e20))
        print('The fitted parameters as followed:\n',popt)

        fit_data = self.funci(freq,*popt)
        residual = freqdata - fit_data
        plt.xlabel("Freq/Hz")
        plt.ylabel("T/K")
        plt.plot(freq,residual)
        plt.title("Residual of polynomial+gaussian fitting")
        plt.show()
        self.popt =popt
        self.residual = residual
    def check_gauss(self):
        popt = self.popt
        freq = self.freq
        fitted_gauss = gaussian_func(freq,popt[-3],popt[-2],popt[-1])
#         fig=figure(figsize=(8,6))
        plt.plot(freq,fitted_gauss,'--',label = 'fit')
        plt.plot(freq,gaussian_func(freq,-0.53,78.1,18.7),label='origion')
        plt.legend(fontsize='x-large')
        plt.xlabel("Freq/Hz")
        plt.ylabel("T/K")
        plt.title('Fitted Gauss',size = 20)
        self.fitted_gauss = fitted_gauss
        
init2= [ 2.725,12, -1.8,0.,-0.0, 0,0]
pmin2= [0,0,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf]
pmax2= [5,15,np.inf,np.inf,np.inf,np.inf,np.inf]

class Fit_polynomial():
    def __init__(self,freq,freqdata,funci = func5,initt =init2,pmin = pmin2, pmax = pmax2 ):
        self.freq = freq
        self.initt = initt
        self.pmin = pmin
        self.pmax = pmax
        self.freqdata = freqdata
        self.funci = funci
    def run_fit(self):
        print('Running fitting under parameter: \n init:',self.initt,
                     '\n pmin:',self.pmin,
                     '\n pmax:',self.pmax,'\n Fitting using Trust Region Reflective, may takes 5~20min ......')
        freq,freqdata = self.freq,self.freqdata
        popt, pcov = curve_fit(self.funci, freq, freqdata,sigma=[1e-18]*len(freq),
                              absolute_sigma=True,p0 = self.initt,
                               bounds=(self.pmin,self.pmax),
                               maxfev=int(1e20))
        print('The fitted parameters as followed:\n',popt)

        fit_data = self.funci(freq,*popt)
        residual = freqdata - fit_data
        plt.xlabel("Freq/Hz")
        plt.ylabel("T/K")
        plt.plot(freq,residual)
        plt.title("Residual of polynomial+gaussian fitting")
        plt.show()
        self.popt =popt
        self.residual = residual
    def check_gauss(self):
        popt = self.popt
        freq = self.freq
        fitted_gauss = gaussian_func(freq,popt[-3],popt[-2],popt[-1])

        plt.plot(freq,fitted_gauss,'--',label = 'fit')
        plt.plot(freq,gaussian_func(freq,-0.53,78.1,18.7),label='origion')
        plt.legend(fontsize='x-large')
        plt.xlabel("Freq/Hz")
        plt.ylabel("T/K")
        plt.title('Fitted Gauss',size = 20)
        self.fitted_gauss = fitted_gauss