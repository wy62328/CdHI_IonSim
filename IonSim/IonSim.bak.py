#~
import h5py
import numpy as np
import pandas as pd
import healpy as hp
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
import astropy.units as u
from astropy.time import Time
from tqdm import tqdm
from astropy import units as u
from astropy import constants as const
import logging
logging.getLogger().setLevel(logging.INFO)
from IonSim.IonSim_Constant import *
'''Automatic download needs to be shut down due to local certificate error'''
from astropy.utils.iers import conf
conf.auto_download = False

class IonSim():
    def __init__(self,alt,az,nulist,outpath,nside = 512,dtime = 30,gaussian_beam=True):
        self.nulist = nulist #Hz
        self.alt_collections = np.array(alt)
        self.az_collections = np.array(az)
        self.nside = nside
        self.dtime = dtime#min
        self.outpath = outpath
        self.outfile1 = self.outpath+"for+gaussian_ion_ver_newbeam_cmbcorr1_d10.txt"
        self.outfile2 = self.outpath+"for+gaussian_noion_ver_newbeam_cmbcorr1_d10.txt"
        self.gaussian_beam = gaussian_beam
        
    def run_sim(self,data):
        if data.shape[0]!= self.nulist.shape[0]:
            logging.error('frequency list have incorrect shape!')
        if data.shape[1]!= 12*self.nside**2:
            logging.error('have wrong nside!')        
        self.calculate_sigma()
        gauss_norm_corr_nu = self.norm_corr()
        logging.info('start simulation for 24 hours,dtime = %d min'%self.dtime)
        for i in range(1,int(24*60/self.dtime)):
            t = i*self.dtime
            hour,minute = int((t-t%60)/60),t%60
            logging.info('simulation for all frequency given...Now processing 2022-01-02T{0}:{1}:00'.format(hour,minute))
            for j in tqdm(range(len(self.nulist))):
                nu = self.nulist[j] #Hz
                alt,az = self.alt_collections[i-1],self.az_collections[i-1]
                map_data,alt_corr,az = self.coord_(alt,az,data[j],nu)
                T_FG_1,T_FG_2 = self.add_beam(map_data,alt,alt_corr,az,nu,self.sigma[j],gauss_norm_corr_nu[j])
                ##T_FG_1是电离层+beam,T_FG_2是只加beam
                self.write_1(str(np.mean(np.nan_to_num(T_FG_1, copy=True, nan=0.0)))+' ')
                self.write_2(str(np.mean(np.nan_to_num(T_FG_2, copy=True, nan=0.0)))+' ')
            self.write_1('\n')
            self.write_2('\n')
        logging.info('\n All Done!')
    def calculate_sigma(self):
        logging.info('Calculating sigma of gaussian beam ...')
                #calculate FWHM
        freq_list = self.nulist/1e6
        # calculate FWHM of beam roughly
        MHz = 1e6*u.s**(-1)
        wavelength = []
        for channel in range(len(freq_list)):
            wavelength.append(const.c/(freq_list[channel]*MHz))
        D = 15*u.meter
        resolution_angle = []
        for channel in range(len(wavelength)):
            resolution_angle.append(1.*wavelength[channel]/D)

        FWHM_beam = np.array(resolution_angle)
        self.sigma = np.array(FWHM_beam / 2.355)
    def norm_corr(self):
        logging.info('Calculating normalized spherical Gaussian coefficient ...')
        dist = self.spheredist(self.alt_collections[0],self.az_collections[0],np.pi/2,np.pi)
        ffd = np.ones([12*self.nside**2,])
        yy = []
        for ii in tqdm(range(len(self.nulist))):
            beamm1 = self.GenerateBeam(dist,self.sigma[ii],self.nulist[ii])
            yy.append(beamm1*ffd)
        cyy = 1/np.mean(yy,axis = 1)
        return cyy
    def coord_(self,alt,az,dataii,nu):
        #地平之下坐标赋nan，以及电离层的角度改正
        alt_corr = self.refraction_corr(nu,alt)
        map_data = dataii-2.725
        map_data[alt_corr<0] = np.nan
        az[alt_corr<0] = np.nan
        alt[alt_corr<0] = np.nan
        alt_corr[alt_corr<0] = np.nan
        return map_data,alt_corr,az
    def refraction_corr(self,nu,alt):
        delta_theta = 2*d/(3*R_E)*(nu_p_F/nu)**2*(1+h_m/R_E)*(np.sin(alt)**2+2*h_m/R_E)**(-3/2)*np.cos(alt)
        return alt+delta_theta
    def loss(self,nu_p,nu_c,nu,theta):
        #Absorption of D layer
        yita_D = -0.5*(nu_p**2*nu_c/nu)/(nu**2+nu_c**2)
        delta_s = delta_H_D*(1+H_D/R_E)/np.sqrt(np.cos(np.pi/2-theta)**2+2*H_D/R_E)
        return(np.exp(4*np.pi*nu*delta_s/C*yita_D))
    def emission(self,nu,L):   
        return (np.ones(shape=L.shape)-L)*800
    def cartcoord(self,theta, phi):
        return np.array([np.cos(theta)*np.cos(phi), np.cos(theta)*np.sin(phi), np.sin(theta)])
    def spheredist(self,theta, phi, theta_0, phi_0):
        return np.arccos(np.einsum('ji,j->i', self.cartcoord(theta, phi), self.cartcoord(theta_0, phi_0)))
    def GenerateBeam(self,dist,sigma,nu):
        if self.gaussian_beam:
            return 1/(2*np.pi)/sigma**2 * np.exp(-(dist**2)/(2*sigma**2))
        else:
            nu = nu/1e6
            index = ("%s"%nu).zfill(5)
            df = pd.read_table('/home/wangyue/aureum/Ionosphere_sim/Beam/beam_%sMHz.dat'%index)
            del df['NaN']
            df = df.fillna(0)
            return np.interp(dist,df['Theta[deg]']/360*2*np.pi,df['ZOY []'])
    def add_beam(self,map_data,alt,alt_corr,az,nu,ss,norm_corr):
        ##########Loss&Emission&Beam##################
        L = self.loss(nu_p_D,nu_c,nu,alt_corr)
        E = self.emission(nu,L)
        data_corr = map_data*L+E
        center = [90/180*np.pi,0/180*np.pi]
        dist = self.spheredist(alt_corr, az,*center)
        
        Beam1 = self.GenerateBeam(dist,ss,nu)*norm_corr
        #将nan赋值0，避免无法求均值
        T_FG_1 = Beam1*data_corr
        np.nan_to_num(T_FG_1 , copy=False, nan=0.0)

        #########Only Beam############################
        dist2 = self.spheredist(alt,az,*center)
        Beam2 = self.GenerateBeam(dist2,ss,nu)*norm_corr
        T_FG_2 = Beam2*map_data
        np.nan_to_num(T_FG_2 , copy=False, nan=0.0)
        T_FG_2 = T_FG_2+2.725
        T_FG_1 = T_FG_1+2.725
        return T_FG_1,T_FG_2
    def write_1(self,data):
        with open(self.outfile1,"a") as file:
            file.write(data)
            file.close
    def write_2(self,data):
        with open(self.outfile2,"a") as file:
            file.write(data)
            file.close


        