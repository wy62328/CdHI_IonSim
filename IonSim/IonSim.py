#~
import h5py
import numpy as np
import pandas as pd
import healpy as hp
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
import astropy.units as u
from astropy.time import Time
from tqdm import tqdm
from scipy.integrate import quad
from scipy.interpolate import griddata
from astropy import units as u
from astropy import constants as const
import logging
logging.getLogger().setLevel(logging.INFO)
from IonSim.IonSim_Constant import *
'''Automatic download needs to be shut down due to local certificate error'''
from astropy.utils.iers import conf
conf.auto_download = False

import matplotlib.pyplot as plt

class IonSim():
    def __init__(self,alt,az,nulist,outpath,lon = [],lat = [],NmD = [],SNR=100,nside = 512,dtime = 30,gaussian_beam=True,dynamic_ion_d = False,dynamic_ion_complex = False,integral=False,beam_file = '/public/home/wangyue2/workplace/Ionosphere_sim/Beam_interp/interp_beam.hdf5'):
        self.nulist = nulist #Hz
        self.alt_collections = np.array(alt)
        self.az_collections = np.array(az)
        self.lon_collections = np.array(lon)
        self.lat_collections = np.array(lat)
        self.NmD_collections = np.array(NmD)
        self.SNR = SNR
        self.nside = nside
        self.dtime = dtime#min
#         self.outpath = outpath
        self.outfile = outpath
        self.gaussian_beam = gaussian_beam
        self.dynamic_ion_complex = dynamic_ion_complex
        self.dynamic_ion_d = dynamic_ion_d
        self.beam_file = beam_file
        self.integral = integral
        
    def run_sim(self,data):
        if data.shape[0]!= self.nulist.shape[0]:
            logging.error('frequency list have incorrect shape!')
        if data.shape[1]!= 12*self.nside**2:
            logging.error('have wrong nside!')        
        self.calculate_sigma()
        gauss_norm_corr_nu = self.norm_corr()
        if self.dynamic_ion_complex:
            logging.info('start interpolation for complex dynamic ionosphere')
            self.ionosphere_process()
        if self.dynamic_ion_d:
            logging.info('start calculating the dynamic Electron density for D-layer')
            self.ne_D_d10min = ne_D_d10min
        if self.integral:
            logging.info('start calculate refraction corrrect for F-layer ionosphere')
            deviation = []
            for nu in tqdm(self.nulist):
                calculated_deviation = self.deviation_angle_calculate(nu)
                deviation.append(calculated_deviation)
            self.deviation = np.array(deviation)
            print(self.deviation.shape)
        logging.info('start simulation for dtime = %d min'%self.dtime)
        T_FG_1_collection,T_FG_2_collection = [],[]
#         print(int(self.alt_collections.shape[0]))
        for i in tqdm(range(int(self.alt_collections.shape[0]))):

            alt,az = self.alt_collections[i],self.az_collections[i]
#             logging.info('Generating gaussian noise ...')
#             self.add_noise()
#            logging.info('simulation for all frequency given...Now processing 2018-01-02T{0}:{1}:00'.format(hour,minute))
            tran1,tran2 =[],[]
            for j in range(len(self.nulist)):
                nu = self.nulist[j] #Hz
                map_data,alt_corr,az = self.coord_(alt,az,data[j],nu,j)
                T_FG_1,T_FG_2 = self.add_beam(map_data,alt,alt_corr,az,nu,self.sigma[j],gauss_norm_corr_nu[j],i)
                tran1.append(np.mean(np.nan_to_num(T_FG_1, copy=True, nan=0.0)))
                tran2.append(np.mean(np.nan_to_num(T_FG_2, copy=True, nan=0.0)))
                ##T_FG_1是电离层+beam,T_FG_2是只加beam
##20/3/2023 wy change from txt output to h5py, cause run speed has been highly improved.
#                 T_FG_1 = T_FG_1 + self.gaussian_noise[j]
#                 T_FG_2 = T_FG_2 + self.gaussian_noise[j]
#                 self.write_1(str(np.mean(np.nan_to_num(T_FG_1, copy=True, nan=0.0)))+' ')
#                 self.write_2(str(np.mean(np.nan_to_num(T_FG_2, copy=True, nan=0.0)))+' ')
#             self.write_1('\n')
#             self.write_2('\n')
            T_FG_1_collection.append(tran1)
            T_FG_2_collection.append(tran2)
        hf = h5py.File(self.outfile, 'w')
        hf.create_dataset('T_ion', data=T_FG_1_collection)
        hf.create_dataset('T_noion', data=T_FG_2_collection)
        hf.close()
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
        logging.info('Calculating normalized spherical Beam coefficient ...')
        dist = self.spheredist(self.alt_collections[0],self.az_collections[0],np.pi/2,np.pi)
        ffd = np.ones([12*self.nside**2,])
        yy = []
        for ii in tqdm(range(len(self.nulist))):
            beamm1 = self.GenerateBeam(dist,self.sigma[ii],self.nulist[ii])
            yy.append(beamm1*ffd)
        cyy = 1/np.mean(yy,axis = 1)
        return cyy
    def coord_(self,alt,az,dataii,nu,ii):
        #地平之下坐标赋nan，以及电离层的角度改正
        alt_corr = self.refraction_corr(nu,alt,ii)
        map_data = dataii#-2.725
        map_data[alt_corr<0] = np.nan
        az[alt_corr<0] = np.nan
        alt[alt_corr<0] = np.nan
        alt_corr[alt_corr<0] = np.nan
        return map_data,alt_corr,az
    def deviation_angle_calculate(self,freq):
        lower_limit = h_m+d
        upper_limit = h_m-d
        theta_list = np.arange(-90,90.05,0.05)/180*np.pi

        deviation = []
        for ii in range(len(theta_list)):
            theta = theta_list[ii]
            result, error = quad(integral_parabola, lower_limit, upper_limit, args=(h_m, d, R_E, theta, nu_p_F,freq))
            final_result = result*R_E/(d)**2*(nu_p_F/freq)**2*np.cos(theta)
            deviation.append(final_result)
        return np.array(deviation)

    def refraction_corr(self,nu,alt,ii):
        #delta_theta = 2*d/(3*R_E)*(nu_p_F/nu)**2*(1+h_m/R_E)*(np.sin(alt)**2+2*h_m/R_E)**(-3/2)*np.cos(alt)
        #This formula is a ROUGH ESTIMATE of integral, after verification found difference was three times at 30MHz
        #But above 50MHz, the difference actually is smaller than 4 minutes
        #So better use the complete integration formula.
        if self.integral:
            theta_list = np.arange(-90,90.05,0.05)/180*np.pi
            calculated_deviation = self.deviation[ii]
            delta_theta = np.interp(alt,theta_list,calculated_deviation)
        else:
            delta_theta = 2*d/(3*R_E)*(nu_p_F/nu)**2*(1+h_m/R_E)*(np.sin(alt)**2+2*h_m/R_E)**(-3/2)*np.cos(alt)
        return alt+delta_theta
    def loss(self,nu_p,nu,theta,i):
        if self.dynamic_ion_d:
            Ne_D = ne_D_d10min[int(i%72)]
        else:
            Ne_D = ne_D
        #Absorption of D layer
        nu_c = 3.65*Ne_D/(Te_D**(3/2))*(19.8+np.log(Te_D**(3/2)/nu)) #Hz
        yita_D = -0.5*(nu_p**2*nu_c/nu)/(nu**2+nu_c**2)
        delta_s = delta_H_D*(1+H_D/R_E)/np.sqrt(np.cos(np.pi/2-theta)**2+2*H_D/R_E)
        return(np.exp(4*np.pi*nu*delta_s/C*yita_D))
    def emission(self,nu,L):   
        return (np.ones(shape=L.shape)-L)*T_e
    def cartcoord(self,theta, phi):
        return np.array([np.cos(theta)*np.cos(phi), np.cos(theta)*np.sin(phi), np.sin(theta)])
    def spheredist(self,theta, phi, theta_0, phi_0):
        return np.arccos(np.einsum('ji,j->i', self.cartcoord(theta, phi), self.cartcoord(theta_0, phi_0)))
    def GenerateBeam(self,dist,sigma,nu):
        if self.gaussian_beam:
            return 1/(2*np.pi)/sigma**2 * np.exp(-(dist**2)/(2*sigma**2))
        else:
            beam_file = self.beam_file#'/public/home/wangyue2/workplace/Ionosphere_sim/Beam_interp/interp_beam.hdf5'
            
            hf = h5py.File(beam_file, 'r')
            Am = hf['Am']
            Theta = hf['Theta']
            beam_freq = np.around(hf['freq'],1)
            nu = round(nu/1e6,1)
            index = np.where(beam_freq==nu)
#             index = ("%s"%nu).zfill(5)
#             df = pd.read_table('/public/home/wangyue2/workplace/Ionosphere_sim/Beam/beam_%sMHz.dat'%index)
#             del df['NaN']
#             df = df.fillna(0)
            return np.interp(dist,Theta,Am[index][0])
    def add_beam(self,map_data,alt,alt_corr,az,nu,ss,norm_corr,i):
        ##########Loss&Emission&Beam##################
        if self.dynamic_ion_complex:
            nu_p_D = self.nu_p_D[i]
        else:
            if self.dynamic_ion_d:
                Ne_D = ne_D_d10min[int(i%72)]
            else:
                Ne_D = ne_D
            nu_p_D = np.sqrt(e**2*ne_D/(eposilon_0*m))/(2*np.pi)
        
        L = self.loss(nu_p_D,nu,alt_corr,i)
        E = self.emission(nu,L)
        data_corr = map_data*L+E
        ### here ignore the Emission to check change of gaussian signal 
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
        T_FG_2 = T_FG_2#+2.725
        T_FG_1 = T_FG_1#+2.725
        return T_FG_1,T_FG_2
    def ionosphere_process(self):
        #先转换坐标，把fk5坐标转换到galactic，再把foreground的点拉开二维进行插值。频率不影响NmD的插值，所以需要插786432*96个点。
        #NmD的银道坐标其实就是x = np.arange(0,12*nsides**2,1)，theta,phi = hp.pixelfunc.pix2ang(nsides,x)
        xx = np.arange(0,12*128**2,1)
        glon,glat = hp.pixelfunc.pix2ang(128,xx)
        
        #Dynamic Ionosphere process. nside = 128 only.
        #3-D map of density in m-3 of D-region inflection point(lon,lat,time).       
        time_collections = np.arange(0,24,15/60)
        glonmesh,Tmesh = np.array(np.meshgrid(np.array(glon),np.array(time_collections)))
        glatmesh,Tmesh = np.array(np.meshgrid(np.array(glat),np.array(time_collections)))
        # 96 is the number of time points, when dtime equals 15min for 24 hours survey
        Tmesh,glonmesh,glatmesh = np.reshape(Tmesh,(96,196608,1)),np.reshape(glonmesh,(96,196608,1)),np.reshape(glatmesh,(96,196608,1))
        glonlatmesh = np.append(glonmesh,glatmesh,axis = 2)
        mesh_points = np.append(glonlatmesh,Tmesh,axis = 2)
        
        values = np.reshape(np.array(self.NmD_collections),(196608*96))#实际点的值
        know_points = np.reshape(mesh_points,(12*128**2*96,3))
        
        #Make meshmap of foreground map
        yy = np.arange(0,12*self.nside**2,1)
        flon,flat = hp.pixelfunc.pix2ang(self.nside,yy)
        
        time_collections = np.arange(0,24,self.dtime/60)
        flonmesh,fTmesh = np.array(np.meshgrid(np.array(flon),np.array(time_collections)))
        flatmesh,fTmesh = np.array(np.meshgrid(np.array(flat),np.array(time_collections)))
        fTmesh,flonmesh,flatmesh = np.reshape(fTmesh,(96,12*self.nside**2,1)),np.reshape(flonmesh,(96,12*self.nside**2,1)),np.reshape(flatmesh,(96,12*self.nside**2,1))
        flonlatmesh = np.append(flonmesh,flatmesh,axis = 2)
        fmesh_points = np.append(flonlatmesh,fTmesh,axis = 2)
        point_grid = np.reshape(fmesh_points,(12*self.nside**2*96,3))
        
        #############interpolation using griddata.
        grid_NmD = griddata(know_points, values, point_grid, method='nearest')
        # grid_NmD.shape is (12*self.nside**2*96)
        self.NmD = np.reshape(grid_NmD,(96,12*self.nside**2))
        # Now we have the matrix of D layer electron Number density.
        self.calculate_nuD()
    def calculate_nuD(self):
        self.nu_p_D = np.sqrt(e**2*self.NmD/(eposilon_0*m))/(2*np.pi) #electron plasma frequency of D layer
#         print('nu_p_D.shape = ',self.nu_p_D.shape)
#     def add_noise(self):
#         # Draw random samples from a normal (Gaussian) distribution.
#         SNR = self.SNR
#         #as amplitude of signal about 0.53K, Am_noise can be written as 0.53/SNR
#         #In cases of EDGE experiment, SNR were 37 and 52.
#         nu = self.nulist
#         noise = np.random.normal(0, 1, len(nu))
#         # Compute the actual standard deviation of the noise
#         Am_noise = 0.53/SNR
#         self.gaussian_noise = Am_noise/np.max(noise) * noise
        
#     def write_1(self,data):
#         with open(self.outfile1,"a") as file:
#             file.write(data)
#             file.close
#     def write_2(self,data):
#         with open(self.outfile2,"a") as file:
#             file.write(data)
#             file.close
def integral_parabola(x, hm, d, R_e, theta, nu_p_F, freq):
    eta_F_parabola = np.sqrt(1-(nu_p_F/freq)**2*(1-((x-h_m)/d)**2))
    top_part = (x-h_m)/eta_F_parabola**2
    bottom_part = np.sqrt((eta_F_parabola**2)*(x+R_e)**2-R_e**2*(np.cos(theta))**2)
    return top_part/bottom_part

        