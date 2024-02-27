#~
import h5py
import numpy as np
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
import astropy.units as u
import healpy as hp
from astropy.utils.iers import conf
from tqdm import tqdm

conf.auto_download = False

def coord_trans(time,nside):
    ## Transfer angles of galactic coord to the local
    x = np.arange(0,12*nside**2,1)
    theta,phi = hp.pixelfunc.pix2ang(nside,x)
    galactic_coord = SkyCoord(phi*u.radian,
                    (np.pi/2-theta)*u.radian,
                    frame='galactic')
    t = Time(time, format='isot', scale='utc')#'2022-01-02T09:00:00.5'
    location = EarthLocation(lon=-17.89 * u.deg,
                             lat=40 * u.deg,
                             height=2200 * u.m,)
    frame = AltAz( location=location,obstime=t)#
    local_coord = galactic_coord.transform_to(frame)
    return local_coord.alt.radian,local_coord.az.radian

def run_ts(dtime = 30,nside = 256,otime=np.array([[2018,1,1],]),filepath='/public/home/wangyue/workspace/Ionosphere_sim/Crime data process/Time_Series_0102_test.hdf5'):
    alt_collections = []
    az_collections = []
    for j in tqdm(range(otime.shape[0])):
        year,month,day = otime[j,0],otime[j,1],otime[j,2],
        for i in range(int(6*60/dtime)):
            t = i*dtime + 18*60 #从晚六点开始观测
            hour,minute = int((t-t%60)/60),t%60
            alt,az = coord_trans('{0}-{1}-{2}T{3}:{4}:00'.format(year,month,day,hour,minute),nside)#
            alt_collections.append(alt)
            az_collections.append(az)
        for i in range(int(6*60/dtime)):
            t = i*dtime  #继续从次日0点
            hour,minute = int((t-t%60)/60),t%60
            alt,az = coord_trans('{0}-{1}-{2}T{3}:{4}:00'.format(year,month,day+1,hour,minute),nside)#
            alt_collections.append(alt)
            az_collections.append(az)
    print('Done! Generate time series for %s nights.'%otime.shape[0])
    hf = h5py.File(filepath, 'w')
    hf.create_dataset('alt', data=alt_collections)
    hf.create_dataset('az', data=az_collections)
    hf.close()