import numpy as np

ne_D =2.5*10**8  #electron destiny of Dlayer
ne_F =5*10**11  #critical destiny of F Layer
Te_D = 200#800 #K
# nu_c = 10**(7) #HZ########


eposilon_0 = 8.85*10**(-12) 
e = 1.6*10**(-19) 
m = 9.10956*10**(-31)
R_E=6378 #km
delta_H_D = 30#Km
H_D  =75 #Km
C = 2.99792*10**(5)
#nu_p_D=np.sqrt(e**2*ne_D/(eposilon_0*m))/(2*np.pi) #electron plasma frequency of D layer
nu_p_F=np.sqrt(e**2*ne_F/(eposilon_0*m))/(2*np.pi) #electron plasma frequency of F layer
L=np.zeros(50)
yita_D=np.zeros(50)
T_gama = 2.72548 # CMB temperature
T_s = 2#spin temperature
z =  20#redshift
x_HI = 0.51 # mean neutral hydrogen fraction
T_e = 200   #typical D-layer electron temperature of Te = 800 K for mid-latitude ionosphere
T_eba = 1500 
#ne_D =5*10**8  #electron destiny
# TEC_D = 20#10**(0.5)
h_m = 300 #critical height of F-layer, where electron number desity of layer achieve highest.
d = 100 #semi-thickness of F-layer
# example for NeD
ne_D_d10min_aftermid = np.array([0.25377E+09,0.25530E+09,0.25708E+09,0.25916E+09,0.26156E+09,0.26433E+09,0.26752E+09,0.27116E+09,
                        0.27530E+09,0.27997E+09,0.28519E+09,0.29098E+09,0.29731E+09,0.30415E+09,0.31143E+09,0.31905E+09,
                        0.32686E+09,0.33473E+09,0.34249E+09,0.35000E+09,0.35715E+09,0.36386E+09,0.37015E+09,0.37607E+09,
                        0.38171E+09,0.38719E+09,0.39263E+09,0.39810E+09,0.40369E+09,0.40966E+09,0.41648E+09,0.42459E+09,
                        0.43437E+09,0.44609E+09,0.45989E+09,0.47572E+09,0.49342E+09])-0.00377E+09
ne_D_d10min_beforemid = np.array([0.50864E+09,0.48968E+09,0.47237E+09,0.45697E+09,0.44361E+09,0.43231E+09,0.42290E+09,0.41508E+09,
                        0.40848E+09,0.40262E+09,0.39708E+09,0.39163E+09,0.38620E+09,0.38071E+09,0.37504E+09,0.36907E+09,
                        0.36272E+09,0.35594E+09,0.34874E+09,0.34120E+09,0.33342E+09,0.32557E+09,0.31779E+09,0.31024E+09,
                        0.30303E+09,0.29628E+09,0.29004E+09,0.28436E+09,0.27923E+09,0.27465E+09,0.27059E+09,0.26703E+09,
                        0.26391E+09,0.26120E+09,0.25885E+09,0.25682E+09])-0.00682E+09
ne_D_d10min = np.append(ne_D_d10min_beforemid,ne_D_d10min_aftermid)