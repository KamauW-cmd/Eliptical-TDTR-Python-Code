import numpy as  np

def parameter_example():
    SysParam = {}
    SysParam['Lambda'] = np.array([315 , 0.3 , 145]) # Thermal conductivities (W m^-1 K^-1)
    SysParam['C'] = np.array([2.492, 0.1, 1.62])*1e6 # Volumetric heat capacities (J m^-3 K^-1)
    SysParam['h'] = np.array([133, 1, 1e6])*1e-9; # Thicknesses (m)   
    SysParam['eta'] = np.ones(SysParam['Lambda'].size) #Anisotropy parameter eta=kx/ky;

    SysParam['X_heat'] = np.array(range(1,36+1,5))*1e-9#.reshape(-1,1) # Temperature response is calculated for each entry i of the COLUMN vector X_heat, where X_heat(i) defines the ininitesimal surface that is being heated 
    SysParam['X_temp']  = 1e-9 #depth at which temperature response is calculated (SCALAR); to consider depth sensitivity of TDTR, solve for various X_temp's through the optical skin depth and weight solutions according to the depth sensitivity (assuming linear problem)
    SysParam['AbsProf'] = np.exp(-4*np.pi*4.9*SysParam['X_heat']/785e-9)#.reshape(-1,1) # COLUMN vector describing absorption profile through depths X_heat of the sample; does not need to be normalized

    #SETUP PROPERTIES
    SysParam['f']       = 1e5 #Laser modulation frequency (Hz)
    SysParam['r_pump']   = 7.541e-6 #Pump 1/e^2 radius (m)
    SysParam['r_probe']  = 0.780e-6 #Probe 1/e^2 radius, (m)
    SysParam['tau_rep']  = 1/80e6 #Laser repetition period (s) 
    SysParam['P_pump']   = 20e-3#0.8*0.3*7e-3 #absorbed pump power (transmission of objective * absorbance * pump power) 
    SysParam['P_probe']  = 0.1e-3#0.8*0.3*6e-3 #absorbed pump power (transmission of objective * absorbance * pump power); assumes AF chopper is OFF!  If not, then you need to multiply the probe power by 2.
    SysParam['tdelay_model'] = np.logspace(np.log10(100e-12),np.log10(1400e-12),30)#.reshape(-1,1) # time delays for model curve (s)

    #FIT SPECIFICATIONS
    #Choose layer indices; respective parameters are then adjusted in the fit;
    SysParam['FITNLambda'] = np.array([2,3]) #Fit for the thermal conductivities of the 2nd and 3rd layer
    SysParam['FITNC'] = [] # Not fitting the Volumetric Heat Capacity for any layer
    SysParam['FITNh'] = [] # Not fitting for the layer thickness of any layer
    # Choose range of time delays to fit (s)
    SysParam['tdelay_min'] = 100e-12
    SysParam['tdelay_max'] = 1400e-12

    return SysParam

