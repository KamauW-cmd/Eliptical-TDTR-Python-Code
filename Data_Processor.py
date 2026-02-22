import numpy as np
from scipy.io import loadmat,savemat
import os
from pathlib import Path


def Data_Processor(filepath :str):
    
    Filename = 'Processed_Data.mat'
    data = np.loadtxt(filepath)

    tdelay = data[:,0]  # [ps]
    amplitude = data[:,1] # [mV]
    phase_deg = data[:,2] # [deg]

    Vin = np.array([])
    Vout = np.array([])
    Ratio = np.array([])
    phase = np.deg2rad(phase_deg)

    Vin_temp = amplitude * np.cos(phase)
    Vout_temp = amplitude * np.sin(phase)

    phi_delta = 6.156891480000001 #3.015299
    Vin = Vin_temp*np.cos(phi_delta)-Vout_temp*np.sin(phi_delta)
    Vout = Vout_temp*np.cos(phi_delta)+Vin_temp*np.sin(phi_delta)

    Ratio = -Vin/Vout
    Vdet = amplitude

    Data = {
        'tdelay':tdelay,
        'Vin':Vin,
        'Vout':Vout,
        'Ratio':Ratio,
        'Vdet':Vdet
    }

    savemat(Filename, {'Data':Data})
    return Filename