import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.integrate import quad
from scipy.integrate import quad_vec
import math
import cmath
import pdb
#layer_props structure:
##### col 0: height of the layer
##### col 1: kappa zz
##### col 2: kappa xx
##### col 3: kappa yy
##### col 7: CV

# pump_props structure
### [0]: x radius of pump beam
### [1]: y radius of pump beam

# probe_props structure
### [0]: x radius of probe beam
### [1]: y radius of probe beam


### INTEGRAND FUNCTION: RETURNS INTEGRAND FROM A23 in Jiang (2017)



def conduction_ratio_uv(U, V, N_layers, layer_props, interface_props, freq, use_omega=False):
    """
    Compute (-D_total/C_total) on the U,V grid, vectorized over all points.

    If use_omega=True, interpret `freq` as Hz and use ω = 2πf.
    Otherwise, uses your original (1j*freq*cv)/kzz convention.
    """
    # Start with identity matrix for each (u,v) point:
    M00 = np.ones_like(U, dtype=np.complex128)
    M01 = np.zeros_like(U, dtype=np.complex128)
    M10 = np.zeros_like(U, dtype=np.complex128)
    M11 = np.ones_like(U, dtype=np.complex128)

    wterm = (2*np.pi*freq) if use_omega else freq

    for i in range(N_layers):
        h   = layer_props[i][0]
        kzz = layer_props[i][1]
        kxx = layer_props[i][2]
        kyy = layer_props[i][3]
        cv   = layer_props[i][4]
        #rho = layer_props[i][5]
        #cv  = c * rho  # volumetric heat capacity [J/m^3-K] if c is cp [J/kgK]

        q = np.sqrt(4*np.pi**2*(kxx*U**2 + kyy*V**2)/kzz + (1j*wterm*cv)/kzz)
        #th = np.tanh(q*h)

        qh = q * h
        overflow_mask = np.real(qh) > 100
        th = np.empty_like(qh, dtype=np.complex128)
        th[overflow_mask] = 1.0 + 0.0j
        th[~overflow_mask] = np.tanh(qh[~overflow_mask])

        #th = np.tanh(np.clip(q*h,-100,100))
        
        A = 1.0
        B = (-1.0/(kzz*q)) * th
        C = (-kzz*q) * th
        D = 1.0

        # M <- M @ [[A,B],[C,D]] (elementwise 2x2 mult)
        #n00 = M00*A + M01*C
        #n01 = M00*B + M01*D
        #n10 = M10*A + M11*C
        #n11 = M10*B + M11*D

        # Left-multiply: T_layer @ M_old
        n00 = A*M00 + B*M10
        n01 = A*M01 + B*M11
        n10 = C*M00 + D*M10
        n11 = C*M01 + D*M11
        
        M00, M01, M10, M11 = n00, n01, n10, n11

        if i < N_layers - 1:
            G = interface_props[i]
            # Interface: [[1, -(1/G)], [0, 1]]
            #n00 = M00
            #n01 = M00*(-1.0/G) + M01
            #n10 = M10
            #n11 = M10*(-1.0/G) + M11

            # Left-multiply: R_interface @ M_old
            n00 = M00 - (1.0/G)*M10
            n01 = M01 - (1.0/G)*M11
            n10 = M10
            n11 = M11

            M00, M01, M10, M11 = n00, n01, n10, n11

    Ctot = M10
    Dtot = M11

    # guard against blowups
    tiny = 1e-30
    ratio = np.where(np.abs(Ctot) < tiny, 0.0 + 0.0j, (-Dtot / Ctot))
    
    return ratio