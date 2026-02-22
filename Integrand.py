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

"""""
def Integrand(s,r, N_layers, layer_props, interface_props, pump_props, probe_props, freq, x_offset, y_offset):
    x_pump_rad = pump_props[0]
    y_pump_rad = pump_props[1]

    x_probe_rad = probe_props[0]
    y_probe_rad = probe_props[1]

    w_x = np.sqrt((x_pump_rad ** 2 + x_probe_rad ** 2) / 2)
    w_y = np.sqrt((y_pump_rad ** 2 + y_probe_rad ** 2) / 2)

    u = s/(np.pi*w_x)
    v = r/(np.pi*w_y)

    if (len(layer_props) != N_layers):
        raise RuntimeError("Size of layer properties and number of layers do not match")

    if (len(interface_props) != (N_layers - 1)):
        raise RuntimeError("Incorrect number of interface properties")

    ConductionMatrix = np.array([[1, 0], [0, 1]])

### for loop, loops over number of layers to compute conduction matrix at each layer as well as interface conductance
    for i in range(N_layers):

        h = layer_props[i][0] #how thick layer N is
        kappa_zz = layer_props[i][1] # kappa Z of layer N
        kappa_xx = layer_props[i][2]
        kappa_yy = layer_props[i][3]
        cv = layer_props[i][4]
        # kappa_xz = layer_props[i][5]
        # kappa_xy = layer_props[i][6]
        # kappa_yz = layer_props[i][7]

        q = cmath.sqrt((4 * (np.pi)**2 * (kappa_xx*u)**2 + 2*(kappa_yy*v)**2)/kappa_zz + (1j * freq * cv)/kappa_zz)
        A = 1.0
        B = (-1.0 / (kappa_zz * q)) * np.tanh(q * h)
        C = -kappa_zz * q * np.tanh(q * h)
        D = 1.0

        HeatLayer = np.array([[A, B], [C, D]])

        ConductionMatrix = ConductionMatrix @ HeatLayer

        if (i < (N_layers - 1)):
            # R MATRIX (See Eq A.16)
            G = interface_props[i]
            InterfaceLayer = np.array([[1, -(1.0 / G)], [0, 1]])

            ConductionMatrix = ConductionMatrix @ InterfaceLayer

        C_total = ConductionMatrix[1][0]
        D_total = ConductionMatrix[1][1]

    return (-D_total / C_total)
    """

'''
def tanh_stable(z, thresh=20.0):
    """
    Stable tanh for complex z.
    For large |Re(z)|, tanh(z) ~ sign(Re(z)) to avoid exp overflow.
    """
    zr = np.real(z)
    out = np.tanh(z)  # default
    out = np.where(zr >  thresh, 1.0 + 0.0j, out)
    out = np.where(zr < -thresh, -1.0 + 0.0j, out)
    return out
'''


def conduction_ratio_uv(U, V, N_layers, layer_props, interface_props, freq, use_omega=True):
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
        th = q*h

        A = 1.0
        B = (-1.0/(kzz*q)) * th
        C = (-kzz*q) * th
        D = 1.0

        # M <- M @ [[A,B],[C,D]] (elementwise 2x2 mult)
        n00 = M00*A + M01*C
        n01 = M00*B + M01*D
        n10 = M10*A + M11*C
        n11 = M10*B + M11*D
        M00, M01, M10, M11 = n00, n01, n10, n11

        if i < N_layers - 1:
            G = interface_props[i]
            # Interface: [[1, -(1/G)], [0, 1]]
            n00 = M00
            n01 = M00*(-1.0/G) + M01
            n10 = M10
            n11 = M10*(-1.0/G) + M11
            M00, M01, M10, M11 = n00, n01, n10, n11

    Ctot = M10
    Dtot = M11

    # guard against blowups
    tiny = 1e-30
    ratio = np.where(np.abs(Ctot) < tiny, 0.0 + 0.0j, (-Dtot / Ctot))
    
    return ratio