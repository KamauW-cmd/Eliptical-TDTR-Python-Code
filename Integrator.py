import numpy as np
from scipy.io import loadmat,savemat
from Integrand import conduction_ratio_uv
from numpy.polynomial.hermite import hermgauss

def Integrator(tdelay, tau_rep, f, Lambda, C, h, eta,
                              r_pump, r_probe, P_pump, nnodes,
                              X_heat, X_temp, pump_props, probe_props, layer_props, interface_props):
    """
    Computes the TDTR reflectance and ratio per Feser (2004) Eqs. 9,19,20,21.
    """

    x_pump_rad = pump_props[0]
    y_pump_rad = pump_props[1]

    x_probe_rad = probe_props[0]
    y_probe_rad = probe_props[1]

    wxsq = 0.5*(x_pump_rad**2+x_probe_rad**2)
    wysq = 0.5*(y_pump_rad**2+y_probe_rad**2)

    w_x = np.sqrt(wxsq)
    w_y = np.sqrt(wysq)
    
    ii = 1j
    # maximum frequency for spectral damping
    fmax = 20 / np.min(np.abs(tdelay))
    # cutoff wavevector
    r_pump_max = r_pump[-1] if hasattr(r_pump, '__len__') else r_pump
    kmax = 2 / np.sqrt(r_pump_max**2 + r_probe**2)


    s_nnodes,s_weights = hermgauss(nnodes)
    r_nnodes,r_weights = hermgauss(nnodes)

    X,Y = np.meshgrid(s_nnodes,r_nnodes, indexing = 'ij')
    WX_2D,WY_2D= np.meshgrid(s_weights,r_weights, indexing ='ij')

    jac = 1/((np.pi**2)*w_x*w_y)

    U_2D = X/(np.pi*w_x)
    V_2D = Y/(np.pi*w_y)

    U = U_2D[:,:,np.newaxis]
    V = V_2D[:,:,np.newaxis]

    WX = WX_2D[:, :,np.newaxis]
    WY = WY_2D[:, :,np.newaxis]

    M = int(20 * np.ceil(tau_rep / np.min(np.abs(tdelay))))
    mvect_1D = np.arange(-M, M + 1)

    mvect = mvect_1D[np.newaxis,np.newaxis,:]

    # spectral damping
    fudgep = np.exp(-np.pi * ((mvect_1D / tau_rep + f) / fmax)**2)  # (Nf,)
    fudgem = np.exp(-np.pi * ((mvect_1D / tau_rep - f) / fmax)**2)

    w0 = 2*np.pi*f
    ws = 2*np.pi/tau_rep

    omega_plus = w0+mvect*ws
    omega_minus = -w0+mvect*ws

    H_uv_plus = conduction_ratio_uv(U,V, len(layer_props), layer_props, interface_props, omega_plus, use_omega = False)
    H_uv_minus = conduction_ratio_uv(U,V, len(layer_props), layer_props, interface_props, omega_minus, use_omega = False)

    integral_plus = np.sum(jac*WX*WY*H_uv_plus, axis = (0,1))
    integral_minus = np.sum(jac*WX*WY*H_uv_minus, axis = (0,1))


    spec = integral_plus * fudgep + integral_minus * fudgem   # (Nf,)
    anti = integral_plus - integral_minus                    # (Nf,)

    exp_mt = np.exp(ii * 2 * np.pi / tau_rep * np.outer(tdelay, mvect_1D))  # (Nt_delay, Nf)

    # construct Retemp and Imtemp: (Nt_delay, Nf)
    Retemp = spec[None, :] * exp_mt
    Imtemp = (-ii * anti)[None, :] * exp_mt

    # sum over Fourier terms
    Resum = np.sum(Retemp, axis=1)  # (Nt_delay,)
    Imsum = np.sum(Imtemp, axis=1)

    # Recombine into the complex signal
    Tm = Resum + ii * Imsum 
    
    # Apply carrier rotation exp(i*2π f tdelay)
    carrier = np.exp(ii * 2 * np.pi * f * tdelay)
    T_final = Tm * carrier
    
    # Calculate the TDTR Ratio
    Ratio = -np.real(T_final) / np.imag(T_final)
    
    return T_final, Ratio


    