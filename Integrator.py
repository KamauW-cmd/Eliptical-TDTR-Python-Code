from numpy.polynomial.hermite import hermgauss
from Integrand import conduction_ratio_uv
import numpy as np



def Integrator(N_layers,nnodes, layer_props, interface_props, pump_props, probe_props, freq,tau_rep, tdelay):
    dRdT = 1
    w0 = 2*np.pi*freq
    ws = 2*np.pi/tau_rep

    x_pump_rad = pump_props[0]
    y_pump_rad = pump_props[1]

    x_probe_rad = probe_props[0]
    y_probe_rad = probe_props[1]

    wxsq = 0.5*(x_pump_rad**2+x_probe_rad**2)
    wysq = 0.5*(y_pump_rad**2+y_probe_rad**2)

    w_x = np.sqrt(wxsq)
    w_y = np.sqrt(wysq)

    j = 1j
    N = 100

    s_nnodes,s_weights = hermgauss(nnodes)
    r_nnodes,r_weights = hermgauss(nnodes)

    X,Y = np.meshgrid(s_nnodes,r_nnodes, indexing = 'ij')
    WX,WY = np.meshgrid(s_weights,r_weights, indexing ='ij')
    jac = 1/((np.pi**2)*w_x*w_y)
    sum_n = 0+0j

    U = X/(np.pi*w_x)
    V = Y/(np.pi*w_y)


    for i in range(-N, N+1):
        n = i
        freq = w0+n*ws
        H_uv = conduction_ratio_uv(U,V, N_layers, layer_props, interface_props, freq, use_omega = True)
        integral = np.sum(WX*WY*H_uv)
        sum_n += jac*integral*np.exp(j*n*ws*tdelay)

    delta_R = (dRdT)*sum_n

    V_in = delta_R.real
    V_out = delta_R.imag

    Ratio = -V_in/V_out

    return delta_R, Ratio