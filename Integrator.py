from numpy.polynomial.hermite import hermgauss
from Integrand import conduction_ratio_uv
import numpy as np



def Integrator(N_layers,nnodes, layer_props, interface_props, pump_props, probe_props, freq,tau_rep, tdelay):

    #print(tdelay)
    #tdelay = tdelay * 1e-12

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
    N = 10000

    s_nnodes,s_weights = hermgauss(nnodes)
    r_nnodes,r_weights = hermgauss(nnodes)

    X,Y = np.meshgrid(s_nnodes,r_nnodes, indexing = 'ij')
    WX,WY = np.meshgrid(s_weights,r_weights, indexing ='ij')
    jac = 1/((np.pi**2)*w_x*w_y)
    vin_sum = 0+0j
    vout_sum = 0+0j

    U = X/(np.pi*w_x)
    V = Y/(np.pi*w_y)

    Tm = 0+0j

    for i in range(-N, N+1):
        n = i
        #wn = w0+n*ws
        omega_plus = w0+n*ws
        omega_minus = -w0+n*ws
        H_uv_plus = conduction_ratio_uv(U,V, N_layers, layer_props, interface_props, omega_plus, use_omega = False)
        H_uv_minus = conduction_ratio_uv(U,V, N_layers, layer_props, interface_props, omega_minus, use_omega = False)
        integral_plus = np.sum(WX*WY*H_uv_plus)
        integral_minus = np.sum(WX*WY*H_uv_minus)
        delta_t_minus = jac*integral_minus
        delta_t_plus = jac*integral_plus
        vin_sum = (delta_t_plus+delta_t_minus)
        vout_sum = (delta_t_plus-delta_t_minus)
        carrier = np.exp(j*n*ws*tdelay)
        Tm_temp  = (vin_sum + vout_sum)
        Tm += 0.5*(dRdT)*Tm_temp * carrier

    #V_in = 0.5*(dRdT)*vin_sum
    #V_out = -0.5*j*(dRdT)*vout_sum

    Vin_real = np.real(Tm)
    #Vout_real = np.imag(V_out)
    Vout_real = np.imag(Tm)

    #Ratio = -Vin_real/Vout_real

    return Vin_real, Vout_real



