#This function is part of the "Bidirectional" TDTR analysis.
#This function computes the argument of the integral in Eq. (9) in Ref. 2004_Cahill_RSI.
#This function is based on TDTR_TEMP_V4.m published by Joseph P. Feser under http://users.mrl.illinois.edu/cahill/tcdata/tcdata.html (12-September-2012).
#This function can handle varying pump size over delay time thanks to Greg Hohensee.

#Note that the temperature rise actually measured by the lock-in requires factor 2/pi for P_pump. Without this factor, the temperature rise at short time delays 
#calculated equals the per-pulse heating calculated with pump fluence averaged by probe pulse.

#Input parameters:
#tdelay   - time delay (ps)
#rau_rep  - repetition rate of laser (s)
#f        - modulation frequency (Hz)
#Lambda   - vector of thermal conductivities, Lambda(1) = top surface,(W/m/K)
#C        - vector of volumetric heat capacity (J/m^3/K)
#h        - thicknesses of each layer (layer N will NOT be used, semiinfinite)
#r_pump   - pump 1/e^2 radius (m)
#r_probe  - probe 1/e^2 radius (m)
#P_pump   - absorbed pump laser power (W) ...doesn't effect ratio
#X_heat   - X_heat defines the ininitesimal surface that is being heated 
#X_temp   - depth at which temperature response is calculated (SCALAR); to consider depth sensitivity of TDTR, solve for various X_temp's through the optical skin depth and weight solutions according to the depth sensitivity (assuming linear problem)

#Changes:
#1/7/2017: corrected error in calculation of transfer matrix for up layers; the error was in the argument of the expterm: exp(unplus*h(n+1)) instead of the correct exp(un*h(n)); the error affects modelling for X_heat > h(1) and/or X_temp > h(1), i.e., temperature measurements in layers below top layer    
#           corrected condition on line 62; prior version was wrong for X_temp > X_heat, as it assigned the wrong layer index to the X_temp, i.e., X_temp was determined for one layer above the correct layer

#---------------------------- BEGIN CODE ----------------------------------


import numpy as np
from scipy.io import savemat

'''
def TDTR_Bidirectional_SUB_A(kvectin, freq, Lambda, C, h, eta,
                              r_pump, r_probe, P_pump, X_heat, X_temp):
    """
    Python translation of the corrected MATLAB TDTR_Bidirectional_SUB_A.
    Computes the integrand for Eq. (9) following Feser (2012) with Hohensee's
    variable pump-size extension and the 2017 transfer-matrix fix.
    """
    # persistent save flag
    if not hasattr(TDTR_Bidirectional_SUB_A, "has_saved"):
        TDTR_Bidirectional_SUB_A.has_saved = False

    # 1) Determine heating layer split
    ii = 1j
    pi = np.pi

    hsum = np.cumsum(h)
    Vheat = np.ceil(hsum / X_heat)
    I_heatA = int(np.sum(Vheat == 1) + 1)

    if I_heatA == 1:
        X_heatL = X_heat
    else:
        X_heatL = X_heat - hsum[I_heatA - 2]

    test_split = False
    if 0 < X_heatL < h[I_heatA-1]:
        # split the heating layer
        Lambda = np.concatenate((Lambda[:I_heatA],    Lambda[I_heatA-1:]), axis=0)
        C      = np.concatenate((C[:I_heatA],         C[I_heatA-1:]),      axis=0)
        h      = np.concatenate((h[:I_heatA-1], [X_heatL, h[I_heatA-1] - X_heatL], h[I_heatA:]))
        eta    = np.concatenate((eta[:I_heatA],       eta[I_heatA-1:]),     axis=0)
        I_heatA += 1
        test_split = True
    I_heat = I_heatA

    # 2) Determine sensing layer split
    Vtemp = np.ceil(hsum / X_temp)
    I_temp = int(np.sum(Vtemp == 1) + 1)
    if I_temp == 1:
        X_tempL = X_temp
    else:
        X_tempL = X_temp - hsum[I_temp - 2]
    if test_split and X_temp >= X_heat and I_temp >= I_heatA:
        I_temp += 1
        X_tempL -= X_heatL

    # 3) Prepare frequency and k-space grids
    Nfreq = freq.size
    kvect = kvectin[:, np.newaxis] @ np.ones((1, Nfreq))
    kvect2 = kvect**2
    Nlayers = len(Lambda)
    Nint = len(kvectin)
    omega = 2 * pi * freq
    D = Lambda / C

    # 4) Transfer matrix downward (substrate side)
    q2 = np.outer(np.ones(Nint), ii * omega / D[-1])
    un = np.sqrt(4*pi**2 * eta[-1] * kvect2 + q2)
    gamman = Lambda[-1] * un
    Bp_down = np.zeros((Nint, Nfreq))
    Bm_down = np.ones((Nint, Nfreq))
    kterm2 = 4 * pi**2 * kvect2

    for n in range(Nlayers-1, I_heat-1, -1):
        q2 = np.outer(np.ones(Nint), ii * omega / D[n-1])
        un_minus = np.sqrt(eta[n-1] * kterm2 + q2)
        gamma_minus = Lambda[n-1] * un_minus
        AA = gamma_minus + gamman
        BB = gamma_minus - gamman
        temp1 = AA * Bp_down + BB * Bm_down
        temp2 = BB * Bp_down + AA * Bm_down
        expterm = np.exp(un_minus * h[n-1])
        Bp_down = 0.5 * temp1 / (gamma_minus * expterm)
        Bm_down = 0.5 * expterm * temp2 / gamma_minus
        # stability fix
        mask = h[n-1] * np.abs(un_minus) > 100
        Bp_down[mask] = 0
        Bm_down[mask] = 1
        gamman = gamma_minus

    alpha_down = Bp_down
    beta_down  = Bm_down
    gamma_Iheat = gamman

    # 5) Transfer matrix upward (film side) with fixed expterm argument
    q2 = np.outer(np.ones(Nint), ii * omega / D[0])
    un = np.sqrt(4*pi**2 * eta[0] * kvect2 + q2)
    gamman = Lambda[0] * un
    Bp_up = np.exp(-2 * un * h[0])
    Bm_up = np.ones((Nint, Nfreq))

    for n in range(1, I_heat-1):
        q2 = np.outer(np.ones(Nint), ii * omega / D[n])
        un_plus = np.sqrt(eta[n] * kterm2 + q2)
        gamma_plus = Lambda[n] * un_plus
        AA = gamma_plus + gamman
        BB = gamma_plus - gamman
        temp1 = AA * Bp_up + BB * Bm_up
        temp2 = BB * Bp_up + AA * Bm_up
        # corrected: use un * h[n-1] (previous layer) rather than un_plus * h[n]
        expterm = np.exp(un * h[n-1])
        Bp_up = 0.5 * temp1 / (gamma_plus * expterm)
        Bm_up = 0.5 * expterm * temp2 / gamma_plus
        mask = h[n] * np.abs(un_plus) > 100
        Bp_up[mask] = 0
        Bm_up[mask] = 1
        gamman = gamma_plus
        un = un_plus

    gamma_Iheatminus = Lambda[I_heat-2] * np.sqrt(4*pi**2 * eta[I_heat-2] * kvect2 + np.outer(np.ones(Nint), ii * omega / D[I_heat-2]))
    alpha_up = Bp_up
    beta_up  = Bm_up

    # 6) Compute coating of heating interface
    BNminus = -(alpha_up + beta_up) / (
        gamma_Iheat*(alpha_down - beta_down)*(alpha_up + beta_up) +
        gamma_Iheatminus*(alpha_down + beta_down)*(alpha_up - beta_up)
    )
    B1minus = (alpha_down + beta_down) * BNminus / (alpha_up + beta_up)
    u1 = np.sqrt(4*pi**2 * eta[0] * kvect2 + np.outer(np.ones(Nint), ii * omega / D[0]))
    B1plus = B1minus * np.exp(-2 * u1 * h[0])

    # 7) Propagate to sensing layer
    if   I_temp == 1:
        BTplus  = B1plus
        BTminus = B1minus
    elif I_temp == Nlayers:
        BTplus  = 0
        BTminus = BNminus
    elif I_temp < I_heat:
        # upward to sensing layer
        Bp = Bp_up.copy(); Bm = Bm_up.copy(); gam = lamb_dummy = None
        for n in range(1, I_temp):
            q2 = np.outer(np.ones(Nint), ii * omega / D[n])
            un_plus = np.sqrt(eta[n] * kterm2 + q2)
            gamma_plus = Lambda[n] * un_plus
            AA = gamma_plus + gamman
            BB = gamma_plus - gamman
            expterm = np.exp(un * h[n-1])
            temp1 = AA * Bp + BB * Bm
            temp2 = BB * Bp + AA * Bm
            Bp = 0.5 * temp1 / (gamma_plus * expterm)
            Bm = 0.5 * expterm * temp2 / gamma_plus
            mask = h[n] * np.abs(un_plus) > 100
            Bp[mask] = 0; Bm[mask] = 1
            gamman = gamma_plus; un = un_plus
        BTplus  = Bp * B1minus
        BTminus = Bm * B1minus
    else:
        # downward to sensing layer
        Bp = Bp_down.copy(); Bm = Bm_down.copy(); gam = None
        for n in range(Nlayers-1, I_temp-1, -1):
            q2 = np.outer(np.ones(Nint), ii * omega / D[n-1])
            un_minus = np.sqrt(eta[n-1] * kterm2 + q2)
            gamma_minus = Lambda[n-1] * un_minus
            AA = gamma_minus + gamman
            BB = gamma_minus - gamman
            expterm = np.exp(un_minus * h[n-1])
            temp1 = AA * Bp + BB * Bm
            temp2 = BB * Bp + AA * Bm
            Bp = 0.5 * temp1 / (gamma_minus * expterm)
            Bm = 0.5 * expterm * temp2 / gamma_minus
            mask = h[n-1] * np.abs(un_minus) > 100
            Bp[mask] = 0; Bm[mask] = 1
            gamman = gamma_minus
        BTplus  = Bp * BNminus
        BTminus = Bm * BNminus

    # 8) Final layer response G(k)
    q2 = np.outer(np.ones(Nint), ii * omega / D[I_temp-1])
    un = np.sqrt(4*pi**2 * eta[I_temp-1] * kvect2 + q2)
    G = BTplus * np.exp(un * X_tempL) + BTminus * np.exp(-un * X_tempL)

    # 9) Build integrand
    arg1 = -pi**2 * (r_pump**2 + r_probe**2) / 2
    expterm = np.exp(kvect2[:, :, np.newaxis] * arg1)
    Kernal  = 2*pi * P_pump * expterm * kvect[:, :, np.newaxis]
    Integrand = G[:, :, np.newaxis] * Kernal

    # 10) Optionally save once
    if not TDTR_Bidirectional_SUB_A.has_saved:
        savemat('TDTR_Var_Py_A.mat', locals())
        TDTR_Bidirectional_SUB_A.has_saved = True

    return Integrand

'''


'''
import numpy as np
from numpy.polynomial.hermite import hermgauss

ii =1j

def lambdas(s,r,C,w,Kx,Ky,Kz,Kxy,Kyz,Kxz,wx2,wy2):
    wx = np.sqrt(wx2)
    wy = np.sqrt(wy2)
    temp_11 = (ii*C*w)/Kz 
    temp_12 = (4/Kz)*((Kx/wx2)*s**2 + ((2*Kxy)/(wx*wy))*s*r+ (Ky/wy2)*r**2)
    lambda_1 = temp_11 + temp_12
    temp_21 = (4*ii)/Kz
    temp_22 = (Kxz*s)/wx + (Kyz*r)/wy
    lambda_2 = temp_21*temp_22

    return lambda_1,lambda_2

def u_s(lambda_1,lambda_2):
    uplus = (-lambda_2 + np.sqrt(lambda_2**2+4*lambda_1))/2
    uminus = (-lambda_2 - np.sqrt(lambda_2**2+4*lambda_1))/2
    if np.real(uplus)<np.real(uminus):
        uplus,uminus = uminus,uplus
    return uplus,uminus

def N_mat(Kz,uplus,uminus,z):
    temp_n1 = np.array([[1,1],[-Kz*uplus, -Kz*uminus]])
    temp_n2 = np.array([[np.exp(uplus*z), 0],[0, np.exp(uminus*z)]])
    N = temp_n1@temp_n2
    return N

def M_mat(Kz,uplus,uminus):
    temp_m1 = 1/(Kz*(uplus-uminus))
    temp_m2 = np.array([[-Kz*uminus, -1],[Kz*uplus, 1]])
    M = temp_m1*temp_m2

    return M

def R_mat(G):
    R = np.array([[1,-1/G],[0,1]])
    return R

#kvectin, freq, Lambda, C, h, eta,r_pumpx,r_pumpy,r_probex, r_probey, P_pump, X_heat, X_temp

def H_mat(s,r,C,freq, Lambda,G, h, eta,r_pumpx,r_pumpy,r_probex, r_probey, P_pump, X_heat, X_temp,num_layers):
    
    w = 2*np.pi*freq
    pi2 = np.pi**2
    wx2 = (r_pumpx**2+r_probex**2)/2
    wy2 = (r_pumpy**2+r_probey**2)/2

    for n in range(num_layers):
        if n == 0:
            z = h[n]
            C_f = C[n]
            Kx,Ky,Kz,Kxy,Kyz,Kxz = Lambda[n]
            lambda_1,lambda_2 = lambdas(s,r,C_f,w,Kx,Ky,Kz,Kxy,Kyz,Kxz,wx2,wy2)
            uplus,uminus = u_s(lambda_1,lambda_2)
            N1 = N_mat(Kz,uplus,uminus,z)
            M1 = M_mat(Kz,uplus,uminus)
            R1 = R_mat(G)
            
        else:
            C_s = C[n]
            Kx,Ky,Kz,Kxy,Kyz,Kxz = Lambda[n]
            lambda_1,lambda_2 = lambdas(s,r,C_s,w,Kx,Ky,Kz,Kxy,Kyz,Kxz,wx2,wy2)
            uplus,uminus = u_s(lambda_1,lambda_2)
            #N2 = N_mat(Kz,uplus,uminus,z)
            M2 = M_mat(Kz,uplus,uminus)

    T = M2 @ R1 @ N1 @ M1
    Cmat,D = T[1,0],T[1,1]

    H = -D/Cmat

    return H


    
def TDTR_Bidirectional_SUB_A(N,C,freq, Lambda,G, h, eta,r_pumpx,r_pumpy,r_probex, r_probey, P_pump, X_heat, X_temp, num_layers):
    ii = 1j
    pi2 = np.pi**2

    wx2 = (r_pumpx**2+r_probex**2)/2
    wy2 = (r_pumpy**2+r_probey**2)/2

    wx = np.sqrt(wx2)
    wy = np.sqrt(wy2)

    s_nodes,w_s = hermgauss(N)
    r_nodes,w_r = hermgauss(N)
    Integrand = 0.0+0.0j

    Jacobian = 1/(pi2*wx*wy)

    for si,s in enumerate(s_nodes):
        for rj,r in enumerate(r_nodes):
            H = H_mat(s,r,C,freq, Lambda,G, h, eta,r_pumpx,r_pumpy,r_probex, r_probey, P_pump, X_heat, X_temp,num_layers)
            Integrand += w_s[si]*w_r[rj]*H


    return Jacobian*Integrand
'''



