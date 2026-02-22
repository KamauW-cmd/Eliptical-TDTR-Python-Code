import numpy as np
from scipy.io import loadmat,savemat
from Sub_A import TDTR_Bidirectional_SUB_A
from lgwt_V4 import lgwt_V4

'''
def TDTR_Bidirectional_SUB_B(tdelay, tau_rep, f, Lambda, C, h, eta,
                              r_pump, r_probe, P_pump, nnodes,
                              X_heat, X_temp):
    """
    Computes the TDTR reflectance and ratio per Feser (2004) Eqs. 9,19,20,21.
    """

    if not hasattr(TDTR_Bidirectional_SUB_B, "has_saved"):
        TDTR_Bidirectional_SUB_B.has_saved = False

    ii = 1j
    # maximum frequency for spectral damping
    fmax = 20 / np.min(np.abs(tdelay))
    # cutoff wavevector
    r_pump_max = r_pump[-1] if hasattr(r_pump, '__len__') else r_pump
    kmax = 2 / np.sqrt(r_pump_max**2 + r_probe**2)

    # Gauss-Legendre nodes & weights
    kvect, weights = lgwt_V4(nnodes, 0, kmax)
    weights = weights.flatten()  # shape (Nk,)

    # Fourier series indices
    M = int(20 * np.ceil(tau_rep / np.min(np.abs(tdelay))))
    mvect = np.arange(-M, M + 1)
    # spectral damping
    fudgep = np.exp(-np.pi * ((mvect / tau_rep + f) / fmax)**2)  # (Nf,)
    fudgem = np.exp(-np.pi * ((mvect / tau_rep - f) / fmax)**2)

    Nt_delay = len(tdelay)
    N_heat  = len(X_heat)
    # preallocate output
    T     = np.zeros((Nt_delay, N_heat), dtype=complex)
    Ratio = np.zeros_like(T, dtype=float)

    # time-phase matrix
    exp_mt = np.exp(ii * 2 * np.pi / tau_rep * np.outer(tdelay, mvect))  # (Nt_delay, Nf)

    for i, Xh in enumerate(X_heat):
        # Sub-A positive and negative sidebands
        Ip = TDTR_Bidirectional_SUB_A(kvect, mvect / tau_rep + f,
                                      Lambda, C, h, eta,
                                      r_pump, r_probe, P_pump,
                                      Xh, X_temp)
        Im = TDTR_Bidirectional_SUB_A(kvect, mvect / tau_rep - f,
                                      Lambda, C, h, eta,
                                      r_pump, r_probe, P_pump,
                                      Xh, X_temp)
        # squeeze third dimension if present: Ip2 shape (Nk, Nf)
        Ip2 = Ip[:, :, 0] if Ip.ndim == 3 else Ip
        Im2 = Im[:, :, 0] if Im.ndim == 3 else Im

        # integrate over k
        dTp = weights @ Ip2    # shape (Nf,)
        dTm = weights @ Im2

        # spectral combination
        spec = dTp * fudgep + dTm * fudgem   # (Nf,)
        anti = dTp - dTm                     # (Nf,)

        # construct Retemp and Imtemp: (Nt_delay, Nf)
        Retemp = spec[None, :] * exp_mt
        Imtemp = (-ii * anti)[None, :] * exp_mt

        # sum over Fourier terms
        Resum = np.sum(Retemp, axis=1)  # (Nt_delay,)
        Imsum = np.sum(Imtemp, axis=1)

        # complex modulation response
        Tm = Resum + ii * Imsum  # (Nt_delay,)

        # apply carrier exp(i*2π f tdelay)
        carrier = np.exp(ii * 2 * np.pi * f * tdelay)
        T[:, i] = Tm * carrier

        # ratio
        Ratio[:, i] = -np.real(T[:, i]) / np.imag(T[:, i])

        if not TDTR_Bidirectional_SUB_B.has_saved:
            savemat('TDTR_Var_B_fixed.mat', locals())
            TDTR_Bidirectional_SUB_A.has_saved = True

    return T, Ratio


import numpy as np
from scipy.io import savemat

def TDTR_Bidirectional_SUB_B(tdelay, tau_rep, f, Lambda, C, h, eta,
                              r_pump, r_probe, P_pump, nnodes,
                              X_heat, X_temp):
    if not hasattr(TDTR_Bidirectional_SUB_B, "has_saved"):
        TDTR_Bidirectional_SUB_B.has_saved = False

    ii = 1j
    fmax = 20 / np.min(np.abs(tdelay))
    r_pump_max = r_pump[-1] if isinstance(r_pump, np.ndarray) and r_pump.size > 1 else r_pump
    kmax = 2 / np.sqrt(r_pump_max**2 + r_probe**2)

    kvect, weights = lgwt_V4(nnodes, 0, kmax)
    weights = weights.flatten()

    M = int(20 * np.ceil(tau_rep / np.min(np.abs(tdelay))))
    mvect = np.arange(-M, M + 1)
    fudgep = np.exp(-np.pi * ((mvect / tau_rep + f) / fmax)**2)
    fudgem = np.exp(-np.pi * ((mvect / tau_rep - f) / fmax)**2)

    Nt_delay = len(tdelay)
    N_heat = len(X_heat)
    T = np.zeros((Nt_delay, N_heat), dtype=complex)
    Ratio = np.zeros_like(T, dtype=float)

    exp_mt = np.exp(ii * 2 * np.pi / tau_rep * np.outer(tdelay, mvect))

    for i, Xh in enumerate(X_heat):
        Ip = TDTR_Bidirectional_SUB_A(kvect, mvect / tau_rep + f,
                                      Lambda, C, h, eta,
                                      r_pump, r_probe, P_pump,
                                      Xh, X_temp)
        Im = TDTR_Bidirectional_SUB_A(kvect, mvect / tau_rep - f,
                                      Lambda, C, h, eta,
                                      r_pump, r_probe, P_pump,
                                      Xh, X_temp)

        # Handle scalar vs vector r_pump cases
        if np.isscalar(r_pump) or (isinstance(r_pump, np.ndarray) and r_pump.size == 1):
            if Ip.ndim == 3:
                Ip = np.squeeze(Ip, axis=-1)
                Im = np.squeeze(Im, axis=-1)
            dTp = np.dot(weights, Ip).reshape(1, -1)  # [1, Nf]
            dTm = np.dot(weights, Im).reshape(1, -1)
        else:
            dTp = np.tensordot(weights, Ip, axes=([0], [0])).T  # [Nt, Nf]
            dTm = np.tensordot(weights, Im, axes=([0], [0])).T

        # Expand to full time-delay matrix
        if dTp.shape[0] == 1:  # Scalar case needs expansion
            dTp_full = np.ones((Nt_delay, 1)) @ dTp
            dTm_full = np.ones((Nt_delay, 1)) @ dTm
        else:
            dTp_full = dTp
            dTm_full = dTm

        # Create fudge factor matrices with proper dimensions
        fudgep_mat = np.ones((Nt_delay, 1)) @ fudgep.reshape(1, -1)
        fudgem_mat = np.ones((Nt_delay, 1)) @ fudgem.reshape(1, -1)

        # Compute spectral components
        spec = dTp_full * fudgep_mat + dTm_full * fudgem_mat
        anti = dTp_full - dTm_full

        # Calculate temperature components
        Retemp = spec * exp_mt
        Imtemp = -ii * anti * exp_mt

        Resum = np.sum(Retemp, axis=1)
        Imsum = np.sum(Imtemp, axis=1)
        Tm = Resum + ii * Imsum

        carrier = np.exp(ii * 2 * np.pi * f * tdelay)
        T[:, i] = Tm * carrier
        Ratio[:, i] = -np.real(T[:, i]) / np.imag(T[:, i])

    if not TDTR_Bidirectional_SUB_B.has_saved:
        savemat('TDTR_Var_B_fixed.mat', locals())
        TDTR_Bidirectional_SUB_B.has_saved = True

    return T, Ratio
'''
'''
import numpy as np
from scipy.io import savemat

def TDTR_Bidirectional_SUB_B(tdelay, tau_rep, f, Lambda, C, h, G, eta,
                              r_pumpx,r_pumpy, r_probex,r_probey, P_pump, nnodes,N_hermite,
                              X_heat, X_temp):
    if not hasattr(TDTR_Bidirectional_SUB_B, "has_saved"):
        TDTR_Bidirectional_SUB_B.has_saved = False

    ii = 1j
    fmax = 20 / np.min(np.abs(tdelay))
    #r_pump_max = r_pump[-1] if isinstance(r_pump, np.ndarray) and r_pump.size > 1 else r_pump
    #kmax = 2 / np.sqrt(r_pump_max**2 + r_probe**2)

    #kvect, weights = lgwt_V4(nnodes, 0, kmax)
    #weights = weights.flatten()

    M = int(20 * np.ceil(tau_rep / np.min(np.abs(tdelay))))
    mvect = np.arange(-M, M + 1)

    #fudgep = np.exp(-np.pi * ((mvect / tau_rep + f) / fmax)**2)
    #fudgem = np.exp(-np.pi * ((mvect / tau_rep - f) / fmax)**2)

    fudgep = mvect / tau_rep + f
    fudgem = mvect / tau_rep - f

    Nt_delay = len(tdelay)
    N_heat = len(X_heat)
    T = np.zeros((Nt_delay, N_heat), dtype=complex)
    Ratio = np.zeros_like(T, dtype=float)

    exp_mt = np.exp(ii * 2 * np.pi / tau_rep * np.outer(tdelay, mvect))
    '''


    '''
    for i, Xh in enumerate(X_heat):
        Ip = TDTR_Bidirectional_SUB_A(kvect, mvect / tau_rep + f,
                                      Lambda, C, h, eta,
                                      r_pump, r_probe, P_pump,
                                      Xh, X_temp)
        Im = TDTR_Bidirectional_SUB_A(kvect, mvect / tau_rep - f,
                                      Lambda, C, h, eta,
                                      r_pump, r_probe, P_pump,
                                      Xh, X_temp)

        # Handle scalar vs vector r_pump cases
        if np.isscalar(r_pump) or (isinstance(r_pump, np.ndarray) and r_pump.size == 1):
            if Ip.ndim == 3:
                Ip = np.squeeze(Ip, axis=-1)
                Im = np.squeeze(Im, axis=-1)
            dTp = np.dot(weights, Ip).reshape(1, -1)  # [1, Nf]
            dTm = np.dot(weights, Im).reshape(1, -1)
        else:
            dTp = np.tensordot(weights, Ip, axes=([0], [0])).T  # [Nt, Nf]
            dTm = np.tensordot(weights, Im, axes=([0], [0])).T

        # Expand to full time-delay matrix
        if dTp.shape[0] == 1:  # Scalar case needs expansion
            dTp_full = np.ones((Nt_delay, 1)) @ dTp
            dTm_full = np.ones((Nt_delay, 1)) @ dTm
        else:
            dTp_full = dTp
            dTm_full = dTm

        # Create fudge factor matrices with proper dimensions
        fudgep_mat = np.ones((Nt_delay, 1)) @ fudgep.reshape(1, -1)
        fudgem_mat = np.ones((Nt_delay, 1)) @ fudgem.reshape(1, -1)

        # Compute spectral components
        spec = dTp_full * fudgep_mat + dTm_full * fudgem_mat
        anti = dTp_full - dTm_full

        # Calculate temperature components
        Retemp = spec * exp_mt
        Imtemp = -ii * anti * exp_mt

        Resum = np.sum(Retemp, axis=1)
        Imsum = np.sum(Imtemp, axis=1)
        Tm = Resum + ii * Imsum

        carrier = np.exp(ii * 2 * np.pi * f * tdelay)
        T[:, i] = Tm * carrier
        Ratio[:, i] = -np.real(T[:, i]) / np.imag(T[:, i])

    return T, Ratio
    '''


    '''
    freq_p = mvect/tau_rep + f
    freq_m = mvect/tau_rep - f

    for i, Xh in enumerate(X_heat):
    # ΔT(ω) already spatially averaged by A23 inside SUB_A
        Tp = TDTR_Bidirectional_SUB_A(freq_p, Lambda, C, h, eta,
                                        r_pumpx, r_pumpy, r_probex, r_probey,
                                        P_pump, Xh, X_temp, N_hermite, G)   # shape (Nf,)
        Tm = TDTR_Bidirectional_SUB_A(freq_m, Lambda, C, h, eta,
                                        r_pumpx, r_pumpy, r_probex, r_probey,
                                        P_pump, Xh, X_temp, N_hermite, G)

        spec = Tp * fudgep + Tm * fudgem             # (Nf,)
        anti = Tp - Tm                                # (Nf,)

        Retemp = spec[None, :] * exp_mt               # (Nt, Nf)
        Imtemp = (-ii * anti)[None, :] * exp_mt

        Tm_t = Retemp.sum(axis=1) + ii*Imtemp.sum(axis=1)   # (Nt,)
        T[:, i] = Tm_t * np.exp(ii*2*np.pi*f*tdelay)        # carrier

        Ratio[:, i] = -np.real(T[:, i]) / np.imag(T[:, i])

    return T, Ratio
    '''
