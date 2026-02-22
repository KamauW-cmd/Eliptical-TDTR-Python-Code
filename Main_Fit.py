#This program is the main program for FITTING of the "Bidirectional" model to TDTR data.
#This program is based on TDTR_MAIN_V4.m published by Joseph P. Feser under http://users.mrl.illinois.edu/cahill/tcdata/tcdata.html (12/Sept/2012).
#This program can handle varying pump size over delay time thanks to Greg Hohensee.

#-------------------------------BEGIN CODE---------------------------------
#USER INPUT
import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from Data_Processor import Data_Processor

from Parameters import parameter_example
from Sub_C import TDTR_Bidirectional_SUB_C
from Sub_B import TDTR_Bidirectional_SUB_B
from extract_interior_V4 import extract_interior_V4  # you must define this helper

# ---- USER INPUT ----
SysParam = parameter_example()
datafile = Data_Processor(r'C:\Users\kamau\OneDrive\Documents\Research 2025\Python Code\Data\TDTR20x7-23-2025-AuonSiSandia_2')
tnorm = 200  # (ps)
auto_on = True
save_results = False
addfilename = ''
Col = 'k'
ClearFig = True
psc = False
frac = 0.19
tmax = 3.6e-9
nnodes = 35

# ---- LOAD PARAMETERS ----
Lamda = SysParam['Lambda'].copy()
C = SysParam['C'].copy()
h = SysParam['h'].copy()
eta = SysParam['eta']
X_heat = SysParam['X_heat']
X_temp = SysParam['X_temp']
AbsProf = SysParam['AbsProf']
f = SysParam['f']
r_pump = SysParam['r_pump']
r_probe = SysParam['r_probe']
tau_rep = SysParam['tau_rep']
P_pump = SysParam['P_pump']
P_probe = SysParam['P_probe']
tdelay_model = SysParam['tdelay_model']

FITNLambda = SysParam['FITNLambda']
FITNC = SysParam['FITNC']
FITNh = SysParam['FITNh']
tdelay_min = SysParam['tdelay_min']
tdelay_max = SysParam['tdelay_max']

if psc:
    r_pump_model = r_pump * (1 + frac * tdelay_model / 3.65e-9)
else:
    r_pump_model = r_pump

# ---- IMPORT DATA ----
Data = loadmat(datafile)['Data']
tdelay_raw = Data['tdelay'][0][0].flatten() * 1e-12
Vin_raw = Data['Vin'][0][0].flatten()
Vout_raw = Data['Vout'][0][0].flatten()
Ratio_raw = Data['Ratio'][0][0].flatten()
Vdet_raw = Data['Vdet'][0][0].flatten()

_, Vin_data = extract_interior_V4(tdelay_raw, Vin_raw, tdelay_min, tdelay_max)
_, Vout_data = extract_interior_V4(tdelay_raw, Vout_raw, tdelay_min, tdelay_max)
tdelay_data, Ratio_data = extract_interior_V4(tdelay_raw, Ratio_raw, tdelay_min, tdelay_max)

if psc:
    r_pump_data = r_pump * (1 + frac * tdelay_data / 3.65e-9)
else:
    r_pump_data = r_pump

# ---- INITIAL GUESS ----

X0 = np.zeros(len(FITNLambda)+len(FITNC)+len(FITNh))
for i in range(len(FITNLambda)):
    X0[i] = Lamda[FITNLambda[i]-1]

for i in range(len(FITNC)):
    X0[i+len(FITNLambda)] = C[FITNC[i]-1]

for i in range(len(FITNh)):
    X0[i+len(FITNh)+len(FITNLambda)] = h[FITNh[i]-1]

##X0 = np.concatenate([
##    Lamda[FITNLambda],
##    C[FITNC],
##    h[FITNh]
##])

# ---- FITTING ----
if auto_on:
    result = minimize(lambda X: TDTR_Bidirectional_SUB_C(
        X, Ratio_data, tdelay_data, tau_rep, f, Lamda.copy(), C.copy(), h.copy(),
        eta, r_pump_data, r_probe, P_pump, nnodes,
        FITNLambda, FITNC, FITNh, X_heat, X_temp, AbsProf
    )[0], X0, method='Nelder-Mead')
    Xsol = result.x
    Z, _ = TDTR_Bidirectional_SUB_C(Xsol, Ratio_data, tdelay_data, tau_rep, f, Lamda.copy(), C.copy(), h.copy(),
                                    eta, r_pump_data, r_probe, P_pump, nnodes,
                                    FITNLambda, FITNC, FITNh, X_heat, X_temp, AbsProf)
    print("Data fit completed.")
else:
    Xsol = X0.copy()
    while True:
        TDTR_Bidirectional_SUB_C(Xsol, Ratio_data, tdelay_data, tau_rep, f, Lamda.copy(), C.copy(), h.copy(),
                                 eta, r_pump_data, r_probe, P_pump, nnodes,
                                 FITNLambda, FITNC, FITNh, X_heat, X_temp, AbsProf)
        cont = input("Continue? (Yes: '1', No: '0'): ")
        if cont != '1':
            break
        for i, idx in enumerate(FITNLambda):
            Xsol[i] = float(input(f"New value for Lambda[{idx}]: "))

# ---- ASSIGN BACK ----
for i, idx in enumerate(FITNLambda):
    Lamda[idx-1] = Xsol[i]
for i, idx in enumerate(FITNC):
    C[idx] = Xsol[len(FITNLambda) + i]
for i, idx in enumerate(FITNh):
    h[idx] = Xsol[len(FITNLambda) + len(FITNC) + i]

# ---- FINAL MODEL ----
Ts, _ = TDTR_Bidirectional_SUB_B(tdelay_model, tau_rep, f, Lamda, C, h, eta,
                                 r_pump_model, r_probe, P_pump, nnodes, X_heat, X_temp)

Tin_model = (np.real(Ts) @ AbsProf) / (np.ones(AbsProf.shape).T @ AbsProf)
Tout_model = (np.imag(Ts) @ AbsProf) / (np.ones(AbsProf.shape).T @ AbsProf)
Ratio_model = -Tin_model / Tout_model

# ---- SAVE RESULTS ----
if save_results:
    from scipy.io import savemat
    savemat(f"{datafile[:-4]}_FIT{addfilename}.mat", {
        "Lambda": Lamda, "C": C, "h": h, "Ratio_model": Ratio_model,
        "Tin_model": Tin_model, "Tout_model": Tout_model
    })

# ---- PLOT RESULTS ----
plt.figure(figsize=(12, 5))

# Plot Ratio
plt.subplot(1, 2, 1)
plt.loglog(tdelay_raw * 1e12, Ratio_raw, 'ro', markersize=4)
plt.loglog(tdelay_model * 1e12, Ratio_model, Col, linewidth=2)
plt.xlabel('Time delay (ps)')
plt.ylabel('-Vin/Vout')
plt.grid(True)
plt.xlim([1, 5000])
plt.ylim([0.1, 10 * np.ceil(np.log10(np.max(Ratio_model)))])
plt.title("TDTR Ratio Fit")

# Plot Tin/Tout
plt.subplot(1, 2, 2)
Vin_norm = np.interp(tnorm, tdelay_raw * 1e12, Vin_raw)
Tin_model_norm = np.interp(tnorm, tdelay_model * 1e12, Tin_model)
Tin_data = Vin_raw / Vin_norm * Tin_model_norm

Vout_norm = np.interp(tnorm, tdelay_raw * 1e12, Vout_raw)
Tout_model_norm = np.interp(tnorm, tdelay_model * 1e12, Tout_model)
Tout_data = Vout_raw / Vout_norm * Tout_model_norm

plt.loglog(tdelay_model * 1e12, Tin_model, Col, linewidth=2)
plt.loglog(tdelay_model * 1e12, -Tout_model, Col, linewidth=2)
plt.loglog(tdelay_raw * 1e12, Tin_data, 'ro', markersize=4)
plt.loglog(tdelay_raw * 1e12, -Tout_data, 'ro', markersize=4)
plt.xlabel('Time delay (ps)')
plt.ylabel('ΔT')
plt.grid(True)
plt.xlim([1, 5000])
plt.ylim([0.1, 10 * np.ceil(np.log10(np.max(Tin_model)))])
plt.title("Temperature Fit")

plt.tight_layout()
plt.show()