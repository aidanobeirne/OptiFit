import numpy as np
import matplotlib.pyplot as plt
from lmfit import Minimizer, Parameters, report_fit
import tmm as tmm
import sys
sys.path.append('/Users/aidan/Documents/Python/class_lmfit_solcore/') ##### Set your path to the class file here
from lmfit_function_solcore import *
import glob
from scipy import fftpack
from scipy.signal import butter, lfilter
import os
rootdir = os.getcwd()



####### I needed to filter out some stuff in my raw data
def butter_bandstop_filter(data, lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    i, u = butter(order, [low, high], btype='bandstop')
    y = lfilter(i, u, data)
    return y

fs = np.around(1/abs(np.diff(energies).mean()))
dat = np.load('test_dat.npz')['spec']
filtered_dat = butter_bandstop_filter(dat,13.86,36.7,fs,2)[141:]
energies = np.load('test_dat.npz')['energies'][141:]
plt.close('all')

# # RC Fitting Using the LMFIT + Solcore modules
#  - Add however many peaks (Gaussian or Lorentzian) you expect in the dielectric function. Each peak should contain 'pk' in the name (first argument of the "add_lorentzian" method).
#  
#  - Each peak addition in the "add_lorentzian" method contains 3 lists that represent the fit parameters defined for the LMFIT module.
#  
#  - Each fit parameter is defined by a list whose values are [name, value, vary (bool), min, max]. The parameter ordering is oscillator strength, energy resonance, and linewidth.
#   
#  - You can add layers with the "add_layer" method. The first parameter of the "add_layer" method is the layer thickness, and the second is the refractive index.
#   
#  - The sample layer must include 'sample' in the name. The refractive index used in the sample doesn't matter but should be set to 0 as a placeholder. If you want to include the energy dependent refractive index for a layer include '_full' at the end of the layer name and set the 'Vary' for the nlist to False.


model = tmm_model(Parameters(), filtered_dat, energies)
model.add_lorentzian('pk1', model.energies,
                    [0.26885770, True, 0, np.inf],
                    [1.72186643, True, 0, 2],
                    [0.08191602, True, 0.05, 0.25])
model.add_lorentzian('pk2', model.energies,
                    [0.06885770, True, 0, np.inf],
                    [1.73186643, True, 1.72186643, 1.85],
                    [0.03191602, True, 0.02, 0.25])

########## Add layers for TMM calculation
model.add_layer('air',    [np.inf, False, 0, np.inf], [1, False, 0, np.inf, None])
model.add_layer('tophbnfull',   [10.87, False, 9, 15],      [2, False, 0, 5.3, None, None])
model.add_layer('sample', [0.3, False, 0.1, 2.7],     [1, True, 0, np.inf, None, None])
model.add_layer('bottomhbnfull',   [13.5, False, 10, 25],      [2, False, 0, 5.3, None, None])
model.add_layer('Grfull',   [1.96, False, 0.3, 2.1],    [1, False, 0, np.inf])
model.add_layer('quartz', [np.inf, False, 0, np.inf], [1.455, False, 1.440, 1.459, None, None])
#### Background dielectric function (not including the +1 for epsilon(inf))
model.params.add('eps_sample_r', 0, False, 2.0, 7)
model.params.add('eps_sample_r_slope', 0, False, -np.inf, np.inf)
model.params.add('eps_sample_i', 0, False, -0.5, 0)
model.params.add('eps_sample_i_slope', 0, False, -np.inf, np.inf)
#### Artifically add a linear background to the RC, could also fit to derivative
model.params.add('offset', 0.1, True, -np.inf, np.inf)
model.params.add('slope', 0, True, -np.inf, np.inf)

model.params.pretty_print()
plt.figure()
plt.plot(model.energies, model.data)
plt.plot(model.energies, model.calc_rc())
plt.title('Measured spectrum and initial guess');

########################################## Fit RC with initial guesses
model.ls_fit()
plt.close('all')
plt.figure()
plt.plot(model.energies, model.data, 'k+')
plt.plot(model.energies, model.calc_rc(), 'r')
# Report fit results
report_fit(model.result)

