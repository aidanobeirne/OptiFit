#%%
import numpy as np
import pickle
import matplotlib.pyplot as plt
from lmfit import report_fit
from OptiFit import models as models

############################################################################ Loading data, ignore this ################################################
def shift_correction_range(spectra, energies, e_min, e_max):
    temp = []
    idx_max = min(range(len(energies)), key=lambda i: abs(energies[i]-e_min))
    idx_min = min(range(len(energies)), key=lambda i: abs(energies[i]-e_max))
    for spec in spectra:
        offset = np.mean(spec[idx_min:idx_max])
        shift = 0 - offset
        temp.append(np.array(spec)+shift)
    return temp

file = open(r'test_data.pkl', 'rb')
master_data = pickle.load(file)
file.close()

identifier = '1LBP RC doping'
sorter = 'n'
energies = 1240/np.array(master_data[identifier]['Em_range'][0])
spectra = shift_correction_range(master_data[identifier]['RC'], energies, 1.372, 1.38)
############################################################################ Done loading data ################################################


############################################################################ TMM calculation with fit below ################################################

# Step 1 is to instantiate the model by feeding it the RC spectrum and the energy domain
model = models.TransferMatrixModel(spectrum=spectra[50], energies=energies)
"""
Step 2 is to add any resonances that we may predict to be in the spectrum. Firs we define the arguments to the function
in the 'pars' dictionary. Here, each key is the argument, and each value is an LMFIT parameter list (see source code docstrings).
The structure of an LMFIT parameter list is as follows: [value, vary, min, max]
                                            value - initial guess
                                            vary - do you want to treat this as a fit parameter? (bool)
                                            min - lower bound in fit
                                            max - upper bound in fit
Here we add two lorentzians, one for the ground state exciton and one for the excited state exciton.
"""
pars_1s = { 'f': [2., True, 0, 5000], # oscillator strength
            'E0': [1.7, True, 1.680, 1.90], # center resonance
            'gamma': [0.030, True, 0.005, 0.05], # broadening
            }
model.add_resonance(peak_name='exciton_1s', fn_name='lorentzian', **pars_1s)

pars_2s = { 'f': [0.3, True, 0, 5000], # oscillator strength
            'E0': [1.78, True, 1.75, 1.85], # center resonance
            'gamma': [0.040, True, 0.005, 0.075], # broadening
            }
model.add_resonance(peak_name='exciton_2s', fn_name='lorentzian', **pars_2s)


"""
Step 2 is to add each layer in the stack, sequentially, from top to bottom. The first list is the thickness parameter list 
(see step 2 to learn what an LMFIT parameter list is), the second list is the refractive index parameter list. If 'full' is in the 
name of the layer and any keywords (defined in the source code docstrings) are in the name of the layer, then the parameter list 
corresponding to 'n' will be ignored (but should still be included when using the function). In this case, the code will import dielectric 
functions from known publications and interpolate them across the energy domain.
"""
model.add_layer('air',    [np.inf, False, 0, np.inf], [1, False, 0, np.inf, None])
model.add_layer('top_graphite_full',   [4.7, False, 0.3, 5],    [1, False, 0, np.inf])
model.add_layer('top_hbn_full',   [18, False, 9, 20],      [3.9, False, 0, 5.3, None, None])
model.add_layer('sample', [0.7, False, 0.1, 2.7],     [1, True, 0, np.inf, None, None])
model.add_layer('bottom_hbn_full',   [13, False, 10, 25],      [3.9, False, 0, 5.3, None, None])
model.add_layer('bottom_graphite_full',   [3.2, False, 0.3, 58.1],    [1, False, 0, np.inf])
model.add_layer('quartz_full', [np.inf, False, 0, np.inf], [1.455, False, 1.440, 1.459, None, None])


"""
Step 3 is to define any background function. there is a lot to say here, see source code for more info. Generally speaking
it is better to include a higher order resonance to account for any background.
"""

def bg(energies, **kwargs):
    value = kwargs['a'] + kwargs['b'] * (energies - kwargs['b0'])
    return value
varlim = 500
varmin = 1.3
varmax = 2.2
var0init = 1.388
pdict = {'a':  {'value': -0.029, 'vary': True, 'min': -varlim*10, 'max': varlim*10},
         'b':  {'value': 0.141, 'vary': True, 'min': -varlim, 'max': varlim},
         'b0': {'value': var0init, 'vary': True, 'min': varmin, 'max': varmax}  
         }
model.add_background(func=bg, params_dict=pdict)


"""
Step 4 is to fit the data
"""
model.verify_params()
initial_guess = model.calc_rc()
result = model.fit()
result_spec = model.calc_rc()

"""
Step 5 is to plot the fit
"""
plt.figure()
plt.plot(model.energies, model.spectrum, 'k+')
plt.plot(model.energies, model.calc_rc(), 'r')
plt.close('all')
fig, ax = plt.subplots(2,1)
ax[0].plot(model.energies, model.spectrum, marker='.', markersize=2, linestyle="None", color='0', label='Raw Data')
ax[0].plot(model.energies, initial_guess, '--', color='C0', label='Inital Guess')
ax[0].plot(model.energies, result_spec, color='C1', label='Fit Data')
ax[0].set_title('RC Fit example')
ax[0].set_ylabel('$\Delta R/R$')
ax[0].legend()
eps = model.calc_n_sample()**2
ax[1].plot(model.energies, np.imag(eps), color = 'C2', label = '$\epsilon_i$ of fit')
ax[1].plot(model.energies, np.real(eps), color = 'C3', label = '$\epsilon_r$ of fit')
ax[1].legend()
ax[1].set_xlabel('Energy (eV)')

report_fit(result)


