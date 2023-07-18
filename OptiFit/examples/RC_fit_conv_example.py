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

model = models.TransferMatrixModel(spectrum=spectra[50], energies=energies)
"""
Here we model the 1s peak as the convolution between a Gaussian and Lorentzian (Voigt)
"""
def gaussian(x, mu, sigma):
    result = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma)**2)
    result = result/np.sum(result)
    return result

pars_1s = {'f': [2.59, True, 1.9, 5000],
        'E0': [1.685, True, 1.680, 1.90],
        'gamma': [0.0140, True, 0.005, 0.025],    
        'conv_fn': gaussian,
        'mu' : [model.energies[int((len(model.energies) - 1)/2)], False, 1.0, 1.90],
        'sigma': [0.01, True, 0.00005, 0.5]
        }
model.add_resonance(peak_name='exciton_1s', fn_name='convd_lorentzian', **pars_1s)


pars_2s = { 'f': [0.3, True, 0, 5000], # oscillator strength
            'E0': [1.78, True, 1.75, 1.85], # center resonance
            'gamma': [0.040, True, 0.005, 0.075], # broadening
            }
model.add_resonance(peak_name='exciton_2s', fn_name='lorentzian', **pars_2s)


model.add_layer('air',    [np.inf, False, 0, np.inf], [1, False, 0, np.inf, None])
model.add_layer('top_graphite_full',   [4.7, False, 0.3, 5],    [1, False, 0, np.inf])
model.add_layer('top_hbn_full',   [18, False, 9, 20],      [3.9, False, 0, 5.3, None, None])
model.add_layer('sample', [0.7, False, 0.1, 2.7],     [1, True, 0, np.inf, None, None])
model.add_layer('bottom_hbn_full',   [13, False, 10, 25],      [3.9, False, 0, 5.3, None, None])
model.add_layer('bottom_graphite_full',   [3.2, False, 0.3, 58.1],    [1, False, 0, np.inf])
model.add_layer('quartz_full', [np.inf, False, 0, np.inf], [1.455, False, 1.440, 1.459, None, None])


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


model.verify_params()
initial_guess = model.calc_rc()
result = model.fit()
result_spec = model.calc_rc()


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


