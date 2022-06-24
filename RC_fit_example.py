import os
import sys
import time
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from TransferMatrix import models as models

parameters = {'axes.labelsize': 15,
              'axes.titlesize': 25,
              'figure.figsize' : (0.75*16,0.75*9)}
plt.rcParams.update(parameters)

def shift_correction_range(spectra, energies, e_min, e_max):
    temp = []
    idx_max = min(range(len(energies)), key=lambda i: abs(energies[i]-e_min))
    idx_min = min(range(len(energies)), key=lambda i: abs(energies[i]-e_max))
    for spec in spectra:
        offset = np.mean(spec[idx_min:idx_max])
        shift = 0 - offset
        temp.append(np.array(spec)+shift)
    return temp

file = open(r'TransferMatrix/test_data.pkl', 'rb')
master_data = pickle.load(file)
file.close()

identifier = '1LBP RC doping'
sorter = 'n'
energies = 1240/np.array(master_data[identifier]['Em_range'][0])
spectra = shift_correction_range(master_data[identifier]['RC'], energies, 1.372, 1.38)

model = models.tmm_model(spectrum=spectra[50], energies=energies)
model.add_lorentzian(name='pk1',
                    amplitude=[2, True, 0, np.inf],
                    resonant_energy=[1.7, True, 1.65, 1.75],
                    broadening=[0.030, True, 0.005, 0.4])
model.add_lorentzian('pk2',
                    [0.3, True, 0.2, 1],
                    [1.78, True, 1.77, 1.85],
                    [0.040, True, 0.005, 0.055])


########## Add layers for TMM calculation
model.add_layer('air',    [np.inf, False, 0, np.inf], [1, False, 0, np.inf, None])
model.add_layer('top_graphite_full',   [4.7, False, 0.3, 2.1],    [1, False, 0, np.inf])
model.add_layer('top_hbn_full',   [18, False, 9, 20],      [3.9, False, 0, 5.3, None, None])
model.add_layer('sample', [0.7, False, 0.1, 2.7],     [1, True, 0, np.inf, None, None])
model.add_layer('bottom_hbn_full',   [13, False, 10, 25],      [3.9, False, 0, 5.3, None, None])
model.add_layer('bottom_graphite_full',   [3.2, False, 0.3, 58.1],    [1, False, 0, np.inf])
model.add_layer('quartz', [np.inf, False, 0, np.inf], [1.455, False, 1.440, 1.459, None, None])

#### Background dielectric function (not including the +1 for epsilon(inf))
model.params.add('eps_sample_r', 0, False, 0, 7)
model.params.add('eps_sample_r_slope', 0, False, -np.inf, np.inf)
model.params.add('eps_sample_i', 0, False, -0.5, 1)
model.params.add('eps_sample_i_slope', 0, False, -np.inf, np.inf)
model.params.add('offset', 0, True, -np.inf, np.inf)
model.params.add('slope', 0, True, -np.inf, np.inf)

# model.params.pretty_print()
plt.close('all')
plt.figure()
plt.plot(model.energies, model.spectrum)
plt.plot(model.energies, model.calc_rc())
plt.title('Measured spectrum and initial guess');


model.fit()
plt.figure()
plt.plot(model.energies, model.spectrum, 'k+')
plt.plot(model.energies, model.calc_rc(), 'r')


# result_spec = model.calc_rc_resolved()
# fig, ax = plt.subplots(3,1)
# ax[0].plot(model.energies, model.data, 'k+')
# ax[0].plot(model.energies, result_spec['total'], 'r')
# ax[0].set_title('Monolayer RC Fit example')
# ax[0].set_ylabel('$\Delta R/R$')
# ax[1].plot(model.energies, model.data, 'k+')
# ax[1].plot(model.energies, result_spec['pk1'], label = 'pk1')
# ax[1].plot(model.energies, result_spec['pk2'], label = 'pk2');
# # ax[1].plot(model.energies, result_spec['pk3'], label = 'pk3');
# ax[1].legend()
# ax[2].set_xlabel('Energy (eV)')
# ax[1].set_ylabel('$\Delta R/R$');
# eps = model.calc_n_sample(model.energies)**2
# ax[2].plot(model.energies, np.imag(eps), color = 'C4', label = '$\epsilon_2$ of fit')
# ax[2].legend()
# # ax[1].plot(model.energies,result_spec[3], '--');
# # Report fit results
# report_fit(model.result)
# # print('Fit Time = '+ str(tf-t0))
# abs((model.result.residual)).mean()