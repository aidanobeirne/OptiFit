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
                    [1.78, True, 1.75, 1.85],
                    [0.040, True, 0.005, 0.075])


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


model.fit(method='least_squares')
plt.figure()
plt.plot(model.energies, model.spectrum, 'k+')
plt.plot(model.energies, model.calc_rc(), 'r')

model.report_fit()



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

#%%
import matplotlib.pyplot as plt
import numpy as np

import lmfit

x = np.linspace(1, 10, 250)
np.random.seed(0)
y = 3.0*np.exp(-x/2) - 5.0*np.exp(-(x-0.1)/10.) + 0.1*np.random.randn(x.size)

p = lmfit.Parameters()
p.add_many(('a1', 4.), ('a2', 4.), ('t1', 3.), ('t2', 3.))


def residual(p):
    return p['a1']*np.exp(-x/p['t1']) + p['a2']*np.exp(-(x-0.1)/p['t2']) - y


# create Minimizer
mini = lmfit.Minimizer(residual, p, nan_policy='propagate')

# first solve with Nelder-Mead algorithm
out1 = mini.minimize(method='Nelder')

# then solve with Levenberg-Marquardt using the
# Nelder-Mead solution as a starting point
out2 = mini.minimize(method='leastsq', params=out1.params)

lmfit.report_fit(out2.params, min_correl=0.5)

ci, trace = lmfit.conf_interval(mini, out2, sigmas=[1, 2], trace=True)
lmfit.printfuncs.report_ci(ci)

# plot data and best fit
plt.figure()
plt.plot(x, y)
plt.plot(x, residual(out2.params) + y, '-')
plt.show()

# plot confidence intervals (a1 vs t2 and a2 vs t2)
fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
cx, cy, grid = lmfit.conf_interval2d(mini, out2, 'a1', 't2', 30, 30)
ctp = axes[0].contourf(cx, cy, grid, np.linspace(0, 1, 11))
fig.colorbar(ctp, ax=axes[0])
axes[0].set_xlabel('a1')
axes[0].set_ylabel('t2')

cx, cy, grid = lmfit.conf_interval2d(mini, out2, 'a2', 't2', 30, 30)
ctp = axes[1].contourf(cx, cy, grid, np.linspace(0, 1, 11))
fig.colorbar(ctp, ax=axes[1])
axes[1].set_xlabel('a2')
axes[1].set_ylabel('t2')
plt.show()

# plot dependence between two parameters
fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
cx1, cy1, prob = trace['a1']['a1'], trace['a1']['t2'], trace['a1']['prob']
cx2, cy2, prob2 = trace['t2']['t2'], trace['t2']['a1'], trace['t2']['prob']

axes[0].scatter(cx1, cy1, c=prob, s=30)
axes[0].set_xlabel('a1')
axes[0].set_ylabel('t2')

axes[1].scatter(cx2, cy2, c=prob2, s=30)
axes[1].set_xlabel('t2')
axes[1].set_ylabel('a1')
plt.show()