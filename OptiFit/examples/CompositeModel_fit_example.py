import pickle
import matplotlib.pyplot as plt
from lmfit.models import LinearModel, LorentzianModel, VoigtModel, GaussianModel
from OptiFit import models as models



file = open(r'test_data.pkl', 'rb')
master_data = pickle.load(file)
file.close()
energies = 1240/master_data['1LBP PL doping']['Em_range'][0]
spectrum=master_data['1LBP PL doping']['spectrum'][109]


model = models.CompositeModel()
# add a linear background
params_dict = {'slope':{'value': 0.0, 'vary': True, 'min': -1, 'max': 1},
               'intercept':{'value': 0.03, 'vary': True, 'min': -500, 'max':500}
               }
model.add_component(LinearModel, params_dict, name='background_')
# add a Gaussian for the neutral exciton
params_dict = {'height':{'value': 3000, 'vary': True, 'min': 1, 'max': 6000},
               'center':{'value': 1.72, 'vary': True, 'min': 1.5, 'max': 2},
               'sigma':{'value': 0.03, 'vary': True, 'min': 0.001, 'max': 0.5}
               }
model.add_component(GaussianModel, params_dict, name='exciton_')
# add a gaussian for the lower energy feature
params_dict = {'height':{'value': 500, 'vary': True, 'min': 1, 'max': 6000},
               'center':{'value': 1.65, 'vary': True, 'min': 1.5, 'max': 1.7},
               'sigma':{'value': 0.05, 'vary': True, 'min': 0.001, 'max': 0.5}
                }
model.add_component(GaussianModel, params_dict, name='bullshit_')

# fit the data
params = model.Model.make_params()
result = model.Model.fit(spectrum, params, x=energies)

print(result.fit_report())
plt.close('all')
plt.figure()
plt.plot(energies, spectrum)
plt.plot(energies, result.best_fit, '-', label='best fit')
plt.legend()
plt.show()