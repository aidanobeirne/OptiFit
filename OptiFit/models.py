from xml.dom.minidom import Attr
from lmfit import Parameters, Minimizer, report_fit, Model
import matplotlib.pyplot as plt
from more_itertools import last
import numpy as np
import csv
from solcore.absorption_calculator.tmm_core_vec import coh_tmm
import os
import copy


def interpolate_domain(known_yvals, known_domain, desired_domain):
	""" interpolates a dataset within a domain

	Args:
		known_yvals (array or list): the y data points to be interpolated
		known_domain (array or list): the x data points that correspond to known_yvals
		desired_domain (array or list): the domain over which the data set will be interpolated

	Returns:
		(array): the interpolated domain
	"""
	known_domain = np.array(known_domain)
	desired_domain = np.array(desired_domain)
	minidx = np.where(known_domain == known_domain[known_domain < np.min(desired_domain)].max())[0][0]
	maxidx = np.where(known_domain == known_domain[known_domain > np.max(desired_domain)].min())[0][0] +1
	known_yvals = known_yvals[minidx:maxidx]
	known_domain = known_domain[minidx:maxidx]
	return np.interp(desired_domain, known_domain, known_yvals)

def import_rinfo(path):
	"""imports data from a csv that is downloaded from refractiveindex.info

	Args:
		path (str): path to csv

	Returns:
		tuple containing: energies, real part of the refractive index, imaginary part of the refractive index (k)
	"""
	with open(path, 'r') as f:
		reader = csv.reader(f, delimiter=',')
		headers = next(reader)
		rinfo = np.array(list(reader)).astype(float).T
		energies = rinfo[0]
		n = rinfo[1]
		try:
			k = rinfo[2]
			return energies, n, k
		except IndexError:
			return energies, n, 0*np.ones_like(n)
class TransferMatrixModel:
	""" Transfer Matrix model Class
	"""
	def __init__(self, spectrum, energies):
		self.spectrum = spectrum
		self.energies = energies
		self.params = Parameters()
		self.layers = {}
		self.peaks = []
	
	def complex_lorentzian(self, x, a, x0, w):
		denominator=(x0**2-x**2)**2+(x**2*w**2)
		real=a*(x0**2-x**2)/denominator
		imag=x*w*a/denominator
		lor=real+1j*imag
		return lor

	def add_lorentzian(self, name, amplitude, resonant_energy, broadening):
		self.params.add('{}_a'.format(name), *amplitude)
		self.params.add('{}_x0'.format(name), *resonant_energy)
		self.params.add('{}_w'.format(name), *broadening)
		self.peaks.append(name) # store the peaks in an instance attribute for easier access later

	def complex_voigt(self, x, a, x0, w, broadening):
		lor = self.complex_lorentzian(x, a, x0, w)
		g =np.exp((-(x - x0)**2) / (2 * broadening**2)) / broadening / np.sqrt(2 * np.pi) +1j * np.zeros_like(x)
		voigt = np.convolve(g, lor, mode = 'full')
		# chop off signal to be correct size
		center = abs(np.array(x) - x0).argmin()
		cutoff_min = int(center)
		cutoff_max = int(center + len(x))
		voigt_clipped = voigt[cutoff_min : cutoff_max]
		return voigt_clipped

	def add_voigt(self, name, amplitude, resonant_energy, broadening, inhomogeneous_broadening):
		if '_voigt' not in name:
			name = name + '_voigt'
		self.params.add('{}_a'.format(name), *amplitude)
		self.params.add('{}_x0'.format(name), *resonant_energy)
		self.params.add('{}_w'.format(name), *broadening)
		self.params.add('{}_broadening'.format(name), *inhomogeneous_broadening)
		self.peaks.append(name)

	def psuedovoigt(self, x, a, x0, w, mixing):
		## weighted sum of gaussian and lorentzian
		denominator = (x0**2 - x**2)**2 + (x**2 * w**2)
		real = (x0**2 - x**2) / denominator
		imag = x * w / denominator
		sigma = w / (2 * np.sqrt(2*np.log(2)))
		gauss = np.exp(-(x - x0)**2 / (2 * sigma**2))
		weighted_sum_real = mixing * gauss +(1 - mixing) * real
		weighted_sum_imag = mixing * gauss +(1 - mixing) * imag
		return a * (weighted_sum_real + 1j * weighted_sum_imag)

	def add_psuedovoigt(self, name, amplitude, resonant_energy, broadening, mixing):
		if '_psuedovoigt' not in name:
			name = name + '_psuedovoigt'
		self.params.add('{}_a'.format(name), *amplitude)
		self.params.add('{}_x0'.format(name), *resonant_energy)
		self.params.add('{}_w'.format(name), *broadening)
		self.params.add('{}_mixing'.format(name), *mixing)
		self.peaks.append(name)

	def add_background(self, func, params_dict, name='background'):
		# Background NOT from sample
		if name[-1] != '_':
			name += '_'
		for parameter, pdict in params_dict.items():
			self.params.add('{}{}'.format(name, parameter), **pdict)
		self.background = func
		self.background_name = name

	def add_background_eps(self, func, params_dict, name='background_eps'):
		# Background dielectric function of sample
		if name[-1] != '_':
			name += '_'
		for parameter, pdict in params_dict.items():
			self.params.add('{}{}'.format(name, parameter), **pdict)
		self.background_eps = func
		self.background_eps_name = name

	def add_layer(self, name, d, n):
		graphite_words = ['graphene', 'Graphene', 'graphite', 'Graphite', 'gr', 'Gr']
		fused_silica_words = ['SiO2', 'sio2', 'oxide', 'silica']
		crystalline_quartz_words = ['quartz', 'Quartz']
		silicon_words = ['Silicon', 'silicon', 'Si', 'si']
		air_words = ['air', 'Air', 'Empty', 'empty', 'vacuum']
		hbn_words = ['hBN', 'hbn', 'HBN']

		path = 'none'
		if 'full' in name:
			n[1] = False # set vary to False in case it was accidentally set to True
			if any(word in name for word in graphite_words):
				path = os.path.join(os.path.dirname(__file__), 'refractive_index_data', 'graphite(o)_refractive_index_info-eV.csv')
			elif any(word in name for word in fused_silica_words):
				path = os.path.join(os.path.dirname(__file__), 'refractive_index_data', 'quartz_refractive_index_info-eV.csv')
			elif any(word in name for word in silicon_words):
				path = os.path.join(os.path.dirname(__file__), 'refractive_index_data', 'silicon_refractive_index_info-eV.csv')
			elif any(word in name for word in crystalline_quartz_words):
				if '1951' in name:
					path = os.path.join(os.path.dirname(__file__), 'refractive_index_data', 'crystalline_quartz(o)_1951-eV.csv')
				else:
					path = os.path.join(os.path.dirname(__file__), 'refractive_index_data', 'crystalline_quartz(o)_1999-eV.csv')
			elif any(word in name for word in air_words):
				path='custom'
				n_custom = np.ones_like(self.energies)
			elif any(word in name for word in hbn_words):
				path = 'custom'
				wls = 1240/self.energies
				if '2018' in name:
					path = os.path.join(os.path.dirname(__file__), 'refractive_index_data', 'hBN_refractive_index_info_2018-eV.csv') # Phys. Status Solidi B, 256, 1800417 (2018) -- actually same as '2019' but imported
				elif '2019' in name:
					n_custom = np.sqrt(1+3.263*wls**2/(wls**2 - (164.4)**2)) + 1j*0 # Phys. Status Solidi B 2019, 256, 1800417 ----------------- much better in NIR
				else:
					n_custom = 2.23 - 6.9e-4*wls + 1j*0 # apparently in O.  Stenzel et al., Phys.  Status  Solidi  A (1996), but taken from Kim et al. JOSK 19, 503 (2015) -------------------- much better in VIS    

		if 'csv' in path:
			energies, nr, k = import_rinfo(path)
			n_interpolated = interpolate_domain(nr, energies, self.energies)
			k_interpolated = interpolate_domain(k, energies, self.energies)
			n[0] = n_interpolated + 1j*k_interpolated
			self.layers[name] = {'n': n[0], 'd': d[0], 'vary': False, 'imported model': True} # these are refractive indices with pre-existing models and we will not vary them in the fit
		elif 'custom' in path:
			n[0] = n_custom
			self.layers[name] = {'n': n[0], 'd': d[0], 'vary': False, 'imported model': True} 
		elif 'none' in path and 'sample' in name:
			n[0] = self.calc_n_sample()
			self.layers[name] = {'n': n[0], 'd': d[0], 'vary': False, 'imported model': True} 
		# this clause means that we actually want to vary the refractive index as a fit parameter (hopefully this does not occur often, as it only works for constant values of nreal and nimag)
		elif 'none' in path and 'sample' not in name:
			if n[0] is complex:
				nreal = [float(np.real(n[0])), n[1], np.real(n[2]), np.real(n[3])]
				nimag = [float(np.imag(n[0])), n[1], np.imag(n[2]), np.imag(n[3])]
				self.params.add('{}_nreal'.format(name), *nreal)
				self.params.add('{}_nimag'.format(name), *nimag)
				self.layers[name] = {'nreal': nreal, 'nimag': nimag, 'd': d, 'vary': n[1], 'imported model': False} # these refractive indices behave as fit parameters
			else:
				self.params.add('{}_n'.format(name), *n)
				self.layers[name] = {'n': float(n[0]), 'd': d, 'vary': n[1], 'imported model': False} 
		
		# always add the thicknesses to the parameters object
		self.params.add('{}_d'.format(name), *d)

	def get_nvals(self, includesample, exclude_background=False, exclude_peaks=False):
		# Function retrieves the refractive indices, n from the params object, for every layer in the stack
		nvals = []
		for layer, value in self.layers.items():
			if 'sample' in layer and includesample:
				nvals.append(self.calc_n_sample(exclude_background=exclude_background, exclude_peaks=exclude_peaks))
			elif 'sample' in layer and not includesample:
				continue
			else:
				if value['imported model'] == True: 
					nvals.append(self.layers[layer]['n']) # pre-existing model stored in the layer dictionary
				else:
					try:
						nreal = self.params['{}_nreal'.format(layer)].value*np.ones_like(self.energies)
						nimag = self.params['{}_imag'.format(layer)].value*np.ones_like(self.energies)
						nvals.append(nreal+nimag*1j) # fit parameter stored in the params object
					except KeyError:
						nvals.append(self.params['{}_n'.format(layer)].value*np.ones_like(self.energies)) # fit parameter stored in the params object
		return nvals

	def get_dvals(self, includesample):
		# Function retrieves the thicknesses, d from the params object, for every layer in the stack
		dvals = []
		for layer in self.layers.keys():
			if 'sample' in layer and not includesample:
				continue
			dvals.append(self.params['{}_d'.format(layer)].value)
		return dvals

	def calc_n_sample(self, exclude_background=False, exclude_peaks=False):

		eps_sample = np.ones_like(self.energies)
		if hasattr(self, 'background_eps') and not exclude_background:
			eps_sample = eps_sample + self.calc_bg_eps()

		# if hasattr(self, 'background') and not exclude_background:
		# 	eps_sample = eps_sample + self.calc_bg()
		
		if not exclude_peaks:
			for peak in self.peaks:
				if '_voigt' in peak:
					eps_sample = eps_sample + self.complex_voigt(self.energies, 
																self.params['{}_a'.format(peak)].value, 
																self.params['{}_x0'.format(peak)].value, 
																self.params['{}_w'.format(peak)].value, 
																self.params['{}_broadening'.format(peak)].value
																)
				elif 'psuedovoigt' in peak:
					eps_sample = eps_sample + self.psuedovoigt(self.energies, 
																self.params['{}_a'.format(peak)].value, 
																self.params['{}_x0'.format(peak)].value, 
																self.params['{}_w'.format(peak)].value, 
																self.params['{}_mixing'.format(peak)].value
																)
				else:
					eps_sample = eps_sample + self.complex_lorentzian(self.energies, self.params['{}_a'.format(peak)].value, self.params['{}_x0'.format(peak)].value, self.params['{}_w'.format(peak)].value)
						
		return np.sqrt(eps_sample)

	def calc_n_sample_resolved(self, exclude_background=False):
		ns_dict = {}
		ns_dict['total'] = self.calc_n_sample()
		for peak in self.peaks:
			eps_sample = np.ones_like(self.energies)
			if '_voigt' in peak:
				eps_sample = eps_sample + self.complex_voigt(self.energies, 
															self.params['{}_a'.format(peak)].value, 
															self.params['{}_x0'.format(peak)].value, 
															self.params['{}_w'.format(peak)].value, 
															self.params['{}_broadening'.format(peak)].value
															)
			elif 'psuedovoigt' in peak:
				eps_sample = eps_sample + self.psuedovoigt(self.energies, 
															self.params['{}_a'.format(peak)].value, 
															self.params['{}_x0'.format(peak)].value, 
															self.params['{}_w'.format(peak)].value, 
															self.params['{}_mixing'.format(peak)].value
															)
			else:
				eps_sample = eps_sample + self.complex_lorentzian(self.energies, self.params['{}_a'.format(peak)].value, self.params['{}_x0'.format(peak)].value, self.params['{}_w'.format(peak)].value)
			if not exclude_background:
				eps_sample = eps_sample + self.calc_bg()
			ns_dict[peak] = np.sqrt(eps_sample)
		return ns_dict


	def tmm_calc(self, includesample, exclude_background=False, exclude_peaks=False):
		if includesample:
			return coh_tmm('s', self.get_nvals(includesample=True, exclude_background=exclude_background, exclude_peaks=exclude_peaks), self.get_dvals(includesample=True), 0, 1240/self.energies)
		else:
			return coh_tmm('s', self.get_nvals(includesample=False, exclude_background=exclude_background, exclude_peaks=exclude_peaks), self.get_dvals(includesample=False), 0, 1240/self.energies)

	def calc_rc(self, exclude_background=False, exclude_peaks=False):
		target = self.tmm_calc(includesample=True, exclude_background=exclude_background, exclude_peaks=exclude_peaks)['R']
		ref = self.tmm_calc(includesample=False, exclude_background=exclude_background, exclude_peaks=exclude_peaks)['R']
		try:
			if not exclude_background:
				RC = (target - ref) / ref + self.calc_bg()
			else:
				RC = (target - ref) / ref
		except AttributeError:
			RC = (target - ref) / ref
		return RC
	
	def calc_rc_resolved(self):
		RC_dict = {}
		RC_dict['total'] = self.calc_rc()
		for peak in self.peaks:
			if '_voigt' in peak:
				eps_sample = np.ones_like(self.energies) + self.complex_voigt(self.energies, 
															self.params['{}_a'.format(peak)].value, 
															self.params['{}_x0'.format(peak)].value, 
															self.params['{}_w'.format(peak)].value, 
															self.params['{}_broadening'.format(peak)].value
															)
			elif 'psuedovoigt' in peak:
				eps_sample = np.ones_like(self.energies) + self.psuedovoigt(self.energies, 
															self.params['{}_a'.format(peak)].value, 
															self.params['{}_x0'.format(peak)].value, 
															self.params['{}_w'.format(peak)].value, 
															self.params['{}_mixing'.format(peak)].value
															)
			else:
				eps_sample = np.ones_like(self.energies) + self.complex_lorentzian(self.energies, self.params['{}_a'.format(peak)].value, self.params['{}_x0'.format(peak)].value, self.params['{}_w'.format(peak)].value)

			n_sample = np.sqrt(eps_sample)
			target_nvals = self.get_nvals(includesample=True)
			# find and replace the sample n with the n containing only one peak
			for idx, key in enumerate(self.layers.keys()):
				if 'sample' in key:
					target_nvals[idx] = n_sample
			target = coh_tmm('s', target_nvals, self.get_dvals(includesample = True), 0 + 1j*0, 1240/self.energies)['R']
			ref = coh_tmm('s', self.get_nvals(includesample = False), self.get_dvals(includesample = False), 0 + 1j*0, 1240/self.energies)['R']
			RC_dict[peak] = (target-ref)/ref
		return RC_dict

	def calc_bg(self):
		args = {}
		for param in self.params:
			if self.background_name in param and 'eps' not in param:
				args[param.split('_')[1]] = self.params[param].value
		bg = self.background(energies=self.energies, **args)
		return  bg 

	def calc_bg_eps(self):
		args = {}
		for param in self.params:
			if self.background_eps_name in param:
				args[param.split('_')[-1]] = self.params[param].value
		bg_eps = self.background_eps(energies=self.energies, **args)
		return  bg_eps 

	def loss(self, p):
		self.params = p 
		# voigt = False
		# for peak in self.peaks:
		# 	if '_voigt' in peak:
		# 		voigt = True
		# if voigt: #chop off part of the spectrum due to convolution artifact
		# 	return (np.array(self.calc_rc())-np.array(self.spectrum))[int(len(self.spectrum) / 11) : int(10 * len(self.spectrum) / 11)]
		# else:
		return np.array(self.calc_rc())-np.array(self.spectrum) 
	
	def loss_derivative(self, p):
		self.params = p
		# Compute the derivatives
		deriv_calc_rc = np.gradient(self.calc_rc(), self.energies)
		deriv_spectrum = np.gradient(self.spectrum, self.energies)
		return deriv_calc_rc - deriv_spectrum

	def weighted_loss(self, p):
		self.params = p 
		return np.array(self.calc_rc())-np.array(self.spectrum) * self.weights

	def fit(self, weights=None, method='leastsq', **kwargs):
		# self.verify_params()
		if weights is not None:
			self.weights = weights
			A = Minimizer(self.weighted_loss, self.params, nan_policy='omit', **kwargs)
		else:
			A = Minimizer(self.loss, self.params, nan_policy='omit', **kwargs)
		self.result = A.minimize(method=method)
		return self.result

	def fit_derivative(self, method='leastsq', **kwargs):
		A = Minimizer(self.loss_derivative, self.params, nan_policy='omit', **kwargs)
		self.result = A.minimize(method=method)
		return self.result
	
	def verify_params(self):
		for param in self.params:
			minval = self.params[param].min 
			val = self.params[param].value 
			maxval = self.params[param].max
			if (minval == val or maxval == val) and val != np.inf:
				raise ValueError('{} has an initial guess that is not within the bounds'.format(param))
			else:
				pass

	def update_params_initial_guess(self):
		for pname in self.params:
			self.params[pname].init_value = self.params[pname].value
	
	def plot_n_of_every_layer(self):
		ns = self.get_nvals(includesample=False)
		fig, ax = plt.subplots(2,1)
		ax[0].set_title("Re[$\epsilon$]")
		ax[1].set_title("Im[$\epsilon$]")
		layers = [i for i in self.layers.keys() if 'sample' not in i]
		for layer, n, in zip(layers, ns):
			ax[0].plot(self.energies, np.real(n), label=str(layer))
			ax[1].plot(self.energies, np.imag(n), label=str(layer))
		ax[0].legend()
		ax[1].legend()

	def plot_fit_result(self, **kw):
		self.fit_opt = {  # default options
			'derivative': False, 
			'eps_r': True,
			'title':            'Sample',  # plot title
		}
		for key, value in kw.items():
			self.fit_opt[key] = value

		if self.fit_opt['derivative'] == True:
			spec = np.gradient(self.spectrum, self.energies)
			spec_fit = np.gradient(self.calc_rc(), self.energies)
		else:
			spec = self.spectrum
			spec_fit = self.calc_rc()

		fig, ax = plt.subplots(2, 1)
		ax[0].plot(self.energies, spec, marker='.', markersize=2,
				linestyle="None", color='0', label='Raw Data')
		ax[0].plot(self.energies, spec_fit, color='C0', label='Fit Data')
		ax[0].set_title(self.fit_opt['title'])
		ax[0].set_ylabel('$\Delta R/R$')
		ax[0].legend()
		eps = self.calc_n_sample(exclude_background=True)**2
		ax[1].plot(self.energies, np.imag(eps),
				color='C1', label='$\epsilon_i$ of fit')
		if self.fit_opt['eps_r']:
			ax[1].plot(self.energies, np.real(eps),
					color='C2', label='$\epsilon_r$ of fit')
		for peak, value in self.calc_n_sample_resolved(exclude_background=False).items():
			bg = np.zeros_like(self.energies)
			if self.background_name in peak:
				bg = value
		ns = self.calc_n_sample_resolved(exclude_background=True)
		for n, (peak, value) in enumerate(ns.items()):
			if 'total' in peak:
				continue
			ax[1].plot(self.energies, np.imag(value**2) + np.imag(bg**2), linestyle='dashed', color=f'C{n+2}', label=peak)
			ax[1].fill_between(self.energies, np.imag(value**2) + np.imag(bg**2), color=f'C{n+2}', alpha=0.25)
		ax[1].legend()
		ax[1].set_xlabel('Energy (eV)')

class GuessPlot:
	def __init__(self):
		self.model = None
		self.guess_fig_id = None
		self.spec = None
		self.spec_fit = None

	def plot_guess(self, **kw):
		self.guess_opt = {  # default options
			'derivative': False, 
		}
		for key, value in kw.items():
			self.guess_opt[key] = value
		if self.guess_opt['derivative'] == True:
			self.spec = np.gradient(self.model.spectrum, self.model.energies)
			self.spec_fit = np.gradient(self.model.calc_rc(), self.model.energies)
		else:
			self.spec = self.model.spectrum
			self.spec_fit = self.model.calc_rc()
			

		if  self.guess_fig_id is not None and plt.fignum_exists(self.guess_fig_id):  # If the figure exists and was not closed
			# print('already exists')
			self.guess_fig = plt.figure(self.guess_fig_id)  # Get the existing figure
			self.guess_raw_line.set_ydata(self.spec)  # Update the data of the line object
			self.guess_pred_line.set_ydata(self.spec_fit)
		else:  # If the figure does not exist or was closed, create it
			# print('fig doesnt exist or was exited')
			self.guess_fig, self.guess_ax = plt.subplots()
			self.guess_fig_id = self.guess_fig.number  # Save the figure ID
			self.guess_raw_line, = self.guess_ax.plot(self.model.energies, self.spec, 'k+', label='spectrum')
			self.guess_pred_line, = self.guess_ax.plot(self.model.energies, self.spec_fit, 'r', label='initial guess')

		self.guess_ax.relim()
		self.guess_ax.autoscale_view(True, True, True)
		plt.draw()
		plt.pause(0.01)

	
			
class CompositeModel():
	"""Used to fit PL data with a variable number of peaks
	"""
	def __init__(self):
		self.components = {}

	def add_component(self, func_or_model, params_dict, name='peak1'):
		if name[-1] != '_':
			name += '_'
		mod = func_or_model(prefix=name)
		for parameter, pdict in params_dict.items():
			mod.set_param_hint('{}{}'.format(name, parameter), **pdict)
		self.components[name] = mod # internal reference to the specific component
		try:
			self.Model += mod # add component to the model
		except AttributeError:
			self.Model = mod

	def fit(self, data, params=None, weights=None, method='leastsq', iter_cb=None, scale_covar=True, verbose=False, fit_kws=None, nan_policy=None, calc_covar=True, max_nfev=None, **kwargs):
		result = self.Model.fit(data, params, weights, method, iter_cb, scale_covar, verbose, fit_kws, nan_policy, calc_covar, max_nfev, **kwargs)
		self.result = result
		return result

	def plot_fit_resolved(self, energies, data):
		plt.figure()
		plt.plot(energies, data, label='data', marker = 'o', color='black', markersize=1, linewidth=0.5)
		
		for component_name, model in self.components.items():
			params = {}
			for name in model.param_names:
				# find this parameter value in the instance Model and assign it to the parameter here
				params[name.removeprefix(component_name)] = self.result.params[name].value
			spec = model.eval(**params, x=energies)
			plt.plot(energies, spec, label = component_name[:-1])
		plt.legend()
	
	def plot_fit_color_coded(self, energies, data):
		from matplotlib.collections import LineCollection
		from matplotlib.colors import to_rgb, rgb_to_hsv
		from matplotlib.lines import Line2D
		fig, ax = plt.subplots()
		ax.plot(energies, data, label='data', marker = 'o', color='black', markersize=1, linewidth=0.5, zorder=0)

		curves = []
		curvenames = []
		colors = []
		# retrieve each curve
		for count, (component_name, model) in enumerate(self.components.items()):
			params = {}
			for name in model.param_names:
				# find this parameter value in the instance Model and assign it to the parameter here
				params[name.removeprefix(component_name)] = self.result.params[name].value
			spec = model.eval(**params, x=energies)
			curves.append(spec)
			colors.append(to_rgb('C{}'.format(count)))
			curvenames.append(name.split('_')[0])

		curve_to_plot = np.argmax(np.array(curves).T, axis=1)
		curve_changes = np.where(curve_to_plot[:-1] != curve_to_plot[1:])[0]
		segment_names = []
		segments = []
		segment_colors = []

		for count, changeidx in enumerate(curve_changes):
			curveidx = curve_to_plot[changeidx-1]
			if count == 0:
				x = np.array(energies[0:changeidx+1])
				y = np.array(self.result.best_fit[0:changeidx+1])
			else:
				x = np.array(energies[curve_changes[count-1]:changeidx+1])
				y = np.array(self.result.best_fit[curve_changes[count-1]:changeidx+1])
			
			new_segment = np.column_stack((x,y))
			segments.append(new_segment)
			segment_colors.append(colors[curveidx])
			segment_names.append(curvenames[curveidx])
			
			if count == len(curve_changes)-1: # need to add the last curve
				curveidx = curve_to_plot[changeidx+1]
				x = np.array(energies[changeidx:-1])
				y = np.array(self.result.best_fit[changeidx:-1])
				new_segment = np.column_stack((x,y))
				segments.append(new_segment)
				segment_colors.append(colors[curveidx])
				segment_names.append(curvenames[curveidx])
				
		line_segments = LineCollection(segments, colors=segment_colors, linewidths=3)
		proxies = []
		for color in colors:
			proxy = Line2D([0, 1], [0, 1], color=color)
			proxies.append(proxy)
		ax.legend(proxies, curvenames)
		ax.add_collection(line_segments)
		plt.show()

