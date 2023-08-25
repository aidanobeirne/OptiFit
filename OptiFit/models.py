from xml.dom.minidom import Attr
from lmfit import Parameters, Minimizer, report_fit, Model
import matplotlib.pyplot as plt
from more_itertools import last
import numpy as np
import csv
import inspect
from solcore.absorption_calculator.tmm_core_vec import coh_tmm
import os
from scipy.integrate import quad_vec
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
	""" Transfer Matrix Model used to fit to reflectance contrast data from multilayer thin flims
	"""
	def __init__(self, spectrum, energies):
		"""Constructor for TMM model
		Args:
			spectrum (array or list): raw data spectrum
			energies (array or list): correspoinding energy domain
		"""
		self.spectrum = spectrum
		self.energies = energies
		self.params = Parameters()
		self.layers = {}
		self.peaks = {}
		
	@staticmethod
	def lorentzian(E, f, E0, gamma):
		"""Lorentzian spectral line

		Args:
			E (array): energies of spectrum
			f (float): oscillator strength
			E0 (float): center resonance
			gamma (float): broadening 

		Returns:
			complex array: real and imaginary parts of the complex Lorentzian 
		"""
		denominator = (E0**2 - E**2)**2 + (E**2*gamma**2)
		real = f*(E0**2 - E**2) / denominator
		imag = E*gamma*f / denominator
		lor = real + 1j*imag
		return lor
	
	@staticmethod
	def asymmetric_lorentzian(E, E0, gamma1, gamma2, f):
		"""Asymmetric Lorentzian spectral line (also called a split Lorentzian). The broadening factor is allowed to vary on either side
		of the center resonance

		Args:
			E (array): energies of spectrum
			E0 (float): center resonance
			gamma1 (float): broadening below center resonance
			gamma2 (float): broadening above center resonance
			f (float): oscillator strength


		Returns:
			complex array: real and imaginary parts of the asymmetric Lorentzian 

		"""
		result = np.empty_like(E, dtype=complex)
		condition = E <= E0
		E1 = E[condition]
		lor1 = TransferMatrixModel.lorentzian(E1, 1, E0, gamma1)
		normalization1 = abs(TransferMatrixModel.lorentzian(E0, 1, E0, gamma1))
		result[condition] = lor1 / normalization1
		E2 = E[~condition]
		lor2 = TransferMatrixModel.lorentzian(E2, 1, E0, gamma2)
		normalization2 = abs(TransferMatrixModel.lorentzian(E0, 1, E0, gamma2))
		result[~condition] = lor2 / normalization2
		result = f * result
		return result

	@staticmethod
	def convd_lorentzian(E, f, E0, gamma, conv_fn, **kwargs):
		"""Inhomogeneously broadened Lorentzian spectral line. A Lorentzian function is convolved with the user-input broadening function.
		The user should ensure that the broadening function is normalized. Note that there will be convolution artefacts at the boundary.

		Args:
			E (array): energies of spectrum
			f (float): oscillator strength
			E0 (float): center resonance
			gamma (float): broadening 
			conv_fn (function): function to convolve the Lorentzian with (e.g. a gaussian)
			kwargs (dict): dictionary of keyword arguments that correspond to the conv_fn

		Returns:
			complex array: real and imaginary parts of the inhomogeneously broadened Lorentzian 

		"""
		L = TransferMatrixModel.lorentzian(E, f, E0, gamma)
		conv_fn_array = conv_fn(E, **kwargs)
		real = np.convolve(np.real(L), conv_fn_array, mode='same')
		imag = np.convolve(np.imag(L), conv_fn_array, mode='same')
		return real + 1j * imag
		
	@staticmethod
	def psuedovoigt(E, f, E0, gamma, mixing):
		"""Psuedovoigt spectral line. Instead of doing a full convolution between a Gaussian and Lorentzian, this function
		is a weighted sum of a Gaussian and Lorentzian.

		Args:
			E (array): energies of spectrum
			f (float): oscillator strength
			E0 (float): center resonance
			gamma (float): broadening 
			mixing (float): number between 0 (completely Lorentzian) and 1 (completely Gaussian)

		Returns:
			complex array: real and imaginary parts of the psuedovoigt

		"""
		denominator = (E0**2 - E**2)**2 + (E**2 * gamma**2)
		real = (E0**2 - E**2) / denominator
		imag = E * gamma / denominator
		sigma = gamma / (2 * np.sqrt(2*np.log(2)))
		gauss = np.exp(-(E - E0)**2 / (2 * sigma**2))
		weighted_sum_real = mixing * gauss +(1 - mixing) * real
		weighted_sum_imag = mixing * gauss +(1 - mixing) * imag
		return f * (weighted_sum_real + 1j * weighted_sum_imag)
	
	def	add_resonance(self, peak_name, fn_name, **pars):
		"""Add a resonance to the model

		Args:
			peak_name (str): Name of resonance, e.g. '1s_exciton'
			fn_name (str): Name of function used to model resonance, e.g. 'lorentzian'. Look at all of the static methods to see your options
			pars (dict): Dictionary of parameter objects. See LMFIT documentation: https://lmfit.github.io/lmfit-py/parameters.html#
						Keys are arguments to correwsponding function defined by fn_name, values are lmfit parameter lists
						One entry of the pars dictionary could be, for example: 
						 							'E0': [value, vary, min, max]
						where 'value' is the initial guess, 'vary' (bool) is whether or not it should be treated as a fit parameter, 
						and 'min' and 'max' are the bounds of the parameter.

		"""
		if fn_name not in [func for func in dir(self) if callable(getattr(self, func)) and not func.startswith("__")]:
			raise ValueError(f"The function'{fn_name}' is not found.")
		else:
			function = getattr(self, fn_name)
			args = list(inspect.signature(function).parameters)[1:]

			for arg in args:
				if arg not in pars.keys() and 'kwarg' not in arg:
					raise ValueError(f"Parameter '{arg}' is missing.")
	
			# Add parameters to the params dictionary
		for key, value in pars.items():
			if 'convd' in fn_name and 'conv_fn' in key:
				self.conv_fn = value
				continue
			self.params.add(f'{peak_name}_{key}', *value)
		
		self.peaks[peak_name] = function

	def add_background(self, func, params_dict, name='background'):
		"""Add a background to the RC spectrum. This is, stricly speaking, bad practice since the resulting model is no longer
			KKR constrained (i.e. one could violate causality with an appropriately whacky background function). However, this can 
			sometimes be necessary if the RC spectrum has some artefacts (e.g. the reference spectrum is bad)

		Args:
			func (function): user-defined background function
			params_dict (dict): dictionary of keyword arguments that correspond to the background function
			name (str, optional): background name. Defaults to 'background'.
		"""
		if name[-1] != '_':
			name += '_'
		for parameter, pdict in params_dict.items():
			self.params.add('{}{}'.format(name, parameter), **pdict)
		self.background = func
		self.background_name = name

	def add_background_eps(self, func, params_dict, name='background_eps'):
		"""Add a background function to the dielectric function of the sample. Again this is generally 
		   bad practice (see add_background method), but can sometimes be helfpul.

		Args:
			func (function): user-defined background dielectric function
			params_dict (dict): dictionary of keyword arguments that correspond to the background dielectric function
			name (str, optional): Background name. Defaults to 'background_eps'.
		"""
		if name[-1] != '_':
			name += '_'
		for parameter, pdict in params_dict.items():
			self.params.add('{}{}'.format(name, parameter), **pdict)
		self.background_eps = func
		self.background_eps_name = name

	def add_layer(self, name, d, n):
		"""Add a layer in the mutlilayer stack. If 'full' is in the name of the layer and any of the keywords (defined in the 'words' lists below)
		   are in the name of the layer, then the parameter list corresponding to 'n' will be ignored (but should still be included when using the function). 
		   In this case, the code will import dielectric functions from known publications and interpolate them across the energy domain.

		Args:
			name (str): Name of layer (e.g. 'hBN')
			d (list): LMFIT parameter list corresponding to the thickness of the layer (see add_resonance method or LMFIT documentation)
			n (list): LMFIT parameter list corresponding to the refractive index of the layer (see add_resonance method or LMFIT documentation)
		"""
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
		
			if isinstance(n[0], complex):
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

	def add_alternate_ref_d(self, name, d):
		"""Sometimes the reference spectrum is taken in a region where the thicknesses of each layer are slightly different. This function
		adds an additional thickness fit parameter to the model, which corresponds to the thickness of the layer in the reference structure.

		Args:
			name (str): Exact name of the layer with a different thickness in the ref structure
			d (list): LMFIT parameter list corresponding to the thickness of the layer (see add_resonance method or LMFIT documentation)
		"""
		self.layers[name]['d_alternate_ref'] = d
		self.params.add('{}_d_alternate_ref'.format(name), *d)

	def get_nvals(self, includesample, exclude_background=False, exclude_peaks=False):
		"""Retrieves the complex refractive indices of each layer as a function of energy

		Args:
			includesample (Bool): Whether or not to include the sample in the calculation (should be 'False' for reference spectrum)
			exclude_background (bool, optional): Whether or not to exclude any user-defined background function. Defaults to False.
			exclude_peaks (bool, optional): Whether or not to exclude the resonances (this allows one to visualize only the contribution from 
											a user-defined backgroun). Defaults to False.

		Returns:
			array of complex floats: array of refractive indices for each layer
		"""
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
						layer_trimmed = layer.replace('_n', '')
						nreal = self.params['{}_nreal'.format(layer_trimmed)].value*np.ones_like(self.energies)
						nimag = self.params['{}_nimag'.format(layer_trimmed)].value*np.ones_like(self.energies)
						nvals.append(nreal+nimag*1j) # fit parameter stored in the params object
					except KeyError:
						nvals.append(self.params['{}_n'.format(layer)].value*np.ones_like(self.energies)) # fit parameter stored in the params object
		return nvals

	def get_dvals(self, includesample):
		"""Retrives the thicknesses for each layer in the stack

		Args:
			includesample (Bool): Whether or not to include the sample in the calculation (should be 'False' for reference spectrum)

		Returns:
			array of floats: array of thicknesses
		"""
		dvals = []
		for layer in self.layers.keys():
			if 'sample' in layer and not includesample:
				continue
			if not includesample and 'd_alternate_ref' in self.layers[layer].keys():
				dvals.append(self.params['{}_d_alternate_ref'.format(layer)].value)
				continue
				
			dvals.append(self.params['{}_d'.format(layer)].value)
		return dvals

	def calc_n_sample(self, exclude_background=False, exclude_peaks=False):
		"""Calculates the complex refractive index of the sample for the current model

		Args:
			exclude_background (bool, optional): Whether or not to exclude any user-defined background function. Defaults to False.
			exclude_peaks (bool, optional): Whether or not to exclude the resonances (this allows one to visualize only the contribution from 
											a user-defined background). Defaults to False.

		Returns:
			array of complex floats: The complex refractive index of the sample
		"""
		eps_sample = np.ones_like(self.energies) + 1j * np.zeros_like(self.energies)

		if hasattr(self, 'background_eps') and not exclude_background:
			eps_sample += self.calc_bg_eps()

		if not exclude_peaks:
			for peakname, peakfunction in self.peaks.items():
				eps_sample += self.evaluate_resonance(peakname, peakfunction)
		return np.sqrt(eps_sample)


	def calc_n_sample_resolved(self, exclude_background=False):
		"""Calculates the refractive index of the sample for every resonance individually

		Args:
			exclude_background (bool, optional): Whether or not to exclude any user-defined background function. Defaults to False.


		Returns:
			dict: dictionary of refractive indices for each resonance. Keys are peak names, values are refractive indices
		"""
		ns_dict = {}
		ns_dict['total'] = self.calc_n_sample()
		for peakname, peakfunction in self.peaks.items():
			if 'background' in peakname:
				continue
			eps_sample = np.ones_like(self.energies) + 1j * np.zeros_like(self.energies)
			eps_sample += self.evaluate_resonance(peakname, peakfunction)
			if hasattr(self, 'background_eps') and not exclude_background:
				eps_sample += self.calc_bg_eps()

			ns_dict[peakname] = np.sqrt(eps_sample)

		return ns_dict


	def tmm_calc(self, includesample, exclude_background=False, exclude_peaks=False):
		"""Performs the TMM caluclation using the solcore package: https://www.solcore.solar/ . This returns the reflected wave power of light
		normally incident upon the multilayer stack

		Args:
			includesample (bool): include sample?
			exclude_background (bool, optional): exclude background? Defaults to False.
			exclude_peaks (bool, optional): exclude resonances? Defaults to False.

		Returns:
			array: Reflected wave power 
		"""
		if includesample:
			return coh_tmm('s', self.get_nvals(includesample=True, exclude_background=exclude_background, exclude_peaks=exclude_peaks), self.get_dvals(includesample=True), 0, 1240/self.energies)
		else:
			return coh_tmm('s', self.get_nvals(includesample=False, exclude_background=exclude_background, exclude_peaks=exclude_peaks), self.get_dvals(includesample=False), 0, 1240/self.energies)

	def calc_rc(self, exclude_background=False, exclude_peaks=False):
		"""Calculates the reflectance contrast of wave power of light normally incident upon the multilayer stack

		Args:
			exclude_background (bool, optional): exclude background? Defaults to False.
			exclude_peaks (bool, optional): exclude resonances? Defaults to False.

		Returns:
			array: The reflectance contrast of the multilayer stack
		"""
		target = self.tmm_calc(includesample=True, exclude_background=exclude_background, exclude_peaks=exclude_peaks)['R']
		ref = self.tmm_calc(includesample=False, exclude_background=exclude_background, exclude_peaks=exclude_peaks)['R']
		try:
			if not exclude_background:
				RC = (target - ref) / ref
				if hasattr(self, 'background_name'):
					RC += self.calc_bg()
			else:
				RC = (target - ref) / ref
		except AttributeError:
			RC = (target - ref) / ref
		return RC

	def evaluate_resonance(self, peakname, peakfunction):
		"""Evaluates the output of an individual resonance
		Args:
			peakname (str): user-assigned name of resonance (e.g. 'exciton_1s')
			peakfunction (str): associated function name used to model the resonance (e.g. 'lorentzian')

		Returns:
			_type_: _description_
		"""
		if 'conv' not in peakfunction.__name__:
			args = list(inspect.signature(peakfunction).parameters)[1:]
			pars = {arg: self.params['{}_{}'.format(peakname, arg)].value for arg in args}
			return peakfunction(self.energies, **pars)
		else:
			args = list(inspect.signature(peakfunction).parameters)[1:]
			args = [arg for arg in args if 'conv_fn' not in arg and 'kwargs' not in arg]
			convargs = list(inspect.signature(self.conv_fn).parameters)[1:] 
			args.extend(convargs)
			pars = {arg: self.params['{}_{}'.format(peakname, arg)].value for arg in args}
			return peakfunction(self.energies, conv_fn=self.conv_fn, **pars)

			
	def calc_bg(self):
		args = {}
		for param in self.params:
			if self.background_name in param and 'eps' not in param:
				args[param.split('_')[1]] = self.params[param].value
		bg = self.background(energies=self.energies, **args)
		return  bg 

	def calc_bg_eps(self):
		bg_eps = np.zeros_like(self.energies, dtype=complex)
		args = {}
		for param in self.params:
			if self.background_eps_name in param:
				args[param.split('_')[-1]] = self.params[param].value
		bg_eps += self.background_eps(energies=self.energies, **args)#.astype(bg_eps.dtype)
		return  bg_eps 

	def loss(self, p):
		self.params = p 
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
		self.verify_params()
		if weights is not None:
			self.weights = weights
			A = Minimizer(self.weighted_loss, self.params, nan_policy='omit', **kwargs)
		else:
			A = Minimizer(self.loss, self.params, nan_policy='omit', **kwargs)
		self.result = A.minimize(method=method)
		return self.result

	def fit_derivative(self, method='leastsq', **kwargs):
		self.verify_params()
		A = Minimizer(self.loss_derivative, self.params, nan_policy='omit', **kwargs)
		self.result = A.minimize(method=method)
		return self.result
	
	def verify_params(self):
		for param in self.params:
			minval = self.params[param].min 
			val = self.params[param].value 
			maxval = self.params[param].max
			if minval == val and val != np.inf:
				self.params[param].set(min=minval + 1e-20)
			if maxval == val and val != np.inf:
				self.params[param].set(max=minval - 1e-20)
			# if (minval == val or maxval == val) and val != np.inf:
			# 	raise ValueError('{} has an initial guess that is not within the bounds'.format(param))
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
			'legend': True,
			'title':  'Sample',  # plot title
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
		eps = self.calc_n_sample(exclude_background=False)**2
		ax[1].plot(self.energies, np.imag(eps),
				color='C1', label='$\epsilon_i$ of fit')
		if self.fit_opt['eps_r']:
			ax[1].plot(self.energies, np.real(eps),
					color='C2', label='$\epsilon_r$ of fit')
		for peak, value in self.calc_n_sample_resolved(exclude_background=False).items():
			bg = np.zeros_like(self.energies, dtype=complex)
			if hasattr(self, 'background_eps') and self.background_eps_name in peak:
				bg = value
		ns = self.calc_n_sample_resolved(exclude_background=True)
		for n, (peak, value) in enumerate(ns.items()):
			if 'total' in peak or 'background' in peak:
				continue
			ax[1].plot(self.energies, np.imag(value**2) + np.imag(bg**2), linestyle='dashed', color=f'C{n+2}', label=peak)
			ax[1].fill_between(self.energies, np.imag(value**2) + np.imag(bg**2), color=f'C{n+2}', alpha=0.25)
		if self.fit_opt['legend']:
			ax[1].legend()
		ax[1].set_xlabel('Energy (eV)')

class GuessPlot:
	def __init__(self):
		self.model = None
		self.guess_fig_id = None
		self.spec = None
		self.spec_fit = None
		self.params = None

	def plot_guess(self, **kw):
		self.guess_opt = {  # default options
			'derivative': False, 
		}
		for key, value in kw.items():
			self.guess_opt[key] = value
		if self.guess_opt['derivative'] == True:
			self.spec = np.gradient(self.model.spectrum, self.model.energies)
			if isinstance(self.model, TransferMatrixModel):
				self.spec_fit = np.gradient(self.model.calc_rc(), self.model.energies)
			else:
				self.spec_fit = np.gradient(self.model.eval(self.params), self.model.energies)

		else:
			self.spec = self.model.spectrum
			if isinstance(self.model, TransferMatrixModel):
				self.spec_fit = self.model.calc_rc()
			else:
				self.spec_fit = self.model.eval(self.params)
			
		if  self.guess_fig_id is not None and plt.fignum_exists(self.guess_fig_id):  # If the figure exists and was not closed
			self.guess_fig = plt.figure(self.guess_fig_id)  # Get the existing figure
			self.guess_raw_line.set_ydata(self.spec)  # Update the data of the line object
			self.guess_pred_line.set_ydata(self.spec_fit)
		else:  # If the figure does not exist or was closed, create it
			self.guess_fig, self.guess_ax = plt.subplots()
			self.guess_fig_id = self.guess_fig.number  # Save the figure ID
			self.guess_raw_line, = self.guess_ax.plot(self.model.energies, self.spec, 'k', label='spectrum')
			self.guess_pred_line, = self.guess_ax.plot(self.model.energies, self.spec_fit, 'r', label='initial guess')

		self.guess_ax.relim()
		self.guess_ax.autoscale_view(True, True, True)
		plt.draw()
		plt.pause(0.01)

	
			
class CompositeModel():
	"""Used to fit PL data with a variable number of peaks
	"""
	def __init__(self, spectrum=None, energies=None):
		self.spectrum = spectrum
		self.energies = energies
		self.result = None
		self.params = Parameters()
		self.components = {}

	def add_component(self, func_or_model, params_dict, name='peak1'):
		if name[-1] != '_':
			name += '_'
		
		if type(func_or_model) is Model:
			mod = func_or_model
			for parameter, pdict in params_dict.items():
				mod.set_param_hint('{}{}'.format(name, parameter), **pdict)
		else:
			independent_var = list(inspect.signature(func_or_model).parameters)[0]
			args = list(inspect.signature(func_or_model).parameters)[1:]
			mod = Model(func_or_model, prefix=name)
			for parameter, pdict in params_dict.items():
				mod.set_param_hint('{}{}'.format(name, parameter), **pdict)
		self.components[name] = mod # internal reference to the specific component
		try:
			self.Model += mod # add component to the model
		except AttributeError:
			self.Model = mod

	def fit(self, data, params=None, weights=None, method='leastsq', iter_cb=None, scale_covar=True, verbose=False, fit_kws=None, nan_policy=None, calc_covar=True, max_nfev=None, **kwargs):
		if params is None:
			self.params = self.Model.make_params()
		else:
			self.params = params
		self.result = self.Model.fit(data, self.params, weights, method, iter_cb, scale_covar, verbose, fit_kws, nan_policy, calc_covar, max_nfev, **kwargs)
		return self.result

	def eval(self, params=None, **kwargs):
		result = self.Model.eval(params, x=self.energies, **kwargs)
		return result

	def generate_curves(self):
		curves = {}
		for component_name, model in self.components.items():
			params = {}
			for name in model.param_names:
				# find this parameter value in the instance Model and assign it to the parameter here
				params[name.removeprefix(component_name)] = self.result.params[name].value
			if component_name[-1] == '_':
				component_name = component_name[0:-1]
			curves[component_name] = model.eval(**params, x=self.energies)
		return curves
		
	def update_params_initial_guess(self):
		for pname in self.params:
			self.params[pname].init_value = self.params[pname].value




