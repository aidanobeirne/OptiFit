from lmfit import Parameters, Minimizer, report_fit, Model
import matplotlib.pyplot as plt
import numpy as np
import csv
from solcore.absorption_calculator.tmm_core_vec import coh_tmm
import os
import copy


def interpolate_domain(known_yvals, known_domain, desired_domain):
	known_domain = np.array(known_domain)
	desired_domain = np.array(desired_domain)
	minidx = np.where(known_domain == known_domain[known_domain < np.min(desired_domain)].max())[0][0]
	maxidx = np.where(known_domain == known_domain[known_domain > np.max(desired_domain)].min())[0][0] +1
	known_yvals = known_yvals[minidx:maxidx]
	known_domain = known_domain[minidx:maxidx]
	return np.interp(desired_domain, known_domain, known_yvals)

def import_rinfo(path):
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
class tmm_model:
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
		self.verify_params()


	def add_layer(self, name, d, n):
		graphite_words = ['graphene', 'Graphene', 'graphite', 'Graphite', 'gr', 'Gr']
		quartz_words = ['quartz', 'Quartz', 'SiO2', 'sio2', 'oxide']
		silicon_words = ['Silicon', 'silicon', 'Si', 'si']
		air_words = ['air', 'Air', 'Empty', 'empty', 'vacuum']
		hbn_words = ['hBN', 'hbn']

		path = 'none'
		if 'full' in name:
			n[1] = False # set vary to False in case it was accidentally set to True
			if any(word in name for word in graphite_words):
				path = os.path.join(os.path.dirname(__file__), 'refractive_index_data', 'graphite_refractive_index_info-eV.csv')
			elif any(word in name for word in quartz_words):
				path = os.path.join(os.path.dirname(__file__), 'refractive_index_data', 'quartz_refractive_index_info-eV.csv')
			elif any(word in name for word in silicon_words):
				path = os.path.join(os.path.dirname(__file__), 'refractive_index_data', 'silicon_refractive_index_info-eV.csv')
			elif any(word in name for word in air_words):
				path='custom'
				n_custom = np.ones_like(self.energies)
			elif any(word in name for word in hbn_words):
				path = 'custom'
				wls = 1240/self.energies
				if '2019' in name:
					n_custom = np.sqrt(1+3.263*wls**2/(wls**2 - (164.4)**2)) + 1j*0 # Phys. Status Solidi B 2019, 256, 1800417 ----------------- much better in NIR
				else:
					n_custom = 2.23 - 6.9e-4*wls + 1j*0 # apparently in O.  Stenzel et al., Phys.  Status  Solidi  A (1996), but taken from Kim et al. JOSK 19, 503 (2015) -------------------- much better in VIS	

		if 'csv' in path:
			energies, nr, k = import_rinfo(path)
			n_interpolated = interpolate_domain(nr, energies, self.energies)
			k_interpolated = interpolate_domain(k, energies, self.energies)
			n[0] = n_interpolated + 1j*k_interpolated
			self.layers[name] = {'n': n[0], 'd': d[0], 'vary': False} # these are refractive indices with pre-existing models and we will not vary them in the fit
		elif 'custom' in path:
			n[0] = n_custom
			self.layers[name] = {'n': n[0], 'd': d[0], 'vary': False} 
		elif 'none' in path and 'sample' in name:
			n[0] = self.calc_n_sample()
			self.layers[name] = {'n': n[0], 'd': d[0], 'vary': False} 
		# this clause means that we actually want to vary the refractive index as a fit parameter (hopefully this does not occur often, as it only works for constant values of nreal and nimag)
		elif 'none' in path and 'sample' not in name:
			if n[0] is complex:
				nreal = [np.real(n[0]), n[1], np.real(n[2]), np.real(n[3])]
				nimag = [np.imag(n[0]), n[1], np.imag(n[2]), np.imag(n[3])]
				self.params.add('{}_nreal'.format(name), *nreal)
				self.params.add('{}_nimag'.format(name), *nimag)
				self.layers[name] = {'nreal': nreal, 'nimag': nimag, 'd': d, 'vary': n[1]} # these refractive indices behave as fit parameters
			else:
				self.params.add('{}_n'.format(name), *n)
				self.layers[name] = {'nreal': nreal, 'nimag': nimag, 'd': d, 'vary': n[1]} 
		
		# always add the thicknesses to the parameters object
		self.params.add('{}_d'.format(name), *d)

	def get_nvals(self, includesample):
		# Function retrieves the refractive indices, n from the params object, for every layer in the stack
		nvals = []
		for layer, value in self.layers.items():
			if 'sample' in layer and includesample:
				nvals.append(self.calc_n_sample())
			elif 'sample' in layer and not includesample:
				pass
			else:
				if value['vary'] == False: 
					nvals.append(self.layers[layer]['n']) # pre-existing model stored in the layer dictionary
				else:
					nvals.append(self.params['{}_n'.format(layer)].value) # fit parameter stored in the params object
		return nvals

	def get_dvals(self, includesample):
		# Function retrieves the refractive thicknesses, d from the params object, for every layer in the stack
		dvals = []
		for layer in self.layers.keys():
			if 'sample' in layer and not includesample:
				continue
			dvals.append(self.params['{}_d'.format(layer)].value)
		return dvals

	def calc_n_sample(self):  
		try:
			eps_sample = 1 + (self.params['eps_sample_r']+ self.energies*self.params['eps_sample_r_slope']) + 1j*(self.params['eps_sample_i']+ self.energies*self.params['eps_sample_i_slope'])
		except KeyError:
			eps_sample = 1

		for peak in self.peaks:
			eps_sample += self.complex_lorentzian(self.energies, self.params['{}_a'.format(peak)].value, self.params['{}_x0'.format(peak)].value, self.params['{}_w'.format(peak)].value)
		n_sample = np.sqrt(eps_sample)
		return n_sample

	def tmm_calc(self, includesample):
		if includesample:
			return coh_tmm('s', self.get_nvals(includesample=True), self.get_dvals(includesample=True), 0, 1240/self.energies)
		else:
			return coh_tmm('s', self.get_nvals(includesample=False), self.get_dvals(includesample=False), 0, 1240/self.energies)

	def calc_rc(self):
		target = self.tmm_calc(includesample=True)['R']
		ref = self.tmm_calc(includesample=False)['R']
		try:
			RC = (target - ref) / ref + self.params['offset'] + self.params['slope']*(self.energies - min(self.energies))
		except KeyError:
			(target - ref) / ref
		return RC

	def calc_rc_resolved(self):
		RC_dict = {}
		RC_dict['total'] = self.calc_rc()
		for peak in self.peaks:
			n_sample = np.sqrt(1 + self.complex_lorentzian(self.energies, self.params['{}_a'.format(peak)].value, self.params['{}_x0'.format(peak)].value, self.params['{}_w'.format(peak)].value))
			target_nvals = self.get_nvals(includesample=True)
			# find and replace the sample n with the n containing only one peak
			for idx, key in enumerate(self.layers.keys()):
				if 'sample' in key:
					target_nvals[idx] = n_sample
			target = coh_tmm('s', target_nvals, self.get_dvals(includesample = True), 0 + 1j*0, 1240/self.energies)['R']
			ref = coh_tmm('s', self.get_nvals(includesample = False), self.get_dvals(includesample = False), 0 + 1j*0, 1240/self.energies)['R']
			RC_dict[peak] = (target-ref)/ref
		return RC_dict
	
	def loss(self, p):
		self.params = p 
		return np.array(self.calc_rc())-np.array(self.spectrum) 

	def fit(self, method='leastsq'):
		A = Minimizer(self.loss, self.params, nan_policy='omit')
		result = A.minimize(method=method)
		return result
	
	def verify_params(self):
		for param in self.params:
			if self.params[param].min <= self.params[param].value <= self.params[param].max:
				pass
			else:
				raise ValueError('{} has an initial guess that is not within the bounds'.format(param))

class CompositeModel(Model):
	def __init__(self):
		super().__init__()
		self.components = []
		
	# def add_eps_inf(self):
	# 	# execute this function if the model is of a dielectric function
	# 	mod = Model(eps_inf)
	# 	mod.set_param_hint(vary=False)
	# 	try:
	# 		self += mod
	# 	except AttributeError:
	# 		self = mod

	def add_component(self, func, params_dict, name='peak1', prefix='pk1'):
		self.components.append(prefix)
		mod = Model(func=func, name=name, prefix=prefix)
		for parameter, pdict in params_dict.items():
			mod.set_param_hint(parameter, **pdict)
		try:
			self += mod # add component to the model
		except AttributeError:
			self = mod