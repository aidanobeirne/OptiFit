from lmfit import Parameters, Minimizer
import matplotlib.pyplot as plt
import numpy as np
import csv
from solcore.absorption_calculator.tmm_core_vec import coh_tmm
import os


def interpolate_domain(known_yvals, known_domain, desired_domain):
	known_domain = np.array(known_domain)
	desired_domain = np.array(desired_domain)
	minidx = np.where(known_domain == known_domain[known_domain < np.min(desired_domain)].max())[0][0]
	maxidx = np.where(known_domain == known_domain[known_domain > np.max(desired_domain)].min())[0][0] +1
	known_yvals = known_yvals[minidx:maxidx]
	known_domain = known_domain[minidx:maxidx]
	return np.interp(desired_domain, known_domain, known_yvals)

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

	def import_rinfo(self, path):
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
			

	def add_layer(self, name, d, n):
		 # store the layers in an instance attribute for easier access later
		self.params.add(name + '_d', *d)
		self.params.add(name + '_n', *n)
		self.verify_params()
		n = n[0] # grab the initial guess or, in the event that it wont be varied, the value given for the layer
		d = d[0]

		path = 'none'
		if ('graphene' in name or 'graphite' in name) and ('full' in name):
			path = os.path.join(os.path.dirname(__file__),'graphite_refractive_index_info-eV.csv')
		elif ('quartz' in name or 'sio2' in name or 'Sio2' in name or 'SiO2' in name) and ('full' in name):
			path = os.path.join(os.path.dirname(__file__),'quartz_refractive_index_info-eV.csv')
		elif ('bare_si' in name or 'Si' in name or 'si' in name) and ('full' in name):
			path = os.path.join(os.path.dirname(__file__),'silicon_refractive_index_info-eV.csv')
		elif ('hbn' in name or 'hBN' in name) and ('full' in name):
			path = 'custom'
			n = 2.23 - 6.9e-4*1240/self.energies + 1j*0 # apparently in O.  Stenzel et al., Phys.  Status  Solidi  A (1996), but taken from Kim et al. JOSK 19, 503 (2015)

		if 'csv' in path:
			energies, nr, k = self.import_rinfo(path)
			n_interpolated = interpolate_domain(nr, energies, self.energies)
			k_interpolated = interpolate_domain(k, energies, self.energies)
			n = n_interpolated + 1j*k_interpolated
		elif 'custom' in path:
			n=n
		else:
			if isinstance(n, complex):
				n = np.ones_like(self.energies)*n
			else:
				n = np.ones_like(self.energies)*n +1j * np.zeros(np.shape(self.energies))
		self.layers[name] = {'n': n, 'd': d} # store the interpolated domain in the layer dictionary to be called later

	def get_nvals(self, includesample):
		# Function retrieves the refractive indices, n, for every layer in the stack
		nvals = []
		for layer, values in self.layers.items():
			if 'sample' in layer and includesample:
				nvals.append(self.calc_n_sample())
			elif 'sample' in layer and not includesample:
				pass
			else:
				nvals.append(values['n'])
		return nvals

	def get_dvals(self, includesample):
		# Function retrieves the refractive thicknesses, d, for every layer in the stack
		dvals = []
		for layer, values in self.layers.items():
			if 'sample' in layer and not includesample:
				continue
			dvals.append(values['d'])
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

	def calc_target(self):
		target = coh_tmm('s', self.get_nvals(includesample = True), self.get_dvals(includesample = True), 0, 1240/self.energies)
		return target['R']

	def calc_reference(self):
		ref = coh_tmm('s', self.get_nvals(includesample = False), self.get_dvals(includesample = False), 0, 1240/self.energies)
		return ref['R'] 

	def calc_rc(self):
		try:
			RC = (self.calc_target() - self.calc_reference())/ self.calc_reference() + self.params['offset'] + self.params['slope']*(self.energies - min(self.energies))
		except KeyError:
			RC = (self.calc_target() - self.calc_reference())/ self.calc_reference()
		return RC

	def loss(self, p):
		self.params = p
		return self.calc_rc()-self.spectrum 

	def fit(self):
		A = Minimizer(self.loss, self.params)
		result = A.minimize()
		self.params = result.params
	
	def verify_params(self):
		for param in self.params:
			if self.params[param].min <= self.params[param].value <= self.params[param].max:
				pass
			else:
				raise ValueError('{} has an initial guess that is not within the bounds'.format(param))
