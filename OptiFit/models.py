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

	def add_background(self, func, params_dict, name='background'):
		# self.background_pdict = {}
		if name[-1] != '_':
			name += '_'
		for parameter, pdict in params_dict.items():
			self.params.add('{}{}'.format(name, parameter), **pdict)
 
		self.background = func
		self.background_name = name

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

	def get_nvals(self, includesample):
		# Function retrieves the refractive indices, n from the params object, for every layer in the stack
		nvals = []
		for layer, value in self.layers.items():
			if 'sample' in layer and includesample:
				nvals.append(self.calc_n_sample())
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
		# import pdb; pdb.set_trace()
		try:
			# RC = (target - ref) / ref + self.params['offset'] + self.params['slope']*(self.energies - min(self.energies))
			RC = (target - ref) / ref + self.calc_bg()
		except KeyError:
			RC = (target - ref) / ref
			print('ok')
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

	def calc_bg(self):
		args = {}
		for param in self.params:
			if self.background_name in param:
				args[param.split('_')[1]] = self.params[param].value
		bg = self.background(energies=self.energies, **args)
		return  bg 

	def loss(self, p):
		self.params = p 
		return np.array(self.calc_rc())-np.array(self.spectrum) 

	def fit(self, method='leastsq'):
		self.verify_params()
		A = Minimizer(self.loss, self.params, nan_policy='omit')
		result = A.minimize(method=method)
		return result
	
	def verify_params(self):
		for param in self.params:
			if self.params[param].min <= self.params[param].value <= self.params[param].max:
				pass
			else:
				raise ValueError('{} has an initial guess that is not within the bounds'.format(param))



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

