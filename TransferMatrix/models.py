from lmfit import Parameters, Minimizer
import matplotlib.pyplot as plt
import numpy as np
import csv
from solcore.absorption_calculator.tmm_core_vec import coh_tmm
import os
cwd = os.path.dirname(os.path.realpath(__file__))


##### To do:

def interpolate_domain(rinfo, exp):
	known = np.array(rinfo[0])
	exp = np.array(exp)
	minidx = np.where(known == known[known < np.min(exp)].max())[0][0]
	maxidx = np.where(known == known[known > np.max(exp)].min())[0][0] +1
	e = known[minidx: maxidx]
	n = np.array(rinfo[1][minidx : maxidx])
	if len(rinfo) == 3:
		k = np.array(rinfo[2][minidx : maxidx])
	else:
		k = 0
		return np.interp(exp, e, n + 1j*k)


class pl_model(Parameters):

	def __init__(self, params = None, data = None, energies = None):
		super(pl_model).__init__()
		self.params = params
		self.data = data
		self.energies = energies

	def lorentz(self, x, a, x0, w):
		# lor = a/(1+(x-x0)**2/(w**2)
		lor = a/np.pi*(w/2)/((x-x0)**2+((w/2)**2))
		return lor

	def gaussian(self, x, a, x0, w):
		gau = a*np.exp(-np.power(x - x0, 2.) / (2 * np.power(w, 2.)))
		#gau = 1/(w*np.sqrt(2*np.pi))* np.exp(-(x-x0)**2/(2*w**2))
		return gau

	def add_lorentzian(self, name, x, a, x0, w):
		self.params.add(name + '_l_a', *a)
		self.params.add(name + '_l_x0', *x0)
		self.params.add(name + '_l_w', *w)

	def add_gaussian(self, name, x, a, x0, w):
		self.params.add(name + '_g_a', *a)
		self.params.add(name + '_g_x0', *x0)
		self.params.add(name + '_g_w', *w)

	def calc_spec(self):
		spectrum = 0
		pklist = []
		for paramset in self.params.valuesdict():
			pklist.append(paramset[:5])
		pklist = list(set(pklist))
		for pk in pklist:
			if '_l' in pk:	
				spectrum += self.lorentz(self.energies, self.params.valuesdict()[pk+'_a'], self.params.valuesdict()[pk+'_x0'], self.params.valuesdict()[pk+'_w'])
			if '_g' in pk:
				spectrum += self.gaussian(self.energies, self.params.valuesdict()[pk+'_a'], self.params.valuesdict()[pk+'_x0'], self.params.valuesdict()[pk+'_w'])
		return spectrum + self.params['offset'] + self.params['slope']*self.energies + self.params['slope']*(self.energies - min(self.energies))

	def plot_spec(self):
		spectrum = 0
		pklist = []
		for paramset in self.params.valuesdict():
			pklist.append(paramset[:5])
		pklist = list(set(pklist))
		plt.plot(self.energies, self.data, 'k+')
		for pk in pklist:
			if '_l' in pk:	
				plt.plot(self.energies, self.lorentz(self.energies, self.params.valuesdict()[pk+'_a'], self.params.valuesdict()[pk+'_x0'], self.params.valuesdict()[pk+'_w'])+ self.params['offset'] + self.params['slope']*self.energies, '--')
				spectrum += self.lorentz(self.energies, self.params.valuesdict()[pk+'_a'], self.params.valuesdict()[pk+'_x0'], self.params.valuesdict()[pk+'_w'])
			if '_g' in pk:
				spectrum += self.gaussian(self.energies, self.params.valuesdict()[pk+'_a'], self.params.valuesdict()[pk+'_x0'], self.params.valuesdict()[pk+'_w'])
		plt.plot(self.energies, spectrum + self.params['offset'] + self.params['slope']*self.energies, 'r')

	def loss(self, p):
		self.params = p
		return self.calc_spec()-self.data 

	def fit(self):
		A = Minimizer(self.loss, self.params)
		self.result = A.minimize()
		self.params = self.result.params


class tmm_model(Parameters):

	def __init__(self, params = None, data = None, energies = None):
		super(tmm_model).__init__()
		self.params = params
		self.data = data
		self.energies = energies
		
	def lorentz(self, x, a, x0, w):
		denominator=(x0**2-x**2)**2+(x**2*w**2)
		real=a*(x0**2-x**2)/denominator
		imag=x*w*a/denominator
		lor=real+1j*imag
		return lor

	def linear_background(self, m, b, w):
		return m*w + b

	def	add_lorentzian(self, name, x, a, x0, w):
		self.params.add(name + '_a', *a)
		self.params.add(name + '_x0', *x0)
		self.params.add(name + '_w', *w)

	def add_layer(self, name, d, n):
		self.params.add(name + '_d', *d)
		self.params.add(name + '_n', *n)
		if 'Gr' in name or 'gr' in name or 'GR' in name and 'full' in name:
			with open(os.path.join(cwd,'graphenerefractiveindex.csv'), 'r') as f:
				reader = csv.reader(f, delimiter=',')
				headers = next(reader)
				rinfo = np.array(list(reader)).astype(float).T
			self.grfull = interpolate_domain(rinfo, self.energies)
		if 'quartz' in name or 'sio2' in name or 'Sio2' in name or 'SiO2' in name and 'full' in name:
			with open(os.path.join(cwd,'sio2refractiveindex.csv'), 'r') as f:
				reader = csv.reader(f, delimiter=',')
				headers = next(reader)
				rinfo = np.array(list(reader)).astype(float).T
			self.quartzfull = interpolate_domain(rinfo, self.energies)
		if 'bare_si' in name and 'full' in name:
			with open(os.path.join(cwd,'sirefractiveindex-extended.csv'), 'r') as f:
				reader = csv.reader(f, delimiter=',')
				headers = next(reader)
				rinfo = np.array(list(reader)).astype(float).T
			self.sifull = interpolate_domain(rinfo, self.energies)
		# if 'hbn' in name or 'hBN' in name and 'full' in name:
		# 	with open('/Users/aidan/Documents/Python/sio2refractiveindex.csv', 'r') as f:
		# 		reader = csv.reader(f, delimiter=',')
		# 		headers = next(reader)
		# 		rinfo = np.array(list(reader)).astype(float).T
		# 	self.quartzfull = interpolate_domain(rinfo, self.energies)

	def get_nvals(self, includesample):
		somelist = []
		for a in self.params.valuesdict():
			somelist.append(a)
		nlist = []
		for i in somelist:
			if '_n' in i and i != 'baseline':
				nlist.append(i)
		nvals =[]
		if includesample == True:
			for n in nlist:
				if 'sample' in n:
					nvals.append(self.calc_n_sample(self.energies))
				elif 'Gr' in n or 'gr' in n or 'GR' in n and 'full' in n:
					nvals.append(self.grfull)
				elif 'hbn' in n or 'hBN' in n and 'full' in n:
					nvals.append(2.23 - 6.9e-4*1240/self.energies + 1j*0) # apparently in O.  Stenzel et al., Phys.  Status  Solidi  A (1996), but taken from Kim et al. JOSK 19, 503 (2015)
				elif 'quartz' in n or 'sio2' in n or 'Sio2' in n or 'SiO2' in n and 'full' in n:
					nvals.append(self.quartzfull)
				elif 'bare_si' in n and 'full' in n:
					nvals.append(self.sifull)
				else:
					nvals.append(np.ones_like(self.energies)*self.params.valuesdict()[n] +1j * np.zeros(np.shape(self.energies)))
		if includesample == False:
			for n in nlist:
				if 'sample' in n:
					pass
				elif 'Gr' in n or 'gr' in n or 'GR' in n and 'full' in n:
					nvals.append(self.grfull)
				elif 'hbn' in n or 'hBN' in n and 'full' in n:
					nvals.append(2.23 - 6.9e-4*1240/self.energies+ 1j*0)
				elif 'quartz' in n or 'sio2' in n or 'Sio2' in n or 'SiO2' in n and 'full' in n:
					nvals.append(self.quartzfull)
				elif 'bare_si' in n and 'full' in n:
					nvals.append(self.sifull)
				else:
					nvals.append(np.ones_like(self.energies)*self.params.valuesdict()[n] +1j * np.zeros(np.shape(self.energies)))		
		return nvals

	def get_dvals(self, includesample):
		dvals = []
		for d in self.params.valuesdict():
			# print(d)
			if '_d' in d:
				# print(d)
				if includesample == True:
					dvals.append(self.params.valuesdict()[d])
				if includesample == False:
					if 'sample' not in d:
						dvals.append(self.params.valuesdict()[d])
		return dvals

	def calc_n_sample(self, energies):   
		somelist = []
		for paramset in self.params.valuesdict():
			somelist.append(paramset[:3])
		pklist = []
		for i in somelist:
			if 'pk' in i:
				pklist.append(i)
		pkset = set(pklist)
		pklist = list(pkset)
		eps_sample = 1 + (self.params['eps_sample_r']+ self.energies*self.params['eps_sample_r_slope']) + 1j*(self.params['eps_sample_i']+ self.energies*self.params['eps_sample_i_slope']) #self.params['offset'] + self.params['slope']*self.energies+ 1j*self.energies*0
		for pk in pklist:
			eps_sample += self.lorentz(energies, self.params.valuesdict()[pk+'_a'], self.params.valuesdict()[pk+'_x0'], self.params.valuesdict()[pk+'_w'])
		n_sample = np.sqrt(eps_sample)
		return n_sample

	def calc_rc(self):
		target = coh_tmm('s', self.get_nvals(includesample = True), self.get_dvals(includesample = True), 0 + 1j*0, 1240/self.energies)
		ref = coh_tmm('s', self.get_nvals(includesample = False), self.get_dvals(includesample = False), 0 + 1j*0, 1240/self.energies)
		return (target['R']-ref['R'])/ref['R'] + self.params['offset'] + self.params['slope']*(self.energies - min(self.energies))

	def calc_rc_resolved(self):
		RCdict = {}
		RCdict['total'] = self.calc_rc()

		######## Calculate RC for each peak in the model
		somelist = []
		for paramset in self.params.valuesdict():
			somelist.append(paramset[:3])
		pklist = []
		for i in somelist:
			if 'pk' in i:
				pklist.append(i)
		pkset = set(pklist)
		pklist = list(pkset)
		print(pkset)

		ref = coh_tmm('s', self.get_nvals(includesample = False), self.get_dvals(includesample = False), 0 + 1j*0, 1240/self.energies)

		for count, pk in enumerate(pklist):
			# compute n_sample with only one peak
			eps_sample = 1 + (self.params['eps_sample_r']+ self.energies*self.params['eps_sample_r_slope']) + 1j*(self.params['eps_sample_i']+ self.energies*self.params['eps_sample_i_slope']) #self.params['offset'] + self.params['slope']*self.energies+ 1j*self.energies*0
			eps_sample += self.lorentz(self.energies, self.params.valuesdict()[pk+'_a'], self.params.valuesdict()[pk+'_x0'], self.params.valuesdict()[pk+'_w'])
			n_sample = np.sqrt(eps_sample)

			# rgenerate list of nvals
			somelist = []
			for a in self.params.valuesdict():
				somelist.append(a)
			nlist = []
			for i in somelist:
				if '_n' in i and i != 'baseline':
					nlist.append(i)
			nvals_sample =[]
			for n in nlist:
				if 'sample' in n:
					nvals_sample.append(n_sample)
				elif 'Gr' in n or 'gr' in n or 'GR' in n and 'full' in n:
					nvals_sample.append(self.grfull)
				elif 'hbn' in n or 'hBN' in n and 'full' in n:
					nvals_sample.append(2.23 - 6.9e-4*1240/self.energies + 1j*0) # apparently in O.  Stenzel et al., Phys.  Status  Solidi  A (1996), but taken from Kim et al. JOSK 19, 503 (2015)
				elif 'quartz' in n or 'sio2' in n or 'Sio2' in n or 'SiO2' in n and 'full' in n:
					nvals_sample.append(self.quartzfull)
				elif 'bare_si' in n and 'full' in n:
					nvals_sample.append(self.sifull)
				else:
					nvals_sample.append(np.ones_like(self.energies)*self.params.valuesdict()[n] +1j * np.zeros(np.shape(self.energies)))

			target = coh_tmm('s', nvals_sample, self.get_dvals(includesample = True), 0 + 1j*0, 1240/self.energies)
			ref = coh_tmm('s', self.get_nvals(includesample = False), self.get_dvals(includesample = False), 0 + 1j*0, 1240/self.energies)
			RCdict[pk] = (target['R']-ref['R'])/ref['R'] + self.params['offset']+self.params['slope']*(self.energies - min(self.energies))
		return RCdict
		#####################

	def calc_epsilon(self):
		somelist = []
		for paramset in self.params.valuesdict():
			somelist.append(paramset[:3])
		pklist = []
		for i in somelist:
			if 'pk' in i:
				pklist.append(i)
		pkset = set(pklist)
		pklist = list(pkset)
		print(pkset)

		for count, pk in enumerate(pklist):
			# compute n_sample with only one peak
			eps_sample = 1 + (self.params['eps_sample_r']+ self.energies*self.params['eps_sample_r_slope']) + 1j*(self.params['eps_sample_i']+ self.energies*self.params['eps_sample_i_slope']) #self.params['offset'] + self.params['slope']*self.energies+ 1j*self.energies*0
			eps_sample += self.lorentz(self.energies, self.params.valuesdict()[pk+'_a'], self.params.valuesdict()[pk+'_x0'], self.params.valuesdict()[pk+'_w'])
		return np.real(eps_sample), np.imag(eps_sample)

	def calc_target(self):
		target = coh_tmm('s', self.get_nvals(includesample = True), self.get_dvals(includesample = True), 0, 1240/self.energies)
		return (target['R'])

	def calc_ref(self):
		ref = coh_tmm('s', self.get_nvals(includesample = False), self.get_dvals(includesample = False), 0, 1240/self.energies)
		return ref['R'] 

	def check_params(self):
		pass			

	def fit(self):
		A = Minimizer(self.loss, self.params)
		self.result = A.minimize()
		self.params = self.result.params

	def ls_fit(self):
		A = Minimizer(self.loss, self.params)
		self.result = A.leastsq(self.params)
		self.params = self.result.params

	def loss(self, p):
		self.params = p
		return self.calc_rc()-self.data 

	def diffloss(self, p):
		self.params = p
		return np.diff(self.calc_rc())-np.diff(self.data) 

	def diff_fit(self):
		A = Minimizer(self.diffloss, self.params)
		self.result = A.leastsq(self.params)
		self.params = self.result.params
    	
    	
