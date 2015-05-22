"""
Simple 1D position space representation propagator for 
time and velocity independent potentials with a fixed
time and distance step
"""

import numpy as np
from numpy import conj, transpose, real
import h5py
from multiprocessing import Process
import pickle
import os
import sys

i = np.complex(0,1)

def norm(v,dx):
	return dx*np.dot(v,np.conj(v))

class simulation(object):
	"""
	Specific lagrangians are implemented by subclassing
	this and then implementing a propagator() method.

	Optionally, subclasses should implement a _V() method
	to compute the matrix elements of the potential
	"""

	def __init__(self, xi, ti, m = 1, hbar = 1, 
					 xf = None, nx = None, dx = None,
					 tf = None, nt = None, dt = None,
					 dtype = np.complex128):

		
		self.m  = m
		self.hbar = hbar 
		self.dtype = dtype

		self.xi = xi
		self.ti = ti

		if tf == None:
			if ( (dt == None) or (nt == None)):
				raise Exception("must specify timestep and number \
					of time samples if tf not given")

			tf = (dt*nt) + ti
			self.tf = tf
			self.nt = nt
			self.dt = dt
		else:
			self.tf = tf

		if xf == None:
			if ( (dx == None) or (nx == None)):
				raise Exception("must specify x step and number \
					of x samples if xf not given")
			xf = (dx*nx) + xi
			self.xf = xf
			self.nx = nx
			self.dx = dx
		else:
			self.xf = xf

		# if execution gets here, we're guaranteed to have
		# values for xi,xf,ti,tf one way or the other

		if dt == None:
			if nt == None:
				raise Exception("must specify one of dt or nt")
			self.nt = nt
			dt = float(tf-ti)/nt
			self.dt = dt
		else:
			self.dt = dt

		if nt == None:
			if dt == None:
				raise Exception("must specify one of dt or nt")
			self.dt = dt
			nt = float(tf-ti)/dt
			self.nt = nt
		else:
			self.nt = nt

		if dx == None:
			if nx == None:
				raise Exception("must specify one of dx or nx")
			self.nx = nx
			dx = float(xf-xi)/nx
			self.dx = dx
		else:
			self.dx = dx

		if nx == None:
			if dx == None:
				raise Exception("must specify one of dx or nx")
			self.dx = dx
			nx = float(xf-xi)/dx
			self.nx = nx
		else:
			self.nx = nx

		# final consistency check
		if not(xf == xi + (nx*dx)):
			raise Exception("Inconsistent x parameters")

		if not(tf == ti + (nt*dt)):
			raise Exception("Inconsistent t parameters")

		self.arr = np.zeros(shape = (nt,nx), dtype = self.dtype)
		self.x = np.linspace(self.xi,self.xf,nx, dtype = self.dtype)
		self.t = np.linspace(self.ti,self.tf,nt, dtype = self.dtype)
		self.T = tf-ti

		self.set_operators()

	def set_operators(self):

		# won't initialize unless propagator() is defined
		self.k = self.propagator()
		self.P = self._P()

		try:
			self.V = self._V()
			self.H = self._H()
		except:
			pass

	def run(self):

		k = self.dx * self.k

		arr = self.arr
		
		for i in range(0,len(self.arr)-1):

			arr[i+1] = k.dot(self.arr[i])
			arr[i+1] /= norm(self.arr[i+1],self.dx)

	def set_initial_state(self, psi0):
		self.arr[0] = psi0

	def save(self,run_name,fname):

		with h5py.File(fname,'a') as f:
			grp = f.create_group(run_name)
			re = grp.create_dataset("re", data = np.real(self.arr))
			im = grp.create_dataset("im", data = np.imag(self.arr))
			grp.attrs['dx'] = self.dx
			grp.attrs['dt'] = self.dt
			grp.attrs['xi'] = self.xi
			grp.attrs['xf'] = self.xf
			grp.attrs['ti'] = self.ti
			grp.attrs['tf'] = self.tf

			try:
				_e = grp.create_dataset("spectrum", data = self.e)
				_fr = grp.create_dataset("energies", data = self.fr)
			except:
				pass

	def restore(self,run_name,fname):
		with h5py.File(fname,'r') as f:
			grp = f[run_name]
			re,im = np.array(grp['re']), np.array(grp['im'])
			self.dx = grp.attrs['dx']
			self.dt = grp.attrs['dt']
			self.xi = grp.attrs['xi']
			self.xf = grp.attrs['xf']
			self.ti = grp.attrs['ti']
			self.tf = grp.attrs['tf']
			try:
				self.e = np.array(grp['spectrum'])
				self.fr = np.array(grp['energies'])
			except:
				pass

		self.set_operators()

	def pmf(self):

		return self.dx * np.abs(self.arr)**2

	def gaussian(self,a,mu,p = 0):
		"""
		A normalized position representation of a gaussian state centered
		at mu in position space and p in momentum space
		"""
		i = np.complex(0,1)
		phase = np.exp(i*p*np.pi*self.x)
		g = np.power(a/np.pi,.25)*np.exp( -(.5*a)*(self.x - mu)**2 )
		return phase*g


	def spectrum(self,depth = 1,return_negative = False,threads=1):
		if threads ==1:
			return self._spectrum(depth,return_negative)
		else:
			return self._p_spectrum(self,depth,threads)

	def _spectrum(self,depth = 1, return_negative = False):

		from scipy import signal
		pfind = signal.find_peaks_cwt

		n = int(depth * self.nt) 

		dx = self.dx
		last_k = self.k
		k = self.k

		tr = []

		for i in range(n):

			next_k = dx * last_k.dot(k)

			tr.append(np.trace(last_k))

			last_k = next_k

			if (i % 200) == 0:
				print '%s/%s'%(i,n)

		
		tr = dx * np.array(tr)
		fr = self.hbar * 2 * np.pi * np.fft.fftfreq(n,d=self.dt)[::-1]
		spec = np.fft.fft(tr)
		spec = np.abs(spec)**2

		self.fr = fr
		self.spec = spec

		if return_negative == False:
			idx = int(np.ceil((len(spec)/2.)))
			fr = fr[idx:]
			spec = spec[idx:]

		w = np.linspace(.01,.05,15) * np.max(fr)
		idx = pfind(spec,w)
		levels = []
		m = .2 * np.max(spec)
		for i in idx:
			if spec[i] > m:
				levels.append(fr[i])


		return fr,spec,sorted(levels)

	def _p_spectrum(self, depth = 1, threads = 3):
		"""
		Only useful on systems that don't compile numpy 
		against a parallel BLAS
		"""

		n = depth * int(self.nt)
		dx = self.dx
		last_k = self.k
		k = self.k

		seed = []

		for i in range(threads):
			next_k = dx * last_k.dot(k)
			seed.append(last_k)
			last_k = next_k

		step = seed[-1]
		_tr = np.zeros(shape=(n,), dtype = self.dtype) 
		tmp = ["__tmp%s"%(i) for i in range(threads)]

		workers = [Process(target = spectrum_worker, args =
			(i,threads,step,seed[i],_tr,dx,tmp[i])) 
				for i in range(threads)]

		for w in workers:
			w.start()
		for w in workers:
			w.join()

		e = np.zeros((n,),dtype=self.dtype)

		_tr = [pickle.load(open(tmp[i],'r')) for i in range(threads)]

		for tr in _tr:
			e += tr

		fr = self.hbar * self.nt * np.fft.fftfreq(n)[::-1]
		e = np.abs(np.fft.fft(dx*e))**2
		self.fr = fr
		self.e = e

		for name in tmp:
			os.remove(name)

		return fr,e

	def EX(self):

		T = range(int(self.nt))
		arr = self.arr
		x = self.x
		EX = np.zeros((self.nt,), dtype = self.dtype)

		for t in T:
			EX[t] = conj(arr[t]).dot(x * arr[t])

		# any residual imaginary component is
		# a numerical error that will just
		# cause matplotlib to complain
		return real(EX) * self.dx

	def EP(self):

		T = range(int(self.nt))
		arr = self.arr

		EP = np.zeros((self.nt,), dtype = self.dtype)

		for t in T:
			EP[t] = conj(arr[t][:-1]).dot(np.diff(arr[t]))

		# The discrete derivative operation should contribute
		# a factor of 1/dx, which would have been cancelled
		# in the riemann summation, so the factor is just dropped
		# entirely

		return real( (-i * self.hbar / (1) ) * EP)

	def EV(self):

		T = range(int(self.nt))
		arr = self.arr
		EV = np.zeros((self.nt,), dtype = self.dtype)
		V = self.V

		for t in T:
			EV[t] = conj(arr[t]).dot(V.dot(arr[t]))

		return real(EV) * self.dx

	def _P(self):
		"""
		Momentum operator matrix
		"""

		one = np.ones(len(self.k),dtype=self.dtype)
		p = np.diag(one) + np.diag(-one[1:],k=1)

		return (-i*self.hbar) * p

	def _H(self):
		"""
		Return the Hamiltonian matrix
		"""
		p = self.P
		m = self.m
		v = self.V
		return p.dot(p)/(2*m) + v 

	def EH(self):
		H = self.H
		T = range(int(self.nt))
		arr = self.arr
		EH = np.zeros((self.nt,), dtype = self.dtype)

		for t in T:
			EH[t] = conj(arr[t]).dot(H.dot(arr[t]))

		return real(EH) * self.dx



	def EK(self):

		T = range(int(self.nt))
		arr = self.arr

		EK = np.zeros((self.nt,), dtype = self.dtype)

		for t in T:
			EK[t] = conj(arr[t][:-2]).dot(np.diff(np.diff(arr[t])))

		return real( (-1 * self.hbar**2 / (2*self.m*self.dx) ) * EK) 

	def observables(self):

		EX = self.EX()
		EP = self.EP()
		EK = self.EK()
		EV,H = None,None

		try:
			# the expectation value of potential energy
			# has to be implemented by a subclass
			EV = self.EV()
			H = self.EH()
		except:
			pass

		return EX,EP,EV,EK,H		

	def harmonic_basis(self,n, w = 1):
		"""
		Return a sampled nth eigenstate of the 
		harmonic oscillator potential
		"""

		from scipy.special import hermite
		from scipy.misc import factorial as f

		m = self.m
		pi = np.pi
		h = self.hbar

		X = np.sqrt(m*w/h)*np.real(self.x)
		poly = hermite(n)
		poly = np.array([poly(x) for x in X], dtype=self.dtype)

		a = (1/np.sqrt(f(n)*(2**n)))*np.power(m*w/(pi*h),.25)
		b = np.exp(-m*w*(X**2)/(2*h))

		poly = a*b*poly

		if norm(poly,self.dx) < .9:
			print >>sys.stderr, "harmonic_basis(%s): consider extending space mesh" % (n)

		return poly

	def harmonic_matrix(self,operator,n=7, w=1):
		"""
		Compute the matrix elements of an operator
		in the harmonic oscillator basis
		"""

		basis = [ self.harmonic_basis(i,w) for i in range(n) ]
		m = np.zeros((n,n), dtype = operator.dtype)

		for i in range(n):
			for j in range(i,n):
				m[i,j] = np.conj(basis[i]).dot( operator.dot(basis[j]) )
				m[j,i] = np.conj(m[i,j])

		m *= self.dx

		return m

def spectrum_worker(tid,
	nthreads,
	step,
	start,
	tr,
	dx,
	tmp):

	last_k = start

	j = 0
	target = range(nthreads+tid,len(tr),nthreads)
	l = len(target)
	ll = len(tr)
	for i in target:
		
		next_k = dx * last_k.dot(step)

		tr[i] = np.trace(last_k) 

		last_k = next_k

		if (j % 200) == 0:
			print "thread %s: %s/%s (%s/%s)" %(tid,j,l,i,ll)

		j += 1

	pickle.dump(tr,open(tmp,'w'))
	print "thread %s complete" %(tid)
