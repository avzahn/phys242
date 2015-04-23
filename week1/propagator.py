"""
Simple 1D position space representation propagator for 
time and velocity independent potentials with a fixed
time and distance step
"""

import numpy as np
from numpy import conj, transpose, real
import h5py

i = np.complex(0,1)

def norm(v,dx):
	return dx*np.dot(v,np.conj(v))

class simulation(object):
	"""
	Specific lagrangians are implemented by subclassing
	this and then implementing a propagator() method 
	which must assign the simulation.k matrix
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

	def run(self):

		k = self.dx * self.k

		for i in range(0,len(self.arr)-1):

			self.arr[i+1] = k.dot(self.arr[i])

	def run_renormalize(self):

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
			H = EV + EK
		except:
			pass

		return EX,EP,EV,EK,H		


class harmonic_oscillator(simulation):

	def __init__(self, xi, ti, m = 1, hbar = 1, 
					 xf = None, nx = None, dx = None,
					 tf = None, nt = None, dt = None,
					 dtype = np.complex128):

		simulation.__init__(self, xi, ti, m , hbar, xf, nx, dx,
					 tf, nt, dt, dtype)


		self.k = self.propagator()

	def propagator(self):
		"""
		generate infinitesimal the propagator for the harmonic
		oscillator potential with omega = 1

		This wouldn't be too hard to parallelize in native python,
		but otherwise the usual tricks for speeding up numerical
		python don't really work well with this function. It is
		however a very good candidate for cython.
		"""
		h, m = self.hbar, self.m
		dt,dx = self.dt,self.dx
		nx,nt = self.nx, self.nt
		
		A = np.sqrt(i*2*np.pi*h*dt/m)
		_A = 1./A
		k = np.zeros(shape = (nx,nx), dtype = self.dtype )

		x = self.x
 

		c1 = (i / h) * .5 * m / dt
		c2 = (i / h) * .5 * dt * .25

		l = len(x)

		for ii in range(l):
			for jj in range(ii,l):
				s =  c1*(x[jj]-x[ii])**2 - c2*(x[jj]+x[ii])**2 
				s = _A*np.exp(s)
				k[ii,jj] = s
				k[jj,ii] = s

		return k

	def EV(self):

		T = range(int(self.nt))
		arr = self.arr
		EV = np.zeros((self.nt,), dtype = self.dtype)

		x2 = .5 * (self.x**2)

		for t in T:
			EV[t] = conj(arr[t]).dot( x2 * arr[t])
			#EV[t] = (np.abs(arr[t])**2).dot(x2)

		return real(EV) * self.dx


	def analytic_propagator(self,T):

		#T = self.dtype(T)
		m = self.m
		h = self.hbar
		x = self.x
		l = len(x)
		nx = self.nx

		a = np.sqrt(m/(2*np.pi*i*h*np.sin(T)))
		b = i*m/(2*h*np.sin(T))

		k = np.zeros(shape = (nx,nx), dtype = self.dtype )

		for ii in range(l):
			for jj in range(ii,l):
				s =  a*np.exp(b*( np.cos(T)*(x[ii]**2 +x[jj]**2) -2*x[jj]*x[ii]))
				k[ii,jj] = s
				k[jj,ii] = s

		return k