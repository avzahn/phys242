"""
Simple 1D position space representation propagator for 
time and velocity independent potentials with a fixed
time and distance step
"""

import numpy as np
from copy import copy
import h5py

class wavefunction(object):


	def __init__(self, xi, ti, m = 1, h = 1, 
					 xf = None, nx = None, dx = None,
					 tf = None, nt = None, dt = None,
					 dtype = np.complex128):

		
		self.m  = m
		self.h = h # actually hbar
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

	def harmonic_propagator(self):
		"""
		generate infinitesimal the propagator for the harmonic
		oscillator potential with omega = 1


		This wouldn't be too hard to parallelize in native python,
		but otherwise the usual tricks for speeding up numerical
		python don't really work well with this function. It is
		however a very good candidate for cython.
		"""
		h, m = self.h, self.m
		dt,dx = self.dt,self.dx
		nx,nt = self.nx, self.nt
		i = np.complex(0,1)
		A = np.sqrt(i*2*np.pi*h*dt/m)

		k = np.zeros(shape = (nx,nx), dtype = self.dtype )

		x = self.x

		_h = 1./h
		_A = 1./A
		dx_A = dx*_A
		_dt = 1./dt
		for ii,row in enumerate(x):
			for jj,col in enumerate(copy(x)):
				s = (i*_h)*( .5*m*_dt*(x[jj]-x[ii])**2 - .5*dt*(.5*(x[jj]+x[ii]))**2 )
				k[ii,jj] = dx_A * np.exp(s)

		self.propagator = k

	def run(self):

		k = np.transpose(self.propagator)

		for i in range(0,len(self.arr)-1):

			self.arr[i+1] = k.dot(self.arr[i])

	def run_renormalize(self):

		k = np.transpose(self.propagator)
		
		for i in range(0,len(self.arr)-1):

			self.arr[i+1] = k.dot(self.arr[i])
			self.arr[i+1] *= (1/norm(self.arr[i+1],self.dx))


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


	def gaussian(self,a,mu,p = 0):
		i = np.complex(0,1)
		phase = np.exp(i*p*np.pi*self.x)
		g = np.power(a/np.pi,.25)*np.exp( -(.5*a)*(self.x - mu)**2 )
		return phase*g

def norm(v,dx):
	return dx*np.dot(v,np.conj(v))










