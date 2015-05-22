from diffusion import *

class harmonic_oscillator(mh):

	def __init__(self,xi, ti, m = 1, hbar = 1, 
					 xf = None, nx = None, dx = None,
					 tf = None, nt = None, dt = None,
					 dtype = np.complex128, w = 1):

		self.w = w

		mh.__init__(self, xi, ti, m , hbar, xf, nx, dx,
					 tf, nt, dt, dtype)

		


	def propagator(self):
		"""
		compute the matrix elements of e^(-dS/h),
		in the imaginary time formalism.

		Notice that the implementation for this function is now
		quite different than in the previous two submissions.

		Essentially, we're just working out all the possible 
		proposal probabilities in advance by making a discrete
		space approximation. This is the first time in the course
		so far I've had to make a concession to python's
		performance limitations... 
		"""
		h, m = self.hbar, self.m
		dt,dx = self.dt,self.dx
		nx,nt = self.nx, self.nt
		w = self.w
		
		S = np.zeros(shape = (nx,nx), dtype = self.dtype )
		dS = np.zeros(shape = (nx,nx), dtype = self.dtype )

		x = self.x

		c0 = (m * dt /(2* h))
		c1 = c0/(dt*dt)
		c2 = c0*w*w*.25

		l = len(x)

		for ii in range(l):
			for jj in range(ii,l):
				s =  c1*(x[jj]-x[ii])**2 + c2*(x[jj]+x[ii])**2
				S[ii,jj] = s
				S[jj,ii] = s

		for ii in range(l):
			for jj in range(l):
				ds = (S[ii,ii] - S[ii,jj])
				dS[ii,jj] = ds

		return np.exp(dS)

	def _V(self):

		x = self.x
		x = .5 * (self.m) * (self.w *x)**2 
		return np.diag(x)