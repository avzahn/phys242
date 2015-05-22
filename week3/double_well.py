from propagator import *

class double_well(simulation):

	def __init__(self, xi, ti, m = 1, hbar = 1, 
					 xf = None, nx = None, dx = None,
					 tf = None, nt = None, dt = None,
					 dtype = np.complex128, l=1,v2=0):


		self.l = l
		self.v2 = v2

		mh.__init__(self, xi, ti, m , hbar, xf, nx, dx,
					 tf, nt, dt, dtype)

	def propagator(self):

		l = self.l
		v2 = self.v2

		h, m = self.hbar, self.m
		dt,dx = self.dt,self.dx
		nx,nt = self.nx, self.nt
		
		A = np.sqrt(i*2*np.pi*h*dt/m)
		_A = 1./A
		k = np.zeros(shape = (nx,nx), dtype = self.dtype )

		x = self.x

		c1 = .5 * (i / h) / dt
		c2 =  l * i * dt / h
 
 		l = len(x)

		for ii in range(l):
			for jj in range(ii,l):
				mid = .5 * (x[jj]+x[ii])
				s =  c1*(x[jj]-x[ii])**2 - c2*(mid*mid -v2)**2 
				s = _A*np.exp(s)
				k[ii,jj] = s
				k[jj,ii] = s

		return k

	def _V(self):

		x = self.x
		x = (self.l) * ( x**2 - self.v2)**2
		return np.diag(x)


	def tunneling_time(self):
		ex = self.EX()
		exf = np.fft.rfft(ex)
		fr = np.fft.rfftfreq(len(ex), d=self.dt)
		
		base_freq = fr[np.argmax(exf)]

		# return the half period
		return fr, exf, .5/base_freq
