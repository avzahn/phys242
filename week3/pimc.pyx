import numpy as np
import random

from libc.math cimport exp

cimport numpy as np
cimport cython

cdef:
	double m = 1
	double w = 1
	double h = 1
	double xi = -4
	double xf = 4
	double dx = .004
	double dt = 10

	double L =1
	double v = 2

	int nwalkers = 1000
	float D = 1


cpdef clamp_i(int i, int start, int end):
	
	if i >= end:
		i = start + i - end
	elif i < start:
		i = end - (start-i)
		
	return i

cdef clamp_f(double i, double start, double end):
	
	if i >= end:
		i = start + i - end
	elif i < start:
		i = end - (start-i)
		
	return i

cpdef double V(double x):
	return 0.5*w*w*m*x*x

cdef double dVdx(double x):
	return w*w*m*x

"""
cpdef double V(double x):
	return L*(x*x - v*v)**2
	
cdef double dVdx(double x):
	4*L*(x*x -v*v)*x
"""

cdef double energy(double x):
	return V(x)+0.5*x*dVdx(x)

@cython.boundscheck(False)
cdef double sweep( np.ndarray[np.float64_t,ndim=1] walkers, int* accum_accept):
	
	cdef int l = nwalkers
	cdef int end = l-1
	cdef int accept = 0
	cdef double dest, p, ds
	cdef int ip, im, i

	i = <int>(<float>random.random() * nwalkers)

	dest = walkers[i]+(2.0 * D * (<double>random.random()-0.5))
	ip = clamp_i(i+1,0,end)
	im = clamp_i(i-1,0,end)

	ds = V(dest) - V(walkers[i]) \
		+ .5 *(((walkers[ip]-dest)/dt)**2  \
		+ (( dest - walkers[im] ) /dt )**2 \
		- (( walkers[ip] - walkers[i] )/dt )**2\
		-(( walkers[i] - walkers[im] )/dt )**2)
	
	if ds < 0:
		accept = 1
	else:
		p = exp(-dt*ds)
		if random.random() < p:
			accept = 1
			
	if accept:
		walkers[i] = dest
		accum_accept[0] += 1
		return dest
			
	return walkers[i]

@cython.boundscheck(False)
def run(n, burn_in, measure = 10):

	cdef int accept = 0	
	cdef int _measure = measure
	cdef double x
	cdef int i,_n,j,b, _burn_in

	_burn_in = burn_in
	_n = n

	cdef np.ndarray[np.float64_t,ndim=1] walkers = \
		2*(xf-xi)*(np.random.random(nwalkers)-0.5)
	cdef np.ndarray[np.float64_t,ndim=1] mesh = \
		np.arange(xi,xf,dx)
	cdef np.ndarray[np.float64_t,ndim=1] energies = \
		np.zeros((n,))
	cdef np.ndarray[np.float64_t,ndim=1] state = \
		np.zeros((len(mesh),))

	cdef int nbins = len(mesh)

	cdef double e = 0
	
	for j in range(_n):

		for i in range(nwalkers):

			x = sweep(walkers,&accept)
		
			if j > _burn_in:
				b =<int>( nbins*(x - xi) / (xf-xi) )
				if b < nbins and b >= 0:
					state[b] += 1
			e += energy(x)

		energies[j] = e/<double>nwalkers
		e = 0

		if (j % 32768) == 0:
			print j, accept/(<float>(((j+1)*nwalkers)))

			
	return mesh,state,energies
			
