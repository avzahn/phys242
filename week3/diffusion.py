from propagator import *
import random
import sys
from math import ceil
class mh(simulation):
	def __init__(self, xi, ti, m = 1, hbar = 1, 
					 xf = None, nx = None, dx = None,
					 tf = None, nt = None, dt = None,
					 dtype = np.complex128):



		simulation.__init__(self, xi, ti, m , hbar, xf, nx, dx,
			 tf, nt, dt, dtype)

		self.accept = []
		self.mh_energy = []

	def randomize_initial_state(self,n = 1000):
		"""
		Distribute n walkers completely randomly over the
		space mesh, and return their positions
		"""
		sites = range(self.nx)
		walkers = []
		for i in range(n):
			j = random.choice(sites)
			self.arr[0,j] += 1
			walkers.append(j)

		return np.array(walkers)

	def run(self,walkers=None,
				nwalkers=1000,
				dn = 3,
				save = 100,
				measures_per_save = 10):

		if walkers == None:
			walkers = self.randomize_initial_state(nwalkers)

		end = self.nx - 1
		k = self.k
		dx = self.dx
		H = self.H
		nt = self.nt

		runlen = save * int(self.nt)
		measure = max(1,save / measures_per_save)
		report = runlen/10
		self.mh_energy = np.zeros( 
			(int(ceil(nt*measures_per_save*save)),),
			dtype=np.float64)
		mh_energy = self.mh_energy
		accept = 0

		nxt = np.copy(self.arr[0])

		for t in xrange(1,runlen):

			for i in range(len(walkers)):

				s = walkers[i]

				# propose to move a walker here
				# left or right by up to dn sites

				s1 = s + random.randint(-dn,dn)
				if s1 >= end:
					s1 = s1-end
				elif s1 < 0:
					s1 = end+s1

				# s1 now contains the proposed 
				# new site for the walker

				p = k[s,s1]
				if random.random() < p:
					accept += 1
					walkers[i] = s1
					nxt[s] -= 1
					nxt[s1] += 1

			if t % measure == 0:
				idx = t/(measure)
				nn = norm(nxt,dx)
				e = dx * conj(nxt).dot(H.dot(nxt))/ nn
				mh_energy[idx]=np.real(e)

			if t % save == 0:
				idx = t/save
				self.arr[idx] = np.copy(nxt)
				
			if t % report == 0:
				print """completed sweep %i of %i; acceptance fraction = %f""" % (t,runlen,accept/float(nwalkers*t))
				sys.stdout.flush()

		return accept / float(nwalkers*runlen)






