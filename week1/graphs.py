from propagator import harmonic_oscillator
import matplotlib.pyplot as plt
import numpy as np

T0 = 2*np.pi
dt = T0/128.
nx = 600
xi = -4.
xf = 4.
a = 2.

psi = harmonic_oscillator(xi=xi,xf=xf, dt=dt, nx=nx, ti=0.,tf=T0)

psi0 = psi.gaussian(2,.75)
psi.set_initial_state(psi0)

psi.run_renormalize()
EX,EP,EV,EK,H = psi.observables()

x = np.real(psi.x)
t = np.real(psi.t)

# plot this run at every T0/16
plt.figure()
samples = [(i*16,psi.arr[i*16]) for i in range(8)]
for i,sample in samples:
	plt.plot(x,np.abs(sample)**2, label = str(i)+'*T0/128')
plt.ylabel('sampled probability amplitude')
plt.xlabel('position')

plt.legend()
plt.savefig('wavefunction_timelapse_renormalize.png',dpi=300)
plt.close()

# plot <x> and <p> over the run
plt.figure
plt.xlabel('simulation time')
plt.ylabel('position (momentum)')
plt.plot(t,EX, color ='b')
plt.plot(t,EX,'ro',color='b',label='<x>')
plt.plot(t,EP, color ='r')
plt.plot(t,EP,'ro',color='r',label='<p>')
plt.legend(loc=4)
plt.savefig('EX_EP_renormalize.png',dpi=300)
plt.close()

# energy plot
plt.close('all')
plt.figure()
plt.xlabel('simulation time')
plt.ylabel('E')
plt.plot(t,EV,label='<V>')
plt.plot(t,EK,label='<KE>')
plt.plot(t,H,label='<H>')
plt.legend()
plt.savefig('energy_renormalize.png',dpi=300)
#
# redo some plots, without renormalization
#
psi.run()
EX,EP,EV,EK,H = psi.observables()
# plot this run at every T0/16
plt.figure()
samples = [(i*16,psi.arr[i*16]) for i in range(8)]
for i,sample in samples:
	plt.plot(x,np.abs(sample)**2, label = 'T0/'+max(str(i),1))

plt.ylabel('sampled probability amplitude')
plt.xlabel('position')

plt.legend()
plt.savefig('wavefunction_timelapse.png',dpi=300)
plt.close()

# plot <x> and <p> over the run
plt.figure
plt.xlabel('simulation time')
plt.ylabel('position (momentum)')
plt.plot(t,EX, color ='b')
plt.plot(t,EX,'ro',color='b',label='<x>')
plt.plot(t,EP, color ='r')
plt.plot(t,EP,'ro',color='r',label='<p>')
plt.legend(loc=4)
plt.savefig('EX_EP.png',dpi=300)
plt.close()
