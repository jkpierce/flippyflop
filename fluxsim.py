import flipflopinertial as ffi
import numpy as np

adaptive = (2.0,1) # set up the adaptive timestep.
tmax = 3e5 # max simulatoin time
T = np.geomspace(1e-2, tmax, 40)
Np = 150000 # number of particles in each system to be simulated
kd = 1
ke = 0.05*kd # scale entrainment to deposition rate
gam = 1/(0.01/kd) # relaxation time - 1/20 the motion time
V = 0.1 # velocity.
stop=True # the simulation stops for x far enough beyond x=0.

# these are the specific parameters to each of the 4 flux simulations
Pe = [5,0.5,0.1,0.025] # array of peclet numbers to simulate
# Pe = V^2/(2 kd D).
D = [V**2/2/kd/pe for pe in Pe] # array of velocities to simulate
L = [3*max(np.sqrt(2*d*tmax), ke*V/(ke+kd)) for d in D] # array of domain sizes to simulate
Gam = [d*gam**2 for d in D]
# these domain sizes are set as 3x the maximum expected travel distance of a particle over the simulation time.

dt = min(1/kd,1/ke,1/gam)/35 # 35 times smaller than any other timescale

Nsystems = 50
i = 0
for Li,Gam_i in zip(L,Gam):
    params = [ke,kd,gam,Gam_i,V,T,Np,Li,dt,stop,adaptive]
    q = ffi.parallel(params,Nsystems)
    np.save('sim_%d'%i,q)
    print('sim {} finished.'.format(i))
    print(':::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::')
    i+=1
