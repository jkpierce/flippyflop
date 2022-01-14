import numpy as np
from scipy.special import erfc
from scipy.special import i0,i1,iv,ive
from scipy.integrate import quad
from scipy.misc import derivative
from numba import njit
import concurrent.futures 
import multiprocessing
n_workers = multiprocessing.cpu_count()-1


@njit() #yey, fast
def walkit(ke,kd,gam,Gam,vbar,x0,T,dt,L,stop,adaptive=0,seed=0): # x0 is the initial position
    """
    params is a dict containing
    ke -- entrainment rate 
    kd -- deposition rate
    V -- velocity
    dt -- sampling interval
    tmax -- maximum time to walk
    Nparticles -- number of particles in the system
        (this is not relevant for walkit, but it is later)
    domainLength -- the length scale L within which particles
         are initialized.
    
    This method contains an adaptive timestep. It will change dt during simulation.
    The adaption can sometimes be too aggressive or too moderate. It requires tuning.
    The adaptive timestep can be tuned in the end of this block of code.
    I should write keyword arguments to control it, but it's good enough as is.
    """

    if seed!=0:
        np.random.seed(seed) # set the seed if it's different than 0
    # shorthand
    k = ke+kd
    # initialize the simulation
    t = 0.0 # initial time
    x = x0 # initial position
    ns = 0 # saving index -- index into save time array which X will be filled for next
    ts = T[ns] # next saving time
    s = int(np.random.random() > kd/k) # initial state 1 with prob ke/k and 0 with prob kd/k
    v = np.random.normal(loc=vbar, scale=np.sqrt(Gam/gam))*s # 0 if starts in motion
        # otherwise the velocity starts with an initial maxwellian distribution

    X=[]
    V=[]
    S=[]

    if adaptive!=0:
        maxdt = adaptive
        dtdt = (maxdt-dt)/(T.max()-T.min())*dt # 

    while t<T.max():  # until you reach desired stopping time

        if s==1: # if walker is in a motion state
            # this condition is placed first since it'll be met *usually*
            #    as the rest state only occupies one pass through the loop
            n = np.random.normal() # this is simple euler mayarama scheme
            v = v + gam*(vbar-v)*dt + np.sqrt(2*Gam*dt)*n
            x = x + v*dt
            r = np.random.random()
            t += dt
            if np.exp(-kd*dt) < r: # if walker transitions to a rest state in dt
                s = 0   # assign it to rest 
                v = 0   # give it a 0 velocity.
                
        elif s==0: # if walker is in a rest state
            # choose the rest time
            tr = np.random.exponential(scale=1/ke)
            # step the time
            t += tr
            exit=True

        ### save the state of the walk if you've passed the sampling time
        ### this is a bit complex since the resting state can skip over
        ###       multiple save times. So you have to fill all of those skipped save times.
        if t>ts: # if t met the sampling time
            while t>ts and len(X)<len(T): # until t no longer passes the sampling time
                X.append(x) # fill the array with values
                V.append(v)
                S.append(s)
                ns+=1 # increment the saving index
                ts=T[ns] # find the next time saving happens


        # if you want to stop simulations early for the flux calculation, set stop=True
        if (x>L/10)&stop: # particle is far enough to be unlikely to return
            while len(X)<len(T): # fill all saved values up with the current state.
                X.append(x)
                V.append(v)
                S.append(s)
                break

        if exit: # if the particle existed a rest state and was saved, then make it move with a velocity
            s = 1  # this ordering is important. it cannot be done in the elif because if this s=1 setting were done there,
            v = np.random.normal(loc=vbar, scale=np.sqrt(Gam/gam)) # then the particle would never be saved in a resting state.
            #v=vbar
            exit=False

        # an adaptive timestep... 
        if adaptive!=0: # this is useful only for statistical moments
            dt+=dtdt

    return X,V,S # once the simulation is completed, return the set of positions 

@njit()
def system(ke,kd,gam,Gam,vbar,T,Np,L,dt,stop,adaptive):
    """ takes in parameters, assigns intitial conditions to each of `Np` particles,
        steps these particles through time up to tmax, then returns the array of times and `O`:
        a 2d array whose first index selects a particle and whose second index selects a time.
    """
    X0 = -np.random.random(size=Np)*L # generate set of random init conditions
    O = [walkit(ke,kd,gam,Gam,vbar,x0,T,dt,L,stop,adaptive) for x0 in X0] # move all particles from their random init positions
    return O # ideally would return a numpy array but numba doesn't allow it
    # O[particle index, (position, velocity, state) , time]

def flux(sys):
    """
    takes in output of system function
    returns the number of particles that cross x=0 up to time t
    as an array of the same length as T (from system output)
    """
    sys = np.array(sys) # cause the numba in system won't let me do it
    #print('-----------------')
    #print(sys[1][0].shape[0],sys[1][1].shape[0],sys[1][2].shape[0])
    #print(sys.shape)
    return np.array([ ( sys[:,0,i] > 0 ).sum() for i in range(sys.shape[-1]) ]) # sum up all particles which have passed x=0.
    

def simul(params):
    """
    This is a wrapper for the flux function. 
    It allows one to pass a single "params" list or tuple
    params = (ke,kd,D,V,T,Np,L,dt) 
    for use in parallel simulations, since concurrent futures
    does not like multiple arguments.
    """
    ke,kd,gam,Gam,vbar,T,Np,L,dt,stop,adaptive = params
    sys = system(ke,kd,gam,Gam,vbar,T,Np,L,dt,stop,adaptive=adaptive)
    return flux(sys)


def parallel(params,Nsystems):
    """
    parallelized simulation of many systems relying on concurrent futures.
    Each system is a number Np of particles arranged randomly over x \in (-L,0).
    They are stepped through time undergoing the flipflop dynamics until tmax.
    At the end, the number of systems having m particles to the right of x=0 can be counted.
    This defines the flux.
    Here, 
    params = (ke,kd,D,V,T,Np,L,dt)
    ke -- entrainment rate
    kd -- deposition rate
    D -- diffusion coefficient
    V -- movement velocity
    T -- set of sampling times (typically geometrically spaced for log-space plots)
    Np -- number of particles per system
    L -- length of the domain
    dt -- initial timestep (remembering that the timestep is adaptive by the definition of walkit).
    
    All of these parameters should be tuned to coordinate with one another. For example, if L is too large,
    very few particles would be expected to cross by the resolution times desired unless Np were also very large.
    The tradeoff is that large particle numbers make for slow simulations. 
    As a result, obtaining good resolution in a moderate amount of time is a bit of an art.
    """
    results = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as exe:
        futures = {}
        for i in range(Nsystems):
            futures[exe.submit(simul,params)]=i

        print(len(futures), ' jobs submitted. working....')
        j=0
        for future in concurrent.futures.as_completed(futures):
            i = futures[future]
            out = future.result()
            results.append(out)
            #fname = path+name+'_%05d'%i
            #np.save(fname,out)
            j+=1
            #print('job ', j,' complete of', len(futures),'.')
            n = j/len(futures)
            print('[' + '-'*round(n*30) + ' '*(30-round(n*30)) + ']')

    results = np.array(results)

    return results


# these are auxillary functions to scale the sediment flux simulations


def Q0(params): # returns einstein flux
    ke,kd,gam,Gam,V,T,Np,L,dt,stop,adaptive= params
    return ke*V/(ke+kd)
def tau(params): # returns timescale of diffusion
    ke,kd,gam,Gam,V,T,Np,L,dt,stop,adaptive= params
    D = Gam/gam**2
    return 2*D/V**2
def rho(params): # returns initial density of particles
    ke,kd,gam,Gam,V,T,Np,L,dt,stop,adaptive= params
    return Np/L
