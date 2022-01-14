import numpy as np
from scipy.special import erfc
from scipy.special import i0,i1,iv,ive
from scipy.integrate import quad
from scipy.misc import derivative
from numba import njit
import concurrent.futures 
import multiprocessing
n_workers = multiprocessing.cpu_count()-1

def P(x,t,ke,kd,V,D):
    """
    analytical probability distribution.
    This takes in two numpy arrays or lists x and t
    then it provides a 2d array where the first index is time
    and the second index is space.
    """
    k=ke+kd # shorthand
    def K(u,x,t):
        """ this represents the integrand """
        bes = i0(2*np.sqrt(ke*kd*u*(t-u)))
        exp = np.exp(-ke*(t-u)-kd*u)
        gau = np.exp(-(x-V*u)**2/(4*D*u))/np.sqrt(4*np.pi*D*u)
        return bes*exp*gau
    
    def I(x,t):
        """ this represents the integral """
        return quad(K,0,t,args=(x,t),points=[0])[0]
    
    def partial(func, var=0, point=[],order=1):
        """
        computes the partial derivative of I(x,t) of some order with respect to
        variable x (var=0) or t (var=1) about point = (x,t)
        """
        args = point[:] # copy of the `point` list
        def wrap(x): # temp function which kills one variable and evals the other
            args[var] = x # replace the "var" location of the args array with fn var x
            return func(*args) # then eval the function at the new args
        return derivative(wrap, point[var], dx = 1e-6,n=order)
    
    def Pi(x,t):
        """ this returns the probability function evaluated at a single point. """
        term1 = -D*partial(I,var=0,point=[x,t],order=2) # -D\partial_x^2 I(x,t)
        term2 = V*kd/k*partial(I,var=0,point=[x,t],order=1) # Vkd/k \partial_x I(x,t)
        term3 = k*I(x,t) 
        term4 = partial(I,var=1,point=[x,t],order=1) # \partial_t I(x,t)
        return term1+term2+term3+term4 
    
    return np.array([[Pi(xi,ti) for ti in t] for xi in x]) # first index is time, second is space


def mu(t,params):
    """
    analytical formula for the rate constant within the flux pdf. 
    This is the formula involving erfc and integrals of erfc and besselI(0/1,sqrt(...))
    """

    ke,kd,gam,Gam,V,T,Np,l,dt,stop,adaptive= params
    k = ke+kd
    D = Gam/gam**2


    def T(t):
        """
        This is the component of the formula not involving bessel functions or integration.
        """
        dif = np.sqrt(D*t/np.pi)*np.exp(-V**2*t/4/D)
        vel = V*t/2*erfc(-np.sqrt(V**2*t/4/D))
        return np.exp(-kd*t)*(dif+vel)

    def I1(u,t):
        """
        The component of the formula involving an integral over I0(x)
        """
        #exp = np.exp(-ke*t-(kd-ke)*u)
        #bes = i0(2*np.sqrt(ke*kd*u*(t-u)))
        # use $Iv(z)e^(bx) = e^{log(Ive(z))+z + bx}$ to prevent overflow
        z = 2*np.sqrt(ke*kd*u*(t-u))
        bes_exp = np.exp(np.log(ive(0,z))+z-ke*t-(kd-ke)*u)
        #bes = iv(0,2*np.sqrt(ke*kd*u*(t-u)))
        err = erfc(-np.sqrt(V**2*u/4/D))

        dif = (kd*u-kd/k/2)*np.sqrt(D/np.pi/u)*np.exp(-V**2*u/4/D) 
        #dif = (kd*u-1/2)*np.sqrt(D/np.pi/u)*np.exp(-V**2*u/4/D) 
        vel = V*kd/2*(u-1/k)*err  
        #return bes*exp*(dif+vel)
        return bes_exp*(dif+vel)

    def I2(u,t):
        """
        The component involving an integral over I1(x) 
        """
        #exp = np.exp(-ke*t-(kd-ke)*u)
        #bes = i1(2*np.sqrt(ke*kd*u*(t-u)))*np.sqrt(ke*kd*u/(t-u))
        z = 2*np.sqrt(ke*kd*u*(t-u))
        bes_exp = np.exp(np.log(ive(1,z))+z-ke*t-(kd-ke)*u)*np.sqrt(ke*kd*u/(t-u))
        #bes = iv(1,2*np.sqrt(ke*kd*u*(t-u)))*np.sqrt(ke*kd*u/(t-u))
        err = erfc(-np.sqrt(V**2*u/4/D))
        dif = np.sqrt(D*u/np.pi)*np.exp(-V**2*u/4/D)
        vel = V*u/2*err
        #return exp*bes*(dif+vel)
        return bes_exp*(dif+vel)
    K = lambda tt: quad(lambda u,t: I1(u,t)+I2(u,t),0,tt,args=(tt))
    I = [K(tt) for tt in t]
    #relErrs = [o[1]/o[0] for o in I]
    #if any([o>0.03 for o in relErrs]): print('nonConvergence.')
    return np.array([inte[0]+T(tt) for tt,inte in zip(t,I)]) # does not contain density yet.


def Q(t,params):
    """
    Gives the mean flux as mu/t (mu the rate constant).
    Essentially a simple wrapper wihch prevents division by 0.
    """

    ke,kd,gam,Gam,V,T,Np,l,dt,stop,adaptive= params
    k = ke+kd
    D = Gam/gam**2


    if t[0]==0:
        return np.array([0]+list(mu(t[1:],ke,kd,V,D)/t[1:]))
    else:
        return mu(t,params)/t


def var(t,ke,kd,V,D):
    """
    function to plot the analytical variance formula.
    This represents the variance of POSITION and gives the
    three range scaling for Pe>>1 at crossover t = 2D/V^2.
    Otherwise it gives one range for Pe<<1 when diffusion is strong.
    btw Pe = V^2/2/D/kd. It represents ADVECTION/DIFFUSION.
    """
    k = ke+kd
    term1 = 2*ke*kd*V**2/k**3*(t+1/k*np.exp(-k*t)-1/k)
    term2 = 2*ke*D*t/ka
    return term1+term2


def F(T,params,Nmax=200):
    """
    This is the probability distribution of position.
    """
    from scipy.special import gamma # the gamma function (factorial)
    # this is meant to take a single argument for T.
    T = [T]
    ke,kd,gam,Gam_i,V,_,Np,L,dt,stop,adaptive = params
    rho=Np/L#30
    Lam = rho*mu(T,params)#[:,None]
    n = np.arange(Nmax)
    q = n/T
    prob = (Lam)**n*np.exp(-Lam)/gamma(n+1)
    return q,prob

def Fasym(T,params,Nmax=25000):
    """
    This is the large T asymptotic probability distribution of position.
    """
    from scipy.special import gamma # the gamma function (factorial)
    from scipy.special import loggamma
    # this is meant to take a single argument for T.
    ke,kd,gam,Gam_i,V,_,Np,L,dt,stop,adaptive = params
    rho=Np/L#30
    Lam = rho*ke*V/(ke+kd)*T
    n = np.arange(Nmax)
    q = n/T
    #prob = (Lam)**n*np.exp(-Lam)/gamma(n+1)

    logprob = n*np.log(Lam)-Lam - loggamma(n+1)
    prob = np.exp(logprob)
    return q,prob

def Fvers(T,params):
    # versatile.. switches from regular to asymptotic forms depending on the magnitude of T
    if T>1e2:
        return Fasym(T,params)
    else:
        return F(T,params)
