# Readme
This is a temporary repository of the code used to develop the paper.
contact kpierce@alumni.ubc.ca with any questions.


## Description of simulation algorithm

To simulate the mobile-immobile model, a hybrid strategy is used. This alternates between motion and rest by evaluating the cumulative transition probabilities on a small timestep. It evaluates particle motions using the Euler-Mayarama algorithm.

The probability that a particle deposits in $\delta t$ is $1-\exp[-k_D \delta t]$. Similarly the probability that a particle entrains is $1-\exp[-k_D \delta t]$. The motion-rest switching just compares a random number $r \in [0,1]$ with these probabilities to evaluate whether an event occurs in $\delta t$ or not.

The Euler-Mayarama algorithm is
$$ x(t+\delta t) = x(t) + v(t)\delta t$$
$$ v(t+\delta t) = v(t) + \sqrt{2\Gamma \delta t) + F(v(t))\delta t $$

Alternation between these two procedures simulates a single stochastic trajectory. Aggregation of many trajectories produces probability characteristics.


## Contents:

* `figure*i*.ipynb` are jupyter notebooks which make each plot in the paper.
These notebooks draw upon the code from the `*.py` modules.

* `flipflopanalytical.py` implements the analytical functions derived in the manuscript

* `flipflopinertial.py` conducts monte carlo simulations of individual grains obeying the pair of Langevin equations in the manuscript

* `fluxsim.py` runs an ensemble of monte carlo simulations and counts the rate of particles crossing a control surface in order to compute the flux.

