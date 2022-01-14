This is a temporary repository of the code used to develop the paper.
KP is currently cleaning and commenting this, so that it will accomodate outside use.
Contact: kpierce@alumni.ubc.ca


Contents:

`figure*i*.ipynb` are jupyter notebooks which make each plot in the paper.
These notebooks draw upon the code from the `*.py` modules.

`flipflopanalytical.py` implements the analytical functions derived in the manuscript

`flipflopinertial.py` conducts monte carlo simulations of individual grains obeying the pair of Langevin equations in the manuscript

`fluxsim.py` runs an ensemble of monte carlo simulations and counts the rate of particles crossing a control surface in order to compute the flux.

