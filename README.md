# Near-zero step size in ODE integration

Pressure swing adsorption (PSA) processes are simulated by converting the PDE system into ODEs (through discretization such as the finite volume method) and solving them with ODE solvers. Previously, this has already been implemented in MATLAB by [Leperi et al.](https://doi.org/10.1021/acs.iecr.5b03122) and [Yancy-Caballero et al.](https://doi.org/10.1039/D0ME00060D) The MATLAB codes are available on [GitHub](https://github.com/PEESEgroup/PSA).

Currently I am attempting to adapt the MATLAB codes into Python and solve the PDE/ODE problem using Python libraries. However, when solving the discretized ODE system using `scipy.integrate.solve_ivp` with the `BDF` method, in some cases, I cannot get the final results because of the extremely small (near-zero) step size, which leads to an "endless" ODE integration.

To reproduce the issue, I have streamlined the Python codes and only kept the necessary parts, please see `psa_cycle.py`. `PSA_PDE.pdf` summarized the PDE system and related equations used for PSA simulation.
