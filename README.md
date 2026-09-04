# Monte-Carlo-Options-Pricer

A Monte Carlo pricer for European options under Geometric Brownian Motion. Simulates asset paths, prices calls and puts by averaging discounted payoffs, and uses antithetic variates to reduce the variance of the estimator. Validated against the closed-form Black-Scholes solution across a range of strikes, with the convergence rate measured empirically against the number of simulated paths.

`pricer.py` contains the core library. `Monte_Carlo_Options_Pricer.ipynb` walks through the model, the validation, and the variance reduction, with worked figures throughout.
