import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from dataclasses import dataclass
import pandas as pd

rng = np.random.default_rng(91006)


def call_payoff(S_T, K):
    return (np.maximum(S_T-K, 0))


def put_payoff(S_T, K):
    return (np.maximum(K-S_T, 0))


def straddle_payoff(S_T, K):
    return np.abs(S_T-K)


def simulate_terminal_prices(S0, r, sigma, T, n_paths, rng, antithetic=False):
    if antithetic:
        if n_paths % 2 == 1:
            raise ValueError("n_paths must be even")
        else:
            M = n_paths // 2
            Z = rng.standard_normal(M)
            Z = np.concatenate((Z, -Z))
            return (S0*np.exp((r-0.5*sigma**2)*T + (sigma * np.sqrt(T)*Z)))
    else:
        Z = rng.standard_normal(n_paths)
        return (S0*np.exp((r-0.5*sigma**2)*T + (sigma * np.sqrt(T)*Z)))


def simulate_paths(S0, r, sigma, T, n_steps, n_paths, rng):
    dt = T/n_steps
    paths = np.repeat(S0, n_paths)
    for i in range(n_steps):
        Z = rng.standard_normal(n_paths)
        paths = np.vstack(
            [paths, paths[i]*np.exp((r-0.5*sigma**2)*dt + (sigma * np.sqrt(dt)*Z))])

    return (paths)


@dataclass
class MCResult:
    price: float
    std_error: float
    n_paths: int

    def ci95(self):
        return (self.price - 1.96*self.std_error, self.price+1.96*self.std_error)

    def overview(self):
        return f"95% CI, {self.ci95()}, price: {self.price}"


def mc_european_price(S0, K, T, r, sigma, n_paths, rng, option_type="call", antithetic=False):
    if option_type not in ("call", "put"):
        raise ValueError("put or call?")
    if option_type == 'call':
        payoffs = call_payoff(simulate_terminal_prices(
            S0, r, sigma, T, n_paths, rng, antithetic), K)
        discounted = np.exp(-r*T)*payoffs

    elif option_type == "put":
        payoffs = put_payoff(simulate_terminal_prices(
            S0, r, sigma, T, n_paths, rng, antithetic), K)
        discounted = np.exp(-r*T)*payoffs

    if antithetic:
        averaged_discounted = np.zeros(n_paths//2)
        for i in range(n_paths//2):
            averaged_discounted[i] = 0.5 * \
                (discounted[i] + discounted[i+n_paths//2])
        return (MCResult(np.mean(discounted), np.std(averaged_discounted, ddof=1)/np.sqrt(n_paths//2), n_paths))

    else:
        return (MCResult(np.mean(discounted), np.std(discounted, ddof=1)/np.sqrt(n_paths), n_paths))


def black_scholes_price(S0, K, T, r, sigma, option_type="call"):
    if T <= 0:
        return np.maximum(S0-K, 0.0) if option_type == "call" else np.maximum(K-S0, 0.0)
    d2 = (np.log(S0/K) + (r - 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d1 = d2 + sigma*np.sqrt(T)
    if option_type == "call":
        return S0*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    return K*np.exp(-r*T)*norm.cdf(-d2) - S0*norm.cdf(-d1)


def fit_convergence_rate(path_counts, errors):
    x = np.log(path_counts)
    y = np.log(errors)
    n = len(x)
    slope = (n*np.sum(x*y) - np.sum(x)*np.sum(y)) / \
        (n*np.sum(x**2) - np.sum(x)**2)
    intercept = (np.sum(y) - slope*np.sum(x)) / n
    return slope, np.exp(intercept)


def convergence_study(S0, K, T, r, sigma, path_counts, n_reps, rng, option_type="call"):
    rows = []
    exact = black_scholes_price(S0, K, T, r, sigma, option_type)

    for M in path_counts:
        Prices = []
        SE = []
        for n in range(n_reps):

            result = mc_european_price(S0, K, T, r, sigma, M, rng, option_type)
            Prices.append(result.price)
            SE.append(result.std_error)
        SE = np.array(SE)
        Prices = np.array(Prices)
        rows.append({"n_paths": M, "mean_price": np.mean(Prices), "mean_se": np.mean(
            SE), "exact": exact, "bias": np.mean(Prices)-exact, "RMSE": np.sqrt(np.mean((Prices-exact)**2))})
    df = pd.DataFrame(rows)
    return (df)
