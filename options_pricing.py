import numpy as np
from scipy.stats import norm


class BlackScholes:
    """
    Black-Scholes model for pricing European options & computing Greeks.

    Groups all pricing and risk formulas in one place. 
    It does not store any state : all methods depend only on the inputs provided (S, K, r, sigma, T, q).

    All methods are static : can be called directly from the class without creating an instance
    """

    @staticmethod
    def d1(S, K, r, sigma, T, q=0.0):
  
        # compute d1 using inputs
        return (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    @staticmethod
    def d2(S, K, r, sigma, T, q=0.0):
        # d2 derived directly from d1
        return BlackScholes.d1(S, K, r, sigma, T, q) - sigma * np.sqrt(T)

    @staticmethod
    def inputs_check(S, K, sigma, T):
        # garde fou
        return S > 0 and K > 0 and sigma > 0 and T > 0

    @staticmethod
    def BS_pricing(S, K, r, sigma, T, option_type="call", q=0.0):

        # return NaN if inputs are invalid
        if not BlackScholes.inputs_check(S, K, sigma, T):
            return np.nan

        # compute d1 and d2
        d1 = BlackScholes.d1(S, K, r, sigma, T, q)
        d2 = d1 - sigma * np.sqrt(T)

        # call price
        if option_type == "call":
            return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

        # put price
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

    @staticmethod
    def delta(S, K, r, sigma, T, option_type="call", q=0.0):
        
        if not BlackScholes.inputs_check(S, K, sigma, T):
            return np.nan

        d1 = BlackScholes.d1(S, K, r, sigma, T, q)

        # call delta
        if option_type == "call":
            return np.exp(-q * T) * norm.cdf(d1)

        # put delta
        return np.exp(-q * T) * (norm.cdf(d1) - 1)

    @staticmethod
    def gamma(S, K, r, sigma, T, q=0.0):

        if not BlackScholes.inputs_check(S, K, sigma, T):
            return np.nan

        d1 = BlackScholes.d1(S, K, r, sigma, T, q)

        # gamma call = gamma put
        return np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))

    @staticmethod
    def vega(S, K, r, sigma, T, q=0.0):
        
        if not BlackScholes.inputs_check(S, K, sigma, T):
            return np.nan

        d1 = BlackScholes.d1(S, K, r, sigma, T, q)

        # scaled by /100 to represent 1% vol move / vega call = vega put
        return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T) / 100.0

    @staticmethod
    def theta(S, K, r, sigma, T, option_type="call", q=0.0):

        if not BlackScholes.inputs_check(S, K, sigma, T):
            return np.nan

        d1 = BlackScholes.d1(S, K, r, sigma, T, q)
        d2 = d1 - sigma * np.sqrt(T)

        # common part of theta formula
        base = -(S * np.exp(-q * T) * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))

        # call theta
        if option_type == "call":
            return (base - r * K * np.exp(-r * T) * norm.cdf(d2)
                    + q * S * np.exp(-q * T) * norm.cdf(d1)) / 365.0

        # put theta
        return (base + r * K * np.exp(-r * T) * norm.cdf(-d2)
                - q * S * np.exp(-q * T) * norm.cdf(-d1)) / 365.0

    @staticmethod
    def rho(S, K, r, sigma, T, option_type="call", q=0.0):

        if not BlackScholes.inputs_check(S, K, sigma, T):
            return np.nan

        d2 = BlackScholes.d1(S, K, r, sigma, T, q) - sigma * np.sqrt(T)

        # call rho
        if option_type == "call":
            return K * T * np.exp(-r * T) * norm.cdf(d2) / 100.0

        # put rho
        return -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100.0

    @staticmethod
    def all_greeks(S, K, r, sigma, T, option_type="call", q=0.0):
        # gather all Greeks in one place using previously defined functions
        return {
            "Delta": BlackScholes.delta(S, K, r, sigma, T, option_type, q),
            "Gamma": BlackScholes.gamma(S, K, r, sigma, T, q),
            "Vega":  BlackScholes.vega(S, K, r, sigma, T, q),
            "Theta": BlackScholes.theta(S, K, r, sigma, T, option_type, q),
            "Rho":   BlackScholes.rho(S, K, r, sigma, T, option_type, q),
        }