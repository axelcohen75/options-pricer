# Options Pricer

An interactive web app to price European options and visualise option strategies, built with Python and Dash. Public link : https://options-pricer-856t.onrender.com/

## Pricing model

All pricing is done with the **Black-Scholes-Merton** model. The implementation lives in `options_pricing.py` and covers:

- Option price (call & put)
- The five Greeks: Delta, Gamma, Vega, Theta, Rho

## Project structure

```
main.py               — Dash app: layout, tabs, callbacks
options_pricing.py    — Black-Scholes pricing engine
gas_storage_pricing.py — Gas storage valuation engine
requirements.txt
```

## Tabs

- **Options Pricer** — price a single European option, see its Greeks and payoff diagram
- **Options Strategy Pricer** — select a predefined strategy (straddle, iron condor, etc.), enter strikes, get the combined payoff and net Greeks
- **Gas Storage Pricing** — value a natural gas storage facility using a linear programming approach on a live Henry Hub forward curve fetched from NYMEX futures

## How to run

```
pip install -r requirements.txt
python main.py
```

Then open http://localhost:8050 in your browser.
