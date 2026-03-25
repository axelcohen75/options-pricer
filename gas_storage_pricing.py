import numpy as np
import pandas as pd
from scipy.optimize import linprog
from datetime import datetime
from dateutil.relativedelta import relativedelta
import yfinance as yf

def fetch_henry_hub_curve(n_months=12):
    """
    Builds a simplified HH forward curve by:
    - Generating NYMEX futures tickers for each months
    - Pulling latest prices from yfinance
    - Returning a df (delivery_date, price)
    """

    MONTH_CODES = {
        1: "F", 2: "G", 3: "H", 4: "J", 5: "K", 6: "M",
        7: "N", 8: "Q", 9: "U", 10: "V", 11: "X", 12: "Z",
    }

    today = datetime.today()
    records = []

    for i in range(1, n_months + 1):

        # Use proper month arithmetic to avoid day-count drift
        target = today + relativedelta(months=i)
        month = target.month
        year = target.year

        # Build futures ticker (Yahoo Finance NYMEX format)
        ticker = f"NG{MONTH_CODES[month]}{str(year)[-2:]}.NYM"

        try:
            data = yf.Ticker(ticker).history(period="1mo")
            if not data.empty:
                price = float(data["Close"].iloc[-1])

                # Delivery date = first day of delivery month
                delivery_date = datetime(year, month, 1)

                records.append({
                    "ticker": ticker,
                    "delivery_date": delivery_date,
                    "price": price,
                })
        except Exception:
            continue

    return pd.DataFrame(records)


class GasStorageIntrinsicValue:
    """
    The class prices a gas storage (instrinsic values only for the moment):
    the goal is to optimize injection/withdrawal decisions against the forward curve.
    (I.e Buy gas when cheap → store it → sell when expensive)

    This is solved as a linear programming problem.
    """

    def __init__(self, capacity, initial_inventory, max_injection_rate,
                 max_withdrawal_rate, injection_cost, withdrawal_cost):

        # Physical parameters of the storage
        self.capacity = capacity
        self.initial_inventory = initial_inventory

        # Operational constraints (flows per month)
        self.max_injection_rate = max_injection_rate
        self.max_withdrawal_rate = max_withdrawal_rate

        # Costs of operations
        self.injection_cost = injection_cost
        self.withdrawal_cost = withdrawal_cost

    def storage_price(self, df):
        """
        Inputs: Forward prices per month

        Outputs: Optimal injection/withdrawal schedule & Maximum achievable profit

        The function chooses volumes to maximize profit under constraints.
        """

        # extract forward prices and dates
        prices = df["price"].values.astype(float)
        dates = df["delivery_date"].values
        N = len(prices)

        # garde fou
        if N < 2:
            return 0.0, pd.DataFrame()

        # build cost vector passed to linprog
        # first N entries : inject variables
        # next N entries : withdraw variables
        c = np.concatenate([
            prices + self.injection_cost, # cost of injecting (buy + cost)
            -prices + self.withdrawal_cost, # negative revenue from withdrawing
        ])

        # remaining capacity available above initial inventory
        cap_slack = self.capacity - self.initial_inventory

        A_rows, b_rows = [], []

        for t in range(N):

            # ensure inventory never exceeds capacity
            # cumulative inject - cumulative withdraw ≤ remaining capacity
            row_up = np.zeros(2 * N)
            row_up[:t + 1] = 1.0
            row_up[N:N + t + 1] = -1.0
            A_rows.append(row_up)
            b_rows.append(cap_slack)

            # ensure inventory never goes negative
            # cannot withdraw more than what is in storage
            row_lo = np.zeros(2 * N)
            row_lo[:t + 1] = -1.0
            row_lo[N:N + t + 1] = 1.0
            A_rows.append(row_lo)
            b_rows.append(self.initial_inventory)
        
        # convert list of constraints into matrix form for solver
        A_ub = np.array(A_rows)
        b_ub = np.array(b_rows)

        # terminal constraint: final inventory must equal initial inventory
        # avoids artificial profit from ending empty or full
        A_eq = np.zeros((1, 2 * N))
        A_eq[0, :N] = 1.0
        A_eq[0, N:] = -1.0
        b_eq = np.array([0.0])

        # flow constraints: limit how much can be injected or withdrawn each month
        bounds = (
            [(0, self.max_injection_rate)] * N
            + [(0, self.max_withdrawal_rate)] * N
        )

    
        # solve the optimization: find the best injection/withdrawal strategy under constraints
        result = linprog(
            c,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method="highs"
        )

        # return empty result if error
        if not result.success:
            return 0.0, pd.DataFrame()

        # extract optimal decisions
        inject = result.x[:N]
        withdraw = result.x[N:]

        # rebuild inventory path over time
        inventory = np.zeros(N)
        inventory[0] = self.initial_inventory + inject[0] - withdraw[0]

        for t in range(1, N):
            inventory[t] = inventory[t - 1] + inject[t] - withdraw[t]

        # compute total profit as : revenues from sales - cost of purchases - operational costs
        value = float(np.sum(
            prices * withdraw
            - prices * inject
            - self.injection_cost * inject
            - self.withdrawal_cost * withdraw
        ))

        # build output schedule
        schedule = pd.DataFrame({
            "delivery_date": dates,
            "price": prices,
            "inject": inject,
            "withdraw": withdraw,
            "inventory": inventory,
        })

        return value, schedule