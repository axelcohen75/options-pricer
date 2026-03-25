import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from options_pricing import BlackScholes

bs = BlackScholes

# ---------------------------------------------------------------------------
# utilities
# ---------------------------------------------------------------------------

def format_num(val, decimals=4):

    # Formats a number for display and returns "—" if it's None or NaN
    try:
        return "—" if val is None or np.isnan(val) else f"{val:.{decimals}f}"
    except Exception:
        return "—"

def safe_float(val, default=0.0):

    # Converts user input to a float, returns default on failure
    try:
        v = float(val)
        return v if np.isfinite(v) else default
    except Exception:
        return default

# Base style shared by all Plotly charts (dark theme)
PLOT_BASE = dict(
    paper_bgcolor="#1a1a2e",
    plot_bgcolor="#16213e",
    font=dict(color="#e0e0e0"),
    xaxis=dict(gridcolor="#2a2a4a"),
    yaxis=dict(gridcolor="#2a2a4a"),
    margin=dict(l=50, r=20, t=30, b=40),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
)

# ---------------------------------------------------------------------------
# Strategy definitions
# ---------------------------------------------------------------------------

STRATEGY_STRIKE_CONFIG = {
    "long_straddle":    [("ATM Strike",  1.00)],
    "short_straddle":   [("ATM Strike",  1.00)],
    "long_strangle":    [("Put Strike",  0.95), ("Call Strike",  1.05)],
    "short_strangle":   [("Put Strike",  0.95), ("Call Strike",  1.05)],
    "bull_call_spread": [("Low Strike",  1.00), ("High Strike",  1.05)],
    "bear_put_spread":  [("High Strike", 1.00), ("Low Strike",   0.95)],
    "risk_reversal":    [("Put Strike",  0.95), ("Call Strike",  1.05)],
    "collar":           [("Put Strike",  0.95), ("Call Strike",  1.05)],
    "butterfly":        [("Low Strike",  0.95), ("Mid Strike",   1.00), ("High Strike",  1.05)],
    "iron_condor":      [("Long Put",    0.90), ("Short Put",    0.95), ("Short Call",   1.05), ("Long Call",  1.10)],
}

STRATEGY_LABELS = {
    "long_straddle":    "Long Straddle",
    "short_straddle":   "Short Straddle",
    "long_strangle":    "Long Strangle",
    "short_strangle":   "Short Strangle",
    "bull_call_spread": "Bull Call Spread",
    "bear_put_spread":  "Bear Put Spread",
    "risk_reversal":    "Risk Reversal",
    "collar":           "Collar",
    "butterfly":        "Butterfly Spread",
    "iron_condor":      "Iron Condor",
}

def build_legs(strategy, strikes, sigma_pct, T_days):
    K = [safe_float(s, 100.0) for s in (strikes or [])]
    base = {"sigma_pct": sigma_pct, "T_days": T_days}

    templates = {
        "long_straddle":    [("call", 0, 1),  ("put",  0, 1)],
        "short_straddle":   [("call", 0, -1), ("put",  0, -1)],
        "long_strangle":    [("put",  0, 1),  ("call", 1, 1)],
        "short_strangle":   [("put",  0, -1), ("call", 1, -1)],
        "bull_call_spread": [("call", 0, 1),  ("call", 1, -1)],
        "bear_put_spread":  [("put",  0, 1),  ("put",  1, -1)],
        "risk_reversal":    [("put",  0, -1), ("call", 1, 1)],
        "collar":           [("put",  0, 1),  ("call", 1, -1)],
        "butterfly":        [("call", 0, 1),  ("call", 1, -2), ("call", 2, 1)],
        "iron_condor":      [("put",  0, 1),  ("put",  1, -1), ("call", 2, -1), ("call", 3, 1)],
    }

    if strategy not in templates or len(K) < len(STRATEGY_STRIKE_CONFIG.get(strategy, [])):
        return []

    return [{"type": ot, "K": K[k_idx], "qty": qty, **base}
            for ot, k_idx, qty in templates[strategy]]

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Options Pricer", layout="wide")
st.title("Options Pricer")

tab_pricer, tab_strategy, tab_storage = st.tabs(["Options Pricing", "Strategy", "Gas Storage"])

# ---------------------------------------------------------------------------
# Options Pricing tab
# ---------------------------------------------------------------------------

with tab_pricer:
    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.markdown("#### Parameters")
        c1, c2 = st.columns(2)
        S         = c1.number_input("Spot (S)",       value=100.0, min_value=0.01, step=0.5)
        K         = c2.number_input("Strike (K)",     value=100.0, min_value=0.01, step=0.5)
        r_pct     = c1.number_input("Rate r (%)",     value=5.0,   min_value=0.0,  step=0.1)
        q_pct     = c2.number_input("Div q (%)",      value=0.0,   min_value=0.0,  step=0.1)
        sigma_pct = c1.number_input("Vol σ (%)",      value=20.0,  min_value=0.1,  step=0.5)
        T_days    = c2.number_input("Expiry (days)",  value=30,    min_value=1,    step=1)
        otype     = st.radio("Type", ["call", "put"], horizontal=True)

    # convert inputs
    r     = r_pct / 100
    q     = q_pct / 100
    sigma = sigma_pct / 100
    T     = T_days / 365

    # compute price and Greeks
    price  = bs.BS_pricing(S, K, r, sigma, T, otype, q)
    greeks = bs.all_greeks(S, K, r, sigma, T, otype, q)

    with col_right:

        # metrics row
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Price", format_num(price))
        m2.metric("Delta", format_num(greeks["Delta"]))
        m3.metric("Gamma", format_num(greeks["Gamma"]))
        m4.metric("Vega",  format_num(greeks["Vega"]))
        m5.metric("Theta", format_num(greeks["Theta"]))
        m6.metric("Rho",   format_num(greeks["Rho"]))

        # payoff diagram
        s_range       = np.linspace(max(0.01, S * 0.5), S * 1.5, 300)
        expiry_payoff = (np.maximum(s_range - K, 0) if otype == "call"
                         else np.maximum(K - s_range, 0))
        current_vals  = [safe_float(bs.BS_pricing(s, K, r, sigma, T, otype, q), 0.0)
                         for s in s_range]

        fig_payoff = go.Figure()
        fig_payoff.add_trace(go.Scatter(x=s_range, y=expiry_payoff, name="At Expiry",
                                        line=dict(dash="dot", color="#e0e0e0")))
        fig_payoff.add_trace(go.Scatter(x=s_range, y=current_vals, name="Current Value",
                                        line=dict(color="#2196f3")))
        fig_payoff.add_vline(x=K, line=dict(dash="dot"), annotation_text=f"K={K}")
        fig_payoff.add_vline(x=S, line=dict(dash="dot", color="gray"), annotation_text=f"S={S}")
        fig_payoff.update_layout(**PLOT_BASE, xaxis_title="Spot", yaxis_title="P&L",
                                 title="Payoff Diagram")
        st.plotly_chart(fig_payoff, use_container_width=True)

        # Greeks curves
        st.markdown("#### Greeks Curves")
        gc1, gc2 = st.columns(2)
        xaxis    = gc1.selectbox("X-axis", ["Spot Price", "Volatility σ", "Time to Expiry"],
                                 key="sp_xaxis")
        selected = gc2.multiselect("Greeks", ["Delta", "Gamma", "Vega", "Theta", "Rho"],
                                   default=["Delta", "Gamma", "Vega"], key="sp_greeks")

        if selected:
            if xaxis == "Spot Price":
                x_vals, x_label, x_key = np.linspace(max(0.01, S * 0.5), S * 1.5, 200), "Spot Price", "S"
            elif xaxis == "Volatility σ":
                x_vals, x_label, x_key = np.linspace(0.01, 1.0, 200), "Volatility", "sigma"
            else:
                x_vals, x_label, x_key = np.linspace(1/365, 2.0, 200), "Time to Expiry (years)", "T"

            def compute_greek(greek_name, x):
                # pick which variable is moving, others stay fixed
                sv   = x if x_key == "S"     else S
                sigv = x if x_key == "sigma" else sigma
                Tv   = x if x_key == "T"     else T
                if sv <= 0 or sigv <= 0 or Tv <= 0:
                    return np.nan
                return bs.all_greeks(sv, K, r, sigv, Tv, otype, q)[greek_name]

            n = len(selected)
            fig_greeks = make_subplots(rows=n, cols=1, shared_xaxes=True, vertical_spacing=0.04)
            for i, greek in enumerate(selected, 1):
                fig_greeks.add_trace(go.Scatter(x=x_vals,
                                                y=[compute_greek(greek, x) for x in x_vals],
                                                name=greek, line=dict(color="#2196f3")),
                                     row=i, col=1)
                fig_greeks.update_yaxes(title_text=greek, row=i, col=1, gridcolor="#2a2a4a")
            fig_greeks.update_xaxes(title_text=x_label, row=n, col=1, gridcolor="#2a2a4a")
            fig_greeks.update_layout(paper_bgcolor="#1a1a2e", plot_bgcolor="#16213e",
                                     font=dict(color="#e0e0e0"),
                                     margin=dict(l=60, r=20, t=10, b=40),
                                     legend=dict(bgcolor="rgba(0,0,0,0)"),
                                     height=max(300, n * 150))
            st.plotly_chart(fig_greeks, use_container_width=True)

# ---------------------------------------------------------------------------
# Strategy tab
# ---------------------------------------------------------------------------

with tab_strategy:
    sc_left, sc_right = st.columns([1, 2])

    with sc_left:
        st.markdown("#### Market Parameters")
        sc1, sc2  = st.columns(2)
        st_S      = sc1.number_input("Spot (S)",      value=100.0, min_value=0.01, step=0.5,  key="st_S")
        st_r      = sc2.number_input("Rate r (%)",    value=5.0,   min_value=0.0,  step=0.1,  key="st_r")
        st_q      = sc1.number_input("Div q (%)",     value=0.0,   min_value=0.0,  step=0.1,  key="st_q")
        st_sigma  = sc2.number_input("Vol σ (%)",     value=20.0,  min_value=0.1,  step=0.5,  key="st_sigma")
        st_T      = st.number_input("Expiry (days)",  value=30,    min_value=1,    step=1,    key="st_T")

        st.markdown("#### Strategy")
        strategy_options = ["— Select —"] + list(STRATEGY_LABELS.values())
        strategy_label   = st.selectbox("Strategy", strategy_options, key="st_strategy")

        # reverse lookup from label to key
        strategy_key = next((k for k, v in STRATEGY_LABELS.items()
                             if v == strategy_label), None)

        # strike inputs — shown only when a strategy is selected
        strikes = []
        if strategy_key and strategy_key in STRATEGY_STRIKE_CONFIG:
            strike_defs = STRATEGY_STRIKE_CONFIG[strategy_key]
            st.markdown("**Strikes**")
            n_defs = len(strike_defs)
            strike_cols = st.columns(min(n_defs, 2))
            for i, (lbl, k_pct) in enumerate(strike_defs):
                val = strike_cols[i % 2].number_input(
                    lbl, value=round(st_S * k_pct, 1), min_value=0.01,
                    step=0.5, key=f"st_K{i}"
                )
                strikes.append(val)

    with sc_right:
        if not strategy_key:
            st.info("Select a strategy above to price it.")
        else:
            st_r_ = st_r / 100
            st_q_ = st_q / 100
            legs  = build_legs(strategy_key, strikes, st_sigma, st_T)

            if not legs:
                st.warning("Not enough strikes defined.")
            else:
                net_cost = net_delta = net_gamma = net_vega = net_theta = net_rho = 0.0

                for leg in legs:
                    sig = max(leg["sigma_pct"] / 100, 1e-6)
                    Tv  = max(leg["T_days"] / 365, 1e-6)
                    qty = leg["qty"]
                    ot  = leg["type"]
                    K_  = leg["K"]

                    # accumulate price and Greeks weighted by position size
                    net_cost  += qty * safe_float(bs.BS_pricing(st_S, K_, st_r_, sig, Tv, ot, st_q_), 0.0)
                    net_delta += qty * safe_float(bs.delta(st_S, K_, st_r_, sig, Tv, ot, st_q_), 0.0)
                    net_gamma += qty * safe_float(bs.gamma(st_S, K_, st_r_, sig, Tv, st_q_), 0.0)
                    net_vega  += qty * safe_float(bs.vega(st_S, K_, st_r_, sig, Tv, st_q_), 0.0)
                    net_theta += qty * safe_float(bs.theta(st_S, K_, st_r_, sig, Tv, ot, st_q_), 0.0)
                    net_rho   += qty * safe_float(bs.rho(st_S, K_, st_r_, sig, Tv, ot, st_q_), 0.0)

                # metrics row
                sm1, sm2, sm3, sm4, sm5, sm6 = st.columns(6)
                sm1.metric("Net Cost", format_num(net_cost))
                sm2.metric("Delta",    format_num(net_delta))
                sm3.metric("Gamma",    format_num(net_gamma))
                sm4.metric("Vega",     format_num(net_vega))
                sm5.metric("Theta",    format_num(net_theta))
                sm6.metric("Rho",      format_num(net_rho))

                # payoff diagram
                s_range     = np.linspace(max(0.01, st_S * 0.5), st_S * 1.5, 400)
                expiry_pnl  = np.zeros(len(s_range))
                current_pnl = np.zeros(len(s_range))
                colors = ["#2196f3", "#42a5f5", "#90caf9", "#bbdefb"]
                fig_strat = go.Figure()

                for i, leg in enumerate(legs):
                    sig = max(leg["sigma_pct"] / 100, 1e-6)
                    Tv  = max(leg["T_days"] / 365, 1e-6)
                    qty = leg["qty"]
                    ot  = leg["type"]
                    K_  = leg["K"]
                    p0  = safe_float(bs.BS_pricing(st_S, K_, st_r_, sig, Tv, ot, st_q_), 0.0)

                    # payoff at expiry minus initial premium paid
                    leg_exp = (qty * (np.maximum(s_range - K_, 0) if ot == "call"
                                      else np.maximum(K_ - s_range, 0)) - qty * p0)
                    expiry_pnl += leg_exp

                    # current P&L = value today - initial price
                    curr = np.array([safe_float(bs.BS_pricing(s, K_, st_r_, sig, Tv, ot, st_q_), 0.0)
                                     for s in s_range])
                    current_pnl += qty * (curr - p0)

                    # plot individual leg payoff (faint dotted line)
                    fig_strat.add_trace(go.Scatter(x=s_range, y=leg_exp,
                                                   name=f"{qty:+g} {ot} K={K_}",
                                                   line=dict(dash="dot", color=colors[i % len(colors)], width=1),
                                                   opacity=0.5))

                fig_strat.add_trace(go.Scatter(x=s_range, y=expiry_pnl, name="At Expiry",
                                               line=dict(color="#e0e0e0", width=2)))
                fig_strat.add_trace(go.Scatter(x=s_range, y=current_pnl, name="Current Value",
                                               line=dict(color="#2196f3", width=2)))

                # horizontal zero line → break-even reference
                fig_strat.add_hline(y=0, line=dict(color="gray", dash="dot", width=1))

                # vertical line at current spot
                fig_strat.add_vline(x=st_S, line=dict(color="gray", dash="dot"),
                                    annotation_text=f"S={st_S}")

                # vertical lines at each strike
                for leg in legs:
                    fig_strat.add_vline(x=leg["K"], line=dict(dash="dot", color="#555", width=1))

                # time value = current value minus expiry intrinsic value
                time_value = current_pnl - expiry_pnl
                fig_strat.add_trace(go.Scatter(x=s_range, y=time_value, name="Time Value",
                                               line=dict(color="#ffd54f", width=1, dash="dot")))

                fig_strat.update_layout(**PLOT_BASE,
                                        xaxis_title="Spot at Expiry",
                                        yaxis_title="P&L",
                                        title="Payoff Diagram")
                st.plotly_chart(fig_strat, use_container_width=True)

                # Greeks curves
                st.markdown("#### Greeks Curves")
                gc1s, gc2s  = st.columns(2)
                xaxis_st    = gc1s.selectbox("X-axis",
                                             ["Spot Price", "Volatility σ", "Time to Expiry"],
                                             key="st_xaxis")
                selected_st = gc2s.multiselect("Greeks",
                                               ["Delta", "Gamma", "Vega", "Theta", "Rho"],
                                               default=["Delta", "Gamma", "Vega"],
                                               key="st_greeks")

                if selected_st:
                    sigma_dec = st_sigma / 100
                    T_yr      = st_T / 365

                    if xaxis_st == "Spot Price":
                        x_vals, x_label, x_key = (np.linspace(max(0.01, st_S * 0.5), st_S * 1.5, 200),
                                                   "Spot Price", "S")
                    elif xaxis_st == "Volatility σ":
                        x_vals, x_label, x_key = np.linspace(0.01, 1.0, 200), "Volatility", "sigma"
                    else:
                        x_vals, x_label, x_key = (np.linspace(1/365, 2.0, 200),
                                                   "Time to Expiry (years)", "T")

                    def net_greek(greek_name, x):
                        total = 0.0
                        for leg in legs:
                            sig = max(leg["sigma_pct"] / 100, 1e-6)
                            Tv  = max(leg["T_days"] / 365, 1e-6)
                            qty = leg["qty"]
                            ot  = leg["type"]
                            K_  = leg["K"]
                            sv   = x    if x_key == "S"     else st_S
                            sigv = x    if x_key == "sigma" else sig
                            Tvv  = x    if x_key == "T"     else Tv
                            if sv <= 0 or sigv <= 0 or Tvv <= 0:
                                return np.nan
                            if greek_name == "Delta": total += qty * bs.delta(sv, K_, st_r_, sigv, Tvv, ot, st_q_)
                            if greek_name == "Gamma": total += qty * bs.gamma(sv, K_, st_r_, sigv, Tvv, st_q_)
                            if greek_name == "Vega":  total += qty * bs.vega(sv, K_, st_r_, sigv, Tvv, st_q_)
                            if greek_name == "Theta": total += qty * bs.theta(sv, K_, st_r_, sigv, Tvv, ot, st_q_)
                            if greek_name == "Rho":   total += qty * bs.rho(sv, K_, st_r_, sigv, Tvv, ot, st_q_)
                        return total

                    n = len(selected_st)
                    fig_st_greeks = make_subplots(rows=n, cols=1, shared_xaxes=True,
                                                  vertical_spacing=0.04)
                    for i, greek in enumerate(selected_st, 1):
                        fig_st_greeks.add_trace(
                            go.Scatter(x=x_vals, y=[net_greek(greek, x) for x in x_vals],
                                       name=greek, line=dict(color="#2196f3")),
                            row=i, col=1)
                        fig_st_greeks.update_yaxes(title_text=greek, row=i, col=1,
                                                   gridcolor="#2a2a4a")
                    fig_st_greeks.update_xaxes(title_text=x_label, row=n, col=1,
                                               gridcolor="#2a2a4a")
                    fig_st_greeks.update_layout(paper_bgcolor="#1a1a2e", plot_bgcolor="#16213e",
                                                font=dict(color="#e0e0e0"),
                                                margin=dict(l=60, r=20, t=10, b=40),
                                                legend=dict(bgcolor="rgba(0,0,0,0)"),
                                                height=max(300, n * 150))
                    st.plotly_chart(fig_st_greeks, use_container_width=True)

# ---------------------------------------------------------------------------
# Gas Storage tab
# ---------------------------------------------------------------------------

with tab_storage:
    gs_left, gs_right = st.columns([1, 2])

    with gs_left:
        st.markdown("#### Storage Parameters")
        gs1, gs2    = st.columns(2)
        gs_capacity = gs1.number_input("Capacity (MMBtu)",         value=1_000_000, min_value=1,   step=10_000)
        gs_init_inv = gs2.number_input("Initial Inventory",         value=0,         min_value=0,   step=10_000)
        gs_max_inj  = gs1.number_input("Max Injection/mo",          value=300_000,   min_value=1,   step=10_000)
        gs_max_wdw  = gs2.number_input("Max Withdrawal/mo",         value=500_000,   min_value=1,   step=10_000)
        gs_inj_cost = gs1.number_input("Injection Cost ($/MMBtu)",  value=0.01,      min_value=0.0, step=0.001, format="%.3f")
        gs_wdw_cost = gs2.number_input("Withdrawal Cost ($/MMBtu)", value=0.01,      min_value=0.0, step=0.001, format="%.3f")
        n_months    = st.radio("Months to Fetch", [12, 24], horizontal=True)

        if st.button("Fetch Forward Curve", use_container_width=True):
            from gas_storage_pricing import fetch_henry_hub_curve
            with st.spinner("Fetching..."):
                try:
                    df = fetch_henry_hub_curve(n_months=n_months)
                    if df.empty:
                        st.warning("No data returned — markets may be closed.")
                        st.session_state["gs_curve"] = None
                    else:
                        st.session_state["gs_curve"] = df
                        st.success(f"Fetched {len(df)} contracts — last: {df['ticker'].iloc[-1]}")
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.session_state["gs_curve"] = None

        curve_ready = st.session_state.get("gs_curve") is not None
        if st.button("Calculate Intrinsic Value", use_container_width=True,
                     disabled=not curve_ready):
            from gas_storage_pricing import GasStorageIntrinsicValue
            df = st.session_state["gs_curve"]

            # create storage object with user inputs
            storage = GasStorageIntrinsicValue(
                capacity=float(gs_capacity),
                initial_inventory=float(gs_init_inv),
                max_injection_rate=float(gs_max_inj),
                max_withdrawal_rate=float(gs_max_wdw),
                injection_cost=float(gs_inj_cost),
                withdrawal_cost=float(gs_wdw_cost),
            )

            # run optimization : returns intrinsic value + optimal schedule
            value, schedule = storage.storage_price(df)
            st.session_state["gs_results"] = (value, schedule)

        st.markdown("---")
        st.markdown("#### How it works")
        st.caption(
            "This model computes the intrinsic value of a gas storage asset by solving "
            "an optimization problem over the current Henry Hub forward price curve. "
            "Data are fetched live from Yahoo Finance (NYMEX)."
        )
        st.caption(
            "The model determines the optimal monthly injection and withdrawal volumes "
            "that maximize total profit."
        )
        st.markdown("**Objective**")
        st.caption("Maximize: withdrawal revenue − injection cost − operational costs. "
                   "Internally expressed as a cost minimization (linear program).")
        st.markdown("**Constraints**")
        for constraint in [
            "Inventory cannot exceed maximum capacity.",
            "Storage level can never go below zero.",
            "Injection and withdrawal volumes are capped each month.",
            "Inventory evolves from cumulative injections and withdrawals.",
            "Final inventory must equal initial inventory.",
        ]:
            st.caption(f"• {constraint}")

    with gs_right:

        # Henry Hub forward curve chart
        if st.session_state.get("gs_curve") is not None:
            df = st.session_state["gs_curve"]
            fig_curve = go.Figure()
            fig_curve.add_trace(go.Scatter(
                x=df["delivery_date"].astype(str), y=df["price"],
                mode="lines+markers", line=dict(color="#2196f3"), marker=dict(size=6),
                name="HH Futures",
            ))
            fig_curve.update_layout(**PLOT_BASE, xaxis_title="Delivery Month",
                                    yaxis_title="Price ($/MMBtu)",
                                    title="Henry Hub Forward Curve")
            st.plotly_chart(fig_curve, use_container_width=True)

        # results + schedule chart
        if "gs_results" in st.session_state:
            value, schedule = st.session_state["gs_results"]

            if schedule.empty:
                st.error("Infeasible optimization.")
            else:
                # compute total injected and withdrawn volumes
                total_inj = float(schedule["inject"].sum())
                total_wdw = float(schedule["withdraw"].sum())

                r1, r2, r3 = st.columns(3)
                r1.metric("Intrinsic Value ($)",     f"${value:,.0f}")
                r2.metric("Total Injected (MMBtu)",  f"{total_inj:,.0f}")
                r3.metric("Total Withdrawn (MMBtu)", f"{total_wdw:,.0f}")

                dates = schedule["delivery_date"].astype(str)
                fig_sched = go.Figure()

                # plot injections (positive bars)
                fig_sched.add_trace(go.Bar(x=dates, y=schedule["inject"],
                                           name="Inject", marker_color="#43a047"))

                # plot withdrawals as negative bars (visual symmetry)
                fig_sched.add_trace(go.Bar(x=dates, y=-schedule["withdraw"],
                                           name="Withdraw", marker_color="#e53935"))

                # plot inventory level as a line (on secondary axis)
                fig_sched.add_trace(go.Scatter(x=dates, y=schedule["inventory"],
                                               name="Inventory", yaxis="y2",
                                               line=dict(color="#ffd54f", width=2)))

                fig_sched.update_layout(
                    **PLOT_BASE, barmode="relative",
                    xaxis_title="Delivery Month", yaxis_title="Volume (MMBtu)",
                    yaxis2=dict(title="Inventory", overlaying="y", side="right",
                                gridcolor="rgba(0,0,0,0)"),
                    title="Optimal Injection / Withdrawal Schedule",
                )
                st.plotly_chart(fig_sched, use_container_width=True)
