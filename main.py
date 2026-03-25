import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State, ALL
from dash.exceptions import PreventUpdate

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
    margin=dict(l=50, r=20, t=20, b=40),
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
# Layout helpers
# ---------------------------------------------------------------------------

def make_input(label, id_, **kwargs):
    return dbc.Row([
        dbc.Col(html.Label(label, className="small text-muted mb-1"), width=12),
        dbc.Col(dbc.Input(id=id_, type="number", size="sm", **kwargs), width=12),
    ], className="mb-2")

def make_stat_card(label, id_, color="white"):
    return dbc.Card(dbc.CardBody([
        html.P(label, className="small text-muted mb-1"),
        html.H5(id=id_, children="—", style={"color": color}),
    ]), className="h-100")

def empty_fig():
    fig = go.Figure()
    fig.update_layout(**PLOT_BASE)
    return fig

def payoff_and_greeks_right_panel(payoff_id, greeks_id):
    return [
        dbc.Card(dbc.CardBody([
            html.H6("Payoff Diagram"),
            dcc.Graph(id=payoff_id, config={"displayModeBar": False},
                      style={"height": "300px"}, figure=empty_fig()),
        ]), className="mb-3"),
        dbc.Card(dbc.CardBody([
            html.H6("Greeks Curves"),
            dcc.Graph(id=greeks_id, config={"displayModeBar": False},
                      style={"height": "320px"}, figure=empty_fig()),
        ])),
    ]

# ---------------------------------------------------------------------------
# Options Pricing tab
# ---------------------------------------------------------------------------

def simple_pricer_tab():
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                dbc.Card(dbc.CardBody([
                    html.H6("Parameters", className="text-info mb-3"),
                    dbc.Row([
                        dbc.Col(make_input("Spot (S)",    "sp-S",     value=100,  min=0.01, step=0.5), md=6),
                        dbc.Col(make_input("Strike (K)",  "sp-K",     value=100,  min=0.01, step=0.5), md=6),
                    ]),
                    dbc.Row([
                        dbc.Col(make_input("Rate r (%)",  "sp-r",     value=5.0,  min=0, step=0.1), md=6),
                        dbc.Col(make_input("Div q (%)",   "sp-q",     value=0.0,  min=0, step=0.1), md=6),
                    ]),
                    dbc.Row([
                        dbc.Col(make_input("Vol σ (%)",     "sp-sigma", value=20.0, min=0.1, step=0.5), md=6),
                        dbc.Col(make_input("Expiry (days)", "sp-T",     value=30,   min=1,   step=1),   md=6),
                    ]),
                    html.Label("Type", className="small text-muted"),
                    dbc.RadioItems(
                        id="sp-type",
                        options=[{"label": " Call", "value": "call"},
                                 {"label": " Put",  "value": "put"}],
                        value="call", inline=True, className="mb-3",
                    ),
                    dbc.Button("Price", id="sp-btn-price", color="primary", size="sm", className="w-100"),
                    html.Hr(className="my-3"),
                    html.H6("Greeks Curves", className="text-info mb-2"),
                    html.Label("X-axis", className="small text-muted"),
                    dbc.Select(
                        id="sp-xaxis",
                        options=[
                            {"label": "Spot Price",     "value": "S"},
                            {"label": "Volatility σ",   "value": "sigma"},
                            {"label": "Time to Expiry", "value": "T"},
                        ],
                        value="S", size="sm", className="mb-3",
                    ),
                    html.Label("Greeks", className="small text-muted"),
                    dbc.Checklist(
                        id="sp-greeks-sel",
                        options=[{"label": f" {g}", "value": g}
                                 for g in ["Delta", "Gamma", "Vega", "Theta", "Rho"]],
                        value=["Delta", "Gamma", "Vega"],
                        labelStyle={"display": "block"},
                    ),
                ]))
            ], md=4),

            dbc.Col([
                dbc.Row([
                    dbc.Col(make_stat_card("Price",  "sp-out-price", "#2196f3"), md=True),
                    dbc.Col(make_stat_card("Delta",  "sp-out-delta"), md=True),
                    dbc.Col(make_stat_card("Gamma",  "sp-out-gamma"), md=True),
                    dbc.Col(make_stat_card("Vega",   "sp-out-vega"),  md=True),
                    dbc.Col(make_stat_card("Theta",  "sp-out-theta"), md=True),
                    dbc.Col(make_stat_card("Rho",    "sp-out-rho"),   md=True),
                ], className="mb-3"),
                *payoff_and_greeks_right_panel("sp-chart-payoff", "sp-chart-greeks"),
            ], md=8),
        ], className="g-3 mt-1"),
    ], fluid=True)

# ---------------------------------------------------------------------------
# Strategy Options Pricer
# ---------------------------------------------------------------------------

def strategy_tab():
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                dbc.Card(dbc.CardBody([
                    html.H6("Market Parameters", className="text-info mb-3"),
                    dbc.Row([
                        dbc.Col(make_input("Spot (S)",    "st-S",     value=100,  min=0.01, step=0.5), md=6),
                        dbc.Col(make_input("Rate r (%)",  "st-r",     value=5.0,  min=0, step=0.1),   md=6),
                    ]),
                    dbc.Row([
                        dbc.Col(make_input("Div q (%)",   "st-q",     value=0.0,  min=0, step=0.1),   md=6),
                        dbc.Col(make_input("Vol σ (%)",   "st-sigma", value=20.0, min=0.1, step=0.5), md=6),
                    ]),
                    make_input("Expiry (days)", "st-T", value=30, min=1, step=1),
                    html.Hr(className="my-2"),

                    html.H6("Strategy", className="text-info mb-2"),
                    dbc.Select(
                        id="st-preset",
                        options=[{"label": "— Select —", "value": "none"}] + [
                            {"label": label, "value": key}
                            for key, label in STRATEGY_LABELS.items()
                        ],
                        value="none", size="sm", className="mb-3",
                    ),

                    html.Div(id="st-strikes",
                             children=html.P("Select a strategy above.", className="text-muted small")),

                    dbc.Button("Price Strategy", id="st-btn-price", color="primary",
                               size="sm", className="w-100 mt-3"),
                    html.Hr(className="my-3"),
                    html.H6("Greeks Curves", className="text-info mb-2"),
                    html.Label("X-axis", className="small text-muted"),
                    dbc.Select(
                        id="st-xaxis",
                        options=[
                            {"label": "Spot Price",     "value": "S"},
                            {"label": "Volatility σ",   "value": "sigma"},
                            {"label": "Time to Expiry", "value": "T"},
                        ],
                        value="S", size="sm", className="mb-3",
                    ),
                    html.Label("Greeks", className="small text-muted"),
                    dbc.Checklist(
                        id="st-greeks-sel",
                        options=[{"label": f" {g}", "value": g}
                                 for g in ["Delta", "Gamma", "Vega", "Theta", "Rho"]],
                        value=["Delta", "Gamma", "Vega"],
                        labelStyle={"display": "block"},
                    ),
                ]))
            ], md=4),

            dbc.Col([
                dbc.Row([
                    dbc.Col(make_stat_card("Net Cost", "st-out-cost",  "#2196f3"), md=True),
                    dbc.Col(make_stat_card("Delta",    "st-out-delta"), md=True),
                    dbc.Col(make_stat_card("Gamma",    "st-out-gamma"), md=True),
                    dbc.Col(make_stat_card("Vega",     "st-out-vega"),  md=True),
                    dbc.Col(make_stat_card("Theta",    "st-out-theta"), md=True),
                    dbc.Col(make_stat_card("Rho",      "st-out-rho"),   md=True),
                ], className="mb-3"),
                *payoff_and_greeks_right_panel("st-chart-payoff", "st-chart-greeks"),
            ], md=8),
        ], className="g-3 mt-1"),
    ], fluid=True)

# ---------------------------------------------------------------------------
# Gas Storage Pricing
# ---------------------------------------------------------------------------

def gas_storage_tab():
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                dbc.Card(dbc.CardBody([
                    html.H6("Storage Parameters", className="text-info mb-3"),
                    dbc.Row([
                        dbc.Col(make_input("Capacity (MMBtu)",        "gs-capacity", value=1_000_000, min=1, step=10_000), md=6),
                        dbc.Col(make_input("Initial Inventory",        "gs-init-inv", value=0,         min=0, step=10_000), md=6),
                    ]),
                    dbc.Row([
                        dbc.Col(make_input("Max Injection/mo",         "gs-max-inj",  value=300_000,   min=1, step=10_000), md=6),
                        dbc.Col(make_input("Max Withdrawal/mo",        "gs-max-wdw",  value=500_000,   min=1, step=10_000), md=6),
                    ]),
                    dbc.Row([
                        dbc.Col(make_input("Injection Cost ($/MMBtu)", "gs-inj-cost", value=0.01, min=0, step=0.001), md=6),
                        dbc.Col(make_input("Withdrawal Cost ($/MMBtu)","gs-wdw-cost", value=0.01, min=0, step=0.001), md=6),
                    ]),
                    html.Label("Months to Fetch", className="small text-muted mb-1"),
                    dbc.RadioItems(
                        id="gs-n-months",
                        options=[{"label": " 12 months", "value": 12},
                                 {"label": " 24 months", "value": 24}],
                        value=12, inline=True, className="mb-2",
                    ),
                    dbc.Button("Fetch Forward Curve", id="gs-btn-fetch",
                               color="secondary", size="sm", className="w-100 mb-2"),
                    html.Div(id="gs-fetch-status", className="small text-muted mb-2"),
                    dbc.Button("Calculate Intrinsic Value", id="gs-btn-calc",
                               color="primary", size="sm", className="w-100", disabled=True),
                ])),
                dbc.Card(dbc.CardBody([
                    html.H6("How it works", className="text-info mb-2"),
                    html.P(
                        "This model computes the intrinsic value of a gas storage asset by solving "
                        "an optimization problem over the current Henry Hub forward price curve. "
                        "Data are fetched live from Yahoo Finance (NYMEX).",
                        className="small text-muted",
                    ),
                    html.P(
                        "The model determines the optimal monthly injection and withdrawal volumes "
                        "that maximize total profit.",
                        className="small text-muted",
                    ),
                    html.P("Objective", className="small text-white mb-1"),
                    html.P(
                        "Maximize: withdrawal revenue − injection cost − operational costs. "
                        "Internally expressed as a cost minimization (linear program).",
                        className="small text-muted",
                    ),
                    html.P("Constraints", className="small text-white mb-1"),
                    html.Ul([
                        html.Li("Inventory cannot exceed maximum capacity.", className="small text-muted"),
                        html.Li("Storage level can never go below zero.", className="small text-muted"),
                        html.Li("Injection and withdrawal volumes are capped each month.", className="small text-muted"),
                        html.Li("Inventory evolves from cumulative injections and withdrawals.", className="small text-muted"),
                        html.Li("Final inventory must equal initial inventory.", className="small text-muted"),
                    ]),
                ]), className="mt-3"),
            ], md=4),

            dbc.Col([
                dbc.Row([
                    dbc.Col(make_stat_card("Intrinsic Value ($)",      "gs-out-value",     "#2196f3"), md=4),
                    dbc.Col(make_stat_card("Total Injected (MMBtu)",   "gs-out-injected"),             md=4),
                    dbc.Col(make_stat_card("Total Withdrawn (MMBtu)",  "gs-out-withdrawn"),            md=4),
                ], className="mb-3"),
                dbc.Card(dbc.CardBody([
                    html.H6("Henry Hub Forward Curve"),
                    dcc.Graph(id="gs-chart-curve", config={"displayModeBar": False},
                              style={"height": "280px"}, figure=empty_fig()),
                ]), className="mb-3"),
                dbc.Card(dbc.CardBody([
                    html.H6("Optimal Injection / Withdrawal Schedule"),
                    dcc.Graph(id="gs-chart-schedule", config={"displayModeBar": False},
                              style={"height": "280px"}, figure=empty_fig()),
                ])),
            ], md=8),
        ], className="g-3 mt-1"),
    ], fluid=True)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG],
                suppress_callback_exceptions=True, title="Options Pricer")

app.layout = html.Div([
    dbc.NavbarSimple(
        brand="Options Pricer",
        brand_style={"fontSize": "1.1rem"},
        color="dark", dark=True,
    ),
    dbc.Tabs(
        id="main-tabs", active_tab="tab-pricer",
        children=[
            dbc.Tab(label="Options Pricing", tab_id="tab-pricer"),
            dbc.Tab(label="Strategy",        tab_id="tab-strategy"),
            dbc.Tab(label="Gas Storage",     tab_id="tab-storage"),
        ],
    ),
    html.Div(simple_pricer_tab(), id="content-pricer"),
    html.Div(strategy_tab(),      id="content-strategy", style={"display": "none"}),
    html.Div(gas_storage_tab(),   id="content-storage",  style={"display": "none"}),
    dcc.Store(id="gs-curve-store"),
])

# ---------------------------------------------------------------------------
# Tab switching
# ---------------------------------------------------------------------------

@app.callback(
    Output("content-pricer",   "style"),
    Output("content-strategy", "style"),
    Output("content-storage",  "style"),
    Input("main-tabs", "active_tab"),
)
def switch_tab(tab):
    show, hide = {"display": "block"}, {"display": "none"}
    return (
        show if tab == "tab-pricer"   else hide,
        show if tab == "tab-strategy" else hide,
        show if tab == "tab-storage"  else hide,
    )

# ---------------------------------------------------------------------------
# Options Pricing callbacks
# ---------------------------------------------------------------------------

@app.callback(
    Output("sp-out-price", "children"),
    Output("sp-out-delta", "children"),
    Output("sp-out-gamma", "children"),
    Output("sp-out-vega",  "children"),
    Output("sp-out-theta", "children"),
    Output("sp-out-rho",   "children"),
    Output("sp-chart-payoff", "figure"),
    Input("sp-btn-price", "n_clicks"),
    State("sp-S",     "value"),
    State("sp-K",     "value"),
    State("sp-r",     "value"),
    State("sp-q",     "value"),
    State("sp-sigma", "value"),
    State("sp-T",     "value"),
    State("sp-type",  "value"),
    prevent_initial_call=True,
)

# Pricing function
def price_simple(_, S, K, r_pct, q_pct, sigma_pct, T_days, otype):

    # convert all user inputs into clean numerical values
    S     = safe_float(S,100.0)
    K     = safe_float(K,100.0)
    r     = safe_float(r_pct,5.0) / 100
    q     = safe_float(q_pct,0.0) / 100
    sigma = safe_float(sigma_pct,20.0) / 100
    T     = safe_float(T_days,30.0) / 365

    # compute Black-Scholes price at current spot
    price = bs.BS_pricing(S, K, r, sigma, T, otype, q)

    # compute all Greeks at current spot
    greeks = bs.all_greeks(S, K, r, sigma, T, otype, q)

    # build a range of spot values around current S used to visualize payoff and price sensitivity
    s_range = np.linspace(max(0.01, S * 0.5), S * 1.5, 300)

    # compute payoff at expiry for each spot in the range
    expiry_payoff = (
        np.maximum(s_range - K, 0)
        if otype == "call"
        else np.maximum(K - s_range, 0)
    )

    # compute current option value
    current_vals = [
        safe_float(bs.BS_pricing(s, K, r, sigma, T, otype, q), 0.0)
        for s in s_range
    ]

    # create fig for visualization
    fig = go.Figure()

    # plot payoff at expiry
    fig.add_trace(go.Scatter(
        x=s_range,
        y=expiry_payoff,
        name="At Expiry",
        line=dict(dash="dot", color="#e0e0e0")
    ))

    # plot current theoretical value
    fig.add_trace(go.Scatter(
        x=s_range,
        y=current_vals,
        name="Current Value",
        line=dict(color="#2196f3")
    ))

    # add vertical line at strike
    fig.add_vline(x=K, line=dict(dash="dot"), annotation_text=f"K={K}")

    # add vertical line at current spot
    fig.add_vline(x=S, line=dict(dash="dot", color="gray"), annotation_text=f"S={S}")

    # apply common styling + axis labels
    fig.update_layout(**PLOT_BASE, xaxis_title="Spot", yaxis_title="P&L")

    # return formatted results for display + the chart
    return (
        format_num(price),
        format_num(greeks["Delta"]), format_num(greeks["Gamma"]),
        format_num(greeks["Vega"]),  format_num(greeks["Theta"]),
        format_num(greeks["Rho"]),
        fig,
    )


@app.callback(
    Output("sp-chart-greeks", "figure"),
    Input("sp-btn-price",  "n_clicks"),
    Input("sp-xaxis",      "value"),
    Input("sp-greeks-sel", "value"),
    State("sp-S",     "value"),
    State("sp-K",     "value"),
    State("sp-r",     "value"),
    State("sp-q",     "value"),
    State("sp-sigma", "value"),
    State("sp-T",     "value"),
    State("sp-type",  "value"),
    prevent_initial_call=True,
)

def draw_simple_greeks(_, xaxis, selected, S, K, r_pct, q_pct, sigma_pct, T_days, otype):

    # clean and convert all inputs
    S     = safe_float(S,        100.0)
    K     = safe_float(K,        100.0)
    r     = safe_float(r_pct,     5.0) / 100
    q     = safe_float(q_pct,     0.0) / 100
    sigma = safe_float(sigma_pct, 20.0) / 100
    T     = safe_float(T_days,   30.0) / 365

    # default selection if nothing is chosen in UI
    selected = selected or ["Delta"]

    # build the x-axis depending on what the user wants to vary
    if xaxis == "S":

        # vary spot around current value
        x_vals, x_label = np.linspace(max(0.01, S * 0.5), S * 1.5, 200), "Spot Price"
    elif xaxis == "sigma":

        # vary volatility from low to high
        x_vals, x_label = np.linspace(0.01, 1.0, 200), "Volatility"
    else:

        # vary time to expiry (in years)
        x_vals, x_label = np.linspace(1/365, 2.0, 200), "Time to Expiry (years)"

    def compute(greek, x):
        # decide which variable is moving and which stay fixed
        sv   = x if xaxis == "S"     else S
        sigv = x if xaxis == "sigma" else sigma
        Tv   = x if xaxis == "T"     else T

        # avoid invalid BS inputs
        if sv <= 0 or sigv <= 0 or Tv <= 0:
            return np.nan

        # compute the selected greek at this point
        if greek == "Delta": return bs.delta(sv, K, r, sigv, Tv, otype, q)
        if greek == "Gamma": return bs.gamma(sv, K, r, sigv, Tv, q)
        if greek == "Vega":  return bs.vega(sv, K, r, sigv, Tv, q)
        if greek == "Theta": return bs.theta(sv, K, r, sigv, Tv, otype, q)
        if greek == "Rho":   return bs.rho(sv, K, r, sigv, Tv, otype, q)

    # number of subplots = number of selected Greeks
    n = len(selected)

    # create stacked plots sharing the same x-axis
    fig = make_subplots(rows=n, cols=1, shared_xaxes=True, vertical_spacing=0.04)

    # loop over each selected greek
    for i, greek in enumerate(selected, 1):

        # compute the curve by evaluating the greek for each x value
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=[compute(greek, x) for x in x_vals],
            name=greek,
            line=dict(color="#2196f3")
        ), row=i, col=1)

        # label each subplot with the greek name
        fig.update_yaxes(title_text=greek, row=i, col=1, gridcolor="#2a2a4a")

    # label x-axis only on the last subplot
    fig.update_xaxes(title_text=x_label, row=n, col=1, gridcolor="#2a2a4a")

    # apply style
    fig.update_layout(
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#16213e",
        font=dict(color="#e0e0e0"),
        height=320,
        margin=dict(l=60, r=20, t=10, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0)")
    )

    return fig

# ---------------------------------------------------------------------------
# Strategy callbacks options strategy
# ---------------------------------------------------------------------------

@app.callback(
    Output("st-strikes", "children"),
    Input("st-preset", "value"),
    State("st-S", "value"),
)
def render_strike_bubbles(strategy, S):
    if not strategy or strategy == "none" or strategy not in STRATEGY_STRIKE_CONFIG:
        return html.P("Select a strategy above.", className="text-muted small")

    S = safe_float(S, 100.0)
    strike_defs = STRATEGY_STRIKE_CONFIG[strategy]

    bubbles = []
    for i, (label, k_pct) in enumerate(strike_defs):
        bubbles.append(dbc.Col([
            dbc.Card(dbc.CardBody([
                html.P(label, className="small text-muted mb-1 text-center"),
                dbc.Input(
                    id={"type": "st-strike", "index": i},
                    type="number",
                    value=round(S * k_pct, 1),
                    min=0.01, step=0.5, size="sm",
                    style={"textAlign": "center"},
                ),
            ], className="p-2"))
        ], xs=6, md=6, className="mb-2"))

    return dbc.Row(bubbles)


@app.callback(
    Output("st-out-cost",  "children"),
    Output("st-out-delta", "children"),
    Output("st-out-gamma", "children"),
    Output("st-out-vega",  "children"),
    Output("st-out-theta", "children"),
    Output("st-out-rho",   "children"),
    Output("st-chart-payoff", "figure"),
    Input("st-btn-price", "n_clicks"),
    State("st-preset",  "value"),
    State({"type": "st-strike", "index": ALL}, "value"),
    State("st-S",     "value"),
    State("st-r",     "value"),
    State("st-q",     "value"),
    State("st-sigma", "value"),
    State("st-T",     "value"),
    prevent_initial_call=True,
)

def price_strategy(_, strategy, strikes, S, r_pct, q_pct, sigma_pct, T_days):

    # default output if inputs are missing
    blank = "—"

    # if no strategy or strikes : nothing to price
    if not strategy or strategy == "none" or not strikes:
        return blank, blank, blank, blank, blank, blank, empty_fig()

    # same logic
    S     = safe_float(S, 100.0)
    r     = safe_float(r_pct, 5.0) / 100
    q     = safe_float(q_pct, 0.0) / 100
    sigma = safe_float(sigma_pct, 20.0)  
    T     = safe_float(T_days,   30.0)    

    # build list of option legs composing the strategy
    legs = build_legs(strategy, strikes, sigma, T)

    # if something went wrong in leg construction : stop
    if not legs:
        return blank, blank, blank, blank, blank, blank, empty_fig()

    # initialize aggregated metrics
    net_cost = net_delta = net_gamma = net_vega = net_theta = net_rho = 0.0

    # loop over each leg and sum contributions
    for leg in legs:

        # convert leg parameters to BS inputs
        sig = max(leg["sigma_pct"] / 100, 1e-6)
        Tv  = max(leg["T_days"] / 365, 1e-6)

        qty = leg["qty"]     
        ot  = leg["type"]    
        K   = leg["K"]      

        # accumulate price and Greeks weighted by position size
        net_cost  += qty * safe_float(bs.BS_pricing(S, K, r, sig, Tv, ot, q), 0.0)
        net_delta += qty * safe_float(bs.delta(S, K, r, sig, Tv, ot, q), 0.0)
        net_gamma += qty * safe_float(bs.gamma(S, K, r, sig, Tv, q), 0.0)
        net_vega  += qty * safe_float(bs.vega(S, K, r, sig, Tv, q), 0.0)
        net_theta += qty * safe_float(bs.theta(S, K, r, sig, Tv, ot, q), 0.0)
        net_rho   += qty * safe_float(bs.rho(S, K, r, sig, Tv, ot, q), 0.0)

    # build spot range to visualize payoff and current value
    s_range = np.linspace(max(0.01, S * 0.5), S * 1.5, 400)

    # initialize total P&L curves
    expiry_pnl = np.zeros(len(s_range))
    current_pnl = np.zeros(len(s_range))

    # color palette for individual legs
    colors = ["#2196f3", "#42a5f5", "#90caf9", "#bbdefb"]

    fig = go.Figure()

    # loop again over each leg to build payoff curves
    for i, leg in enumerate(legs):

        sig = max(leg["sigma_pct"] / 100, 1e-6)
        Tv  = max(leg["T_days"] / 365, 1e-6)
        qty = leg["qty"]
        ot  = leg["type"]
        K   = leg["K"]

        # current price of this leg at spot S
        p0  = safe_float(bs.BS_pricing(S, K, r, sig, Tv, ot, q), 0.0)

        # payoff at expiry minus initial premium paid
        leg_exp = qty * (
            np.maximum(s_range - K, 0) if ot == "call"
            else np.maximum(K - s_range, 0)
        ) - qty * p0

        # add this leg contribution to total expiry P&L
        expiry_pnl += leg_exp

        # compute current value across spot range
        curr = np.array([
            safe_float(bs.BS_pricing(s, K, r, sig, Tv, ot, q), 0.0)
            for s in s_range
        ])

        # current P&L = value today - initial price
        current_pnl += qty * (curr - p0)

        # plot individual leg payoff (faint dotted line)
        fig.add_trace(go.Scatter(
            x=s_range,
            y=leg_exp,
            name=f"{qty:+g} {ot} K={K}",
            line=dict(dash="dot", color=colors[i % len(colors)], width=1),
            opacity=0.5
        ))

    # plot total strategy payoff at expiry
    fig.add_trace(go.Scatter(
        x=s_range,
        y=expiry_pnl,
        name="At Expiry",
        line=dict(color="#e0e0e0", width=2)
    ))

    # plot total current P&L
    fig.add_trace(go.Scatter(
        x=s_range,
        y=current_pnl,
        name="Current Value",
        line=dict(color="#2196f3", width=2)
    ))

    # horizontal zero line → break-even reference
    fig.add_hline(y=0, line=dict(color="gray", dash="dot", width=1))

    # vertical line at current spot
    fig.add_vline(x=S, line=dict(color="gray", dash="dot"), annotation_text=f"S={S}")

    # vertical lines at each strike
    for leg in legs:
        fig.add_vline(x=leg["K"], line=dict(dash="dot", color="#555", width=1))

    # adjust y-axis range with padding for readability
    combined = np.concatenate([expiry_pnl, current_pnl])
    pad = max(abs(np.max(combined) - np.min(combined)) * 0.15, 0.5)

    fig.update_layout({**PLOT_BASE, "xaxis_title": "Spot at Expiry", "yaxis_title": "P&L"})
    fig.update_yaxes(range=[float(np.min(combined)) - pad, float(np.max(combined)) + pad])

    # return aggregated metrics + chart
    return (
        format_num(net_cost),
        format_num(net_delta), format_num(net_gamma),
        format_num(net_vega),  format_num(net_theta),
        format_num(net_rho),
        fig,
    )


@app.callback(
    Output("st-chart-greeks", "figure"),
    Input("st-btn-price",   "n_clicks"),
    Input("st-xaxis",       "value"),
    Input("st-greeks-sel",  "value"),
    State("st-preset",  "value"),
    State({"type": "st-strike", "index": ALL}, "value"),
    State("st-S",     "value"),
    State("st-r",     "value"),
    State("st-q",     "value"),
    State("st-sigma", "value"),
    State("st-T",     "value"),
    prevent_initial_call=True,
)

def draw_strategy_greeks(_, xaxis, selected, strategy, strikes, S, r_pct, q_pct, sigma_pct, T_days):
    if not strategy or strategy == "none" or not strikes:
        return empty_fig()

    S     = safe_float(S, 100.0)
    r     = safe_float(r_pct, 5.0) / 100
    q     = safe_float(q_pct, 0.0) / 100
    sigma = safe_float(sigma_pct, 20.0)
    T     = safe_float(T_days, 30.0)
    selected = selected or ["Delta"]

    legs = build_legs(strategy, strikes, sigma, T)
    if not legs:
        return empty_fig()

    if xaxis == "S":
        x_vals, x_label = np.linspace(max(0.01, S * 0.5), S * 1.5, 200), "Spot Price"
    elif xaxis == "sigma":
        x_vals, x_label = np.linspace(0.01, 1.0, 200), "Volatility"
    else:
        x_vals, x_label = np.linspace(1/365, 2.0, 200), "Time to Expiry (years)"

    def net_greek(greek, x):
        total = 0.0
        for leg in legs:
            sig = max(leg["sigma_pct"] / 100, 1e-6)
            Tv  = max(leg["T_days"] / 365, 1e-6)
            qty = leg["qty"]
            ot  = leg["type"]
            K   = leg["K"]
            sv   = x if xaxis == "S"     else S
            sigv = x if xaxis == "sigma" else sig
            Tvv  = x if xaxis == "T"     else Tv
            if sv <= 0 or sigv <= 0 or Tvv <= 0:
                return np.nan
            if greek == "Delta": total += qty * bs.delta(sv, K, r, sigv, Tvv, ot, q)
            if greek == "Gamma": total += qty * bs.gamma(sv, K, r, sigv, Tvv, q)
            if greek == "Vega":  total += qty * bs.vega(sv, K, r, sigv, Tvv, q)
            if greek == "Theta": total += qty * bs.theta(sv, K, r, sigv, Tvv, ot, q)
            if greek == "Rho":   total += qty * bs.rho(sv, K, r, sigv, Tvv, ot, q)
        return total

    n = len(selected)
    fig = make_subplots(rows=n, cols=1, shared_xaxes=True, vertical_spacing=0.04)
    for i, greek in enumerate(selected, 1):
        fig.add_trace(go.Scatter(x=x_vals, y=[net_greek(greek, x) for x in x_vals],
                                 name=greek, line=dict(color="#2196f3")), row=i, col=1)
        fig.update_yaxes(title_text=greek, row=i, col=1, gridcolor="#2a2a4a")
    fig.update_xaxes(title_text=x_label, row=n, col=1, gridcolor="#2a2a4a")
    fig.update_layout(paper_bgcolor="#1a1a2e", plot_bgcolor="#16213e",
                      font=dict(color="#e0e0e0"), height=320,
                      margin=dict(l=60, r=20, t=10, b=40),
                      legend=dict(bgcolor="rgba(0,0,0,0)"))
    return fig

# ---------------------------------------------------------------------------
# Gas Storage callbacks
# ---------------------------------------------------------------------------

@app.callback(
    Output("gs-curve-store",  "data"),
    Output("gs-chart-curve",  "figure"),
    Output("gs-fetch-status", "children"),
    Output("gs-btn-calc",     "disabled"),
    Input("gs-btn-fetch", "n_clicks"),
    State("gs-n-months",  "value"),
    prevent_initial_call=True,
)
def fetch_curve(_, n_months):
    from gas_storage_pricing import fetch_henry_hub_curve
    n = int(n_months) if n_months in (12, 24) else 12
    try:
        df = fetch_henry_hub_curve(n_months=n)
    except Exception as e:
        return None, empty_fig(), f"Error: {e}", True
    if df.empty:
        return None, empty_fig(), "No data returned — markets may be closed.", True

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["delivery_date"].astype(str), y=df["price"],
        mode="lines+markers", line=dict(color="#2196f3"), marker=dict(size=6),
        name="HH Futures",
    ))
    fig.update_layout(**PLOT_BASE, xaxis_title="Delivery Month", yaxis_title="Price ($/MMBtu)")
    status = f"Fetched {len(df)} contracts — last: {df['ticker'].iloc[-1]}"
    return df.to_json(date_format="iso", orient="split"), fig, status, False


@app.callback(
    Output("gs-out-value",     "children"),
    Output("gs-out-injected",  "children"),
    Output("gs-out-withdrawn", "children"),
    Output("gs-chart-schedule","figure"),
    Input("gs-btn-calc", "n_clicks"),
    State("gs-curve-store", "data"),
    State("gs-capacity",    "value"),
    State("gs-init-inv",    "value"),
    State("gs-max-inj",     "value"),
    State("gs-max-wdw",     "value"),
    State("gs-inj-cost",    "value"),
    State("gs-wdw-cost",    "value"),
    prevent_initial_call=True,
)

def calculate_intrinsic(_, curve_json, capacity, init_inv, max_inj, max_wdw, inj_cost, wdw_cost):

    from gas_storage_pricing import GasStorageIntrinsicValue

    # if no curve has been fetched → nothing to compute
    if not curve_json:
        return "—", "—", "—", empty_fig()

    import io

    # rebuild df from stored JSON
    df = pd.read_json(io.StringIO(curve_json), orient="split")

    # ensure dates are in proper format
    df["delivery_date"] = pd.to_datetime(df["delivery_date"])

    # create storage object with user inputs (cleaned)
    storage = GasStorageIntrinsicValue(
        capacity=safe_float(capacity, 1_000_000),
        initial_inventory=safe_float(init_inv, 0.0),
        max_injection_rate=safe_float(max_inj, 300_000),
        max_withdrawal_rate=safe_float(max_wdw, 500_000),
        injection_cost=safe_float(inj_cost, 0.01),
        withdrawal_cost=safe_float(wdw_cost, 0.01),
    )

    # run optimization : returns intrinsic value + optimal schedule
    value, schedule = storage.storage_price(df)

    # if LP failed 
    if schedule.empty:
        return "Infeasible", "—", "—", empty_fig()

    # extract dates for plotting
    dates = schedule["delivery_date"].astype(str)

    fig = go.Figure()

    # plot injections (positive bars)
    fig.add_trace(go.Bar(
        x=dates,
        y=schedule["inject"],
        name="Inject",
        marker_color="#43a047"
    ))

    # plot withdrawals as negative bars (visual symmetry)
    fig.add_trace(go.Bar(
        x=dates,
        y=-schedule["withdraw"],
        name="Withdraw",
        marker_color="#e53935"
    ))

    # plot inventory level as a line (on secondary axis)
    fig.add_trace(go.Scatter(
        x=dates,
        y=schedule["inventory"],
        name="Inventory",
        yaxis="y2",
        line=dict(color="#ffd54f", width=2)
    ))

    # layout:
    fig.update_layout(
        **PLOT_BASE,
        barmode="relative",
        xaxis_title="Delivery Month",
        yaxis_title="Volume (MMBtu)",
        yaxis2=dict(
            title="Inventory",
            overlaying="y",
            side="right",
            gridcolor="rgba(0,0,0,0)"
        ),
    )

    # compute total injected and withdrawn volumes
    total_inj = float(schedule["inject"].sum())
    total_wdw = float(schedule["withdraw"].sum())

    # return formatted results + chart
    return (
        f"${value:,.0f}",
        f"{total_inj:,.0f}",
        f"{total_wdw:,.0f}",
        fig
    )

if __name__ == "__main__":
    app.run(debug=True, port=8050)
