"""
Airline Operations Analytics Dashboard
---------------------------------------
Flask + Dash + Plotly + Pandas frontend for the cleaned airline dataset
produced by the data-engineering notebook (airline_pipeline.ipynb).

This file is responsible ONLY for:
    - loading the already-cleaned dataset
    - calculating aggregations/KPIs
    - rendering an interactive dashboard

All cleaning, standardization, duplicate handling, duration calculation and
overnight-flag logic was already performed in the notebook. This app does
NOT redo or alter that logic.

Run with:
    python app.py

Then open:
    http://127.0.0.1:5000/dashboard/
"""

import os

import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
from flask import Flask

# --------------------------------------------------------------------------
# PATHS
# --------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "cleaned_airlines.csv")

# --------------------------------------------------------------------------
# COLOR / STYLE CONSTANTS
# --------------------------------------------------------------------------
COLORS = {
    "bg": "#F4F6F9",
    "card_bg": "#FFFFFF",
    "text": "#1F2A44",
    "muted": "#6B7280",
    "primary": "#2563EB",
    "primary_dark": "#1D4ED8",
    "accent": "#0EA5E9",
    "success": "#16A34A",
    "warning": "#D97706",
    "danger": "#DC2626",
    "border": "#E5E7EB",
}

CHART_COLORWAY = ["#2563EB", "#0EA5E9", "#16A34A", "#D97706", "#DC2626", "#7C3AED"]

CARD_STYLE = {
    "backgroundColor": COLORS["card_bg"],
    "borderRadius": "12px",
    "border": f"1px solid {COLORS['border']}",
    "boxShadow": "0 1px 3px rgba(16, 24, 40, 0.06)",
    "padding": "20px",
}


# --------------------------------------------------------------------------
# DATA LOADING
# --------------------------------------------------------------------------
REQUIRED_COLUMNS = [
    "flight_id",
    "airline",
    "source",
    "destination",
    "departure_time",
    "arrival_time",
    "route",
    "airline_standardized",
    "departure_date",
    "is_overnight",
    "flight_duration_hours",
]


def load_data():
    """
    Load the cleaned dataset produced by the notebook pipeline.

    Returns a tuple: (dataframe, error_message).
    If loading fails, dataframe is an empty DataFrame and error_message
    explains what went wrong.
    """
    if not os.path.exists(DATA_PATH):
        return pd.DataFrame(), (
            f"cleaned_airlines.csv was not found at: {DATA_PATH}. "
            "Make sure app.py sits in the same folder as cleaned_airlines.csv."
        )

    try:
        df = pd.read_csv(DATA_PATH)
    except Exception as exc:  # noqa: BLE001
        return pd.DataFrame(), f"Failed to read cleaned_airlines.csv: {exc}"

    if df.empty:
        return df, "cleaned_airlines.csv was loaded but contains no rows."

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        return df, (
            "cleaned_airlines.csv is missing expected columns: "
            f"{', '.join(missing_cols)}"
        )

    # Parse timestamps for any date-based display. These columns are already
    # clean coming out of the notebook; this only converts dtype for display,
    # it does not re-derive or correct any values.
    for col in ["departure_time", "arrival_time"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    df["departure_date"] = pd.to_datetime(df["departure_date"], errors="coerce").dt.date

    # is_overnight may load as bool or as string "True"/"False" depending on
    # how it was written to CSV. Normalize to real booleans without changing
    # the underlying meaning.
    if df["is_overnight"].dtype != bool:
        df["is_overnight"] = df["is_overnight"].astype(str).str.strip().str.lower() == "true"

    return df, None


df, LOAD_ERROR = load_data()

if LOAD_ERROR is None:
    AIRLINE_OPTIONS = sorted(df["airline_standardized"].dropna().unique().tolist())
    ROUTE_OPTIONS = sorted(df["route"].dropna().unique().tolist())
else:
    AIRLINE_OPTIONS = []
    ROUTE_OPTIONS = []


# --------------------------------------------------------------------------
# AGGREGATION HELPERS
# --------------------------------------------------------------------------
def filter_data(airline, route):
    """Apply the selected Airline / Route filters to the base dataframe."""
    filtered = df.copy()
    if airline and airline != "ALL":
        filtered = filtered[filtered["airline_standardized"] == airline]
    if route and route != "ALL":
        filtered = filtered[filtered["route"] == route]
    return filtered


def compute_kpis(filtered):
    """Compute the four headline KPI values from a (possibly filtered) df."""
    total_flights = len(filtered)
    avg_duration = filtered["flight_duration_hours"].mean() if total_flights else 0
    overnight_flights = int(filtered["is_overnight"].sum()) if total_flights else 0
    num_airlines = filtered["airline_standardized"].nunique() if total_flights else 0
    return total_flights, avg_duration, overnight_flights, num_airlines


def empty_figure(message="No data available for the selected filters."):
    """A blank, styled figure used whenever a filter combination yields 0 rows."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=14, color=COLORS["muted"]),
    )
    fig.update_layout(
        xaxis={"visible": False},
        yaxis={"visible": False},
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=360,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def base_layout(fig, title=None, height=380):
    fig.update_layout(
        title=title,
        template="plotly_white",
        colorway=CHART_COLORWAY,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Segoe UI, Arial", color=COLORS["text"], size=13),
        margin=dict(l=40, r=30, t=50, b=40),
        height=height,
    )
    return fig


# --------------------------------------------------------------------------
# CHART BUILDERS
# --------------------------------------------------------------------------
def create_route_chart(filtered):
    if filtered.empty:
        return empty_figure()

    route_counts = (
        filtered.groupby("route").size().reset_index(name="flights")
        .sort_values("flights", ascending=True)
    )

    fig = go.Figure(
        go.Bar(
            x=route_counts["flights"],
            y=route_counts["route"],
            orientation="h",
            marker_color=COLORS["primary"],
            hovertemplate="Route: %{y}<br>Flights: %{x}<extra></extra>",
        )
    )
    return base_layout(fig, height=max(360, 24 * len(route_counts)))


def create_airline_chart(filtered):
    if filtered.empty:
        return empty_figure()

    airline_counts = (
        filtered.groupby("airline_standardized").size().reset_index(name="flights")
        .sort_values("flights", ascending=False)
    )

    fig = go.Figure(
        go.Bar(
            x=airline_counts["airline_standardized"],
            y=airline_counts["flights"],
            marker_color=COLORS["accent"],
            hovertemplate="Airline: %{x}<br>Flights: %{y}<extra></extra>",
        )
    )
    return base_layout(fig)


def create_duration_chart(filtered):
    if filtered.empty:
        return empty_figure()

    avg_duration = (
        filtered.groupby("airline_standardized")["flight_duration_hours"]
        .mean()
        .reset_index()
        .sort_values("flight_duration_hours", ascending=False)
    )

    fig = go.Figure(
        go.Bar(
            x=avg_duration["airline_standardized"],
            y=avg_duration["flight_duration_hours"],
            marker_color=COLORS["success"],
            hovertemplate="Airline: %{x}<br>Avg Duration: %{y:.2f} hrs<extra></extra>",
        )
    )
    fig.update_yaxes(title_text="Avg Duration (hrs)")
    return base_layout(fig)


def create_overnight_chart(filtered):
    if filtered.empty:
        return empty_figure()

    counts = filtered["is_overnight"].value_counts()
    labels = ["Overnight" if v else "Same-Day" for v in counts.index]

    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=counts.values,
            hole=0.55,
            marker=dict(colors=[COLORS["warning"], COLORS["primary"]]),
            hovertemplate="%{label}: %{value} flights (%{percent})<extra></extra>",
        )
    )
    return base_layout(fig, height=360)


def create_date_chart(filtered):
    if filtered.empty:
        return empty_figure()

    by_date = filtered.groupby("departure_date").size().reset_index(name="flights")
    by_date = by_date.sort_values("departure_date")

    fig = go.Figure(
        go.Scatter(
            x=by_date["departure_date"],
            y=by_date["flights"],
            mode="lines+markers",
            line=dict(color=COLORS["primary_dark"], width=2),
            marker=dict(size=6),
            hovertemplate="Date: %{x}<br>Flights: %{y}<extra></extra>",
        )
    )
    return base_layout(fig, height=320)


def compute_anomaly_summary(filtered):
    """
    Data quality / anomaly metrics, mirroring the checks performed in the
    notebook (df_clean.duplicated on flight_id, UNKNOWN airline count,
    overnight flight count). No new anomaly categories are invented here.
    """
    if filtered.empty:
        return {
            "Missing / Unknown Airlines": 0,
            "Duplicate Flight IDs": 0,
            "Overnight Flights": 0,
        }

    return {
        "Missing / Unknown Airlines": int((filtered["airline_standardized"] == "UNKNOWN").sum()),
        "Duplicate Flight IDs": int(filtered["flight_id"].duplicated().sum()),
        "Overnight Flights": int(filtered["is_overnight"].sum()),
    }


# --------------------------------------------------------------------------
# UI COMPONENT BUILDERS
# --------------------------------------------------------------------------
def kpi_card(title, value, subtitle, color):
    return html.Div(
        [
            html.Div(title, style={
                "fontSize": "13px", "color": COLORS["muted"],
                "fontWeight": "600", "textTransform": "uppercase",
                "letterSpacing": "0.04em",
            }),
            html.Div(value, style={
                "fontSize": "30px", "fontWeight": "700",
                "color": COLORS["text"], "marginTop": "6px",
            }),
            html.Div(subtitle, style={
                "fontSize": "12px", "color": color, "marginTop": "4px",
                "fontWeight": "600",
            }),
        ],
        style={**CARD_STYLE, "flex": "1", "minWidth": "220px"},
    )


def create_kpi_cards(filtered):
    total_flights, avg_duration, overnight_flights, num_airlines = compute_kpis(filtered)
    overnight_pct = (overnight_flights / total_flights * 100) if total_flights else 0

    return html.Div(
        [
            kpi_card("Total Flights", f"{total_flights:,}", "In current selection", COLORS["primary"]),
            kpi_card("Avg Flight Duration", f"{avg_duration:.2f} hrs", "Across selected flights", COLORS["accent"]),
            kpi_card("Overnight Flights", f"{overnight_flights:,}", f"{overnight_pct:.1f}% of selection", COLORS["warning"]),
            kpi_card("Airlines", f"{num_airlines}", "Unique standardized carriers", COLORS["success"]),
        ],
        style={"display": "flex", "gap": "16px", "flexWrap": "wrap", "marginBottom": "24px"},
    )


def anomaly_card(label, value, color):
    return html.Div(
        [
            html.Div(label, style={
                "fontSize": "13px", "color": COLORS["muted"], "fontWeight": "600",
            }),
            html.Div(str(value), style={
                "fontSize": "24px", "fontWeight": "700", "color": color, "marginTop": "4px",
            }),
        ],
        style={**CARD_STYLE, "flex": "1", "minWidth": "200px"},
    )


def create_anomaly_section(filtered):
    summary = compute_anomaly_summary(filtered)
    colors = [COLORS["danger"], COLORS["warning"], COLORS["primary"]]
    cards = [
        anomaly_card(label, value, color)
        for (label, value), color in zip(summary.items(), colors)
    ]
    return html.Div(cards, style={"display": "flex", "gap": "16px", "flexWrap": "wrap"})


def section_header(title, subtitle=None):
    children = [html.H3(title, style={
        "margin": "0", "color": COLORS["text"], "fontSize": "18px", "fontWeight": "700",
    })]
    if subtitle:
        children.append(html.Div(subtitle, style={
            "fontSize": "13px", "color": COLORS["muted"], "marginTop": "2px",
        }))
    return html.Div(children, style={"marginBottom": "14px"})


def chart_card(children, flex="1", min_width="320px"):
    return html.Div(children, style={**CARD_STYLE, "flex": flex, "minWidth": min_width})


# --------------------------------------------------------------------------
# FLASK + DASH SETUP
# --------------------------------------------------------------------------
server = Flask(__name__)

app = Dash(
    __name__,
    server=server,
    routes_pathname_prefix="/dashboard/",
    title="Airline Operations Analytics",
)


@server.route("/")
def index():
    return (
        '<html><body style="font-family: Arial; padding: 40px;">'
        "<h2>Airline Operations Analytics</h2>"
        '<p>The dashboard is available at <a href="/dashboard/">/dashboard/</a>.</p>'
        "</body></html>"
    )


def build_error_layout():
    return html.Div(
        html.Div(
            [
                html.H2("Unable to load dashboard", style={"color": COLORS["danger"]}),
                html.P(LOAD_ERROR, style={"color": COLORS["text"]}),
            ],
            style={**CARD_STYLE, "maxWidth": "600px", "margin": "80px auto"},
        ),
        style={"backgroundColor": COLORS["bg"], "minHeight": "100vh", "padding": "20px"},
    )


def build_dashboard_layout():
    has_dates = df["departure_date"].notna().any()

    return html.Div(
        [
            # Header
            html.Div(
                [
                    html.H1("Airline Operations Analytics", style={
                        "margin": "0", "color": "white", "fontSize": "26px", "fontWeight": "700",
                    }),
                    html.Div("Cleaned Airline Data · Data Engineering Pipeline", style={
                        "color": "#DBEAFE", "fontSize": "13px", "marginTop": "4px",
                    }),
                ],
                style={
                    "background": f"linear-gradient(135deg, {COLORS['primary_dark']}, {COLORS['primary']})",
                    "padding": "24px 32px", "borderRadius": "12px", "marginBottom": "24px",
                },
            ),

            # KPI cards (populated by callback)
            html.Div(id="kpi-cards"),

            # Filters
            html.Div(
                [
                    section_header("Filters"),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label("Airline", style={"fontSize": "13px", "fontWeight": "600", "color": COLORS["muted"]}),
                                    dcc.Dropdown(
                                        id="airline-filter",
                                        options=[{"label": "All Airlines", "value": "ALL"}]
                                        + [{"label": a, "value": a} for a in AIRLINE_OPTIONS],
                                        value="ALL",
                                        clearable=False,
                                        style={"marginTop": "6px"},
                                    ),
                                ],
                                style={"flex": "1", "minWidth": "220px"},
                            ),
                            html.Div(
                                [
                                    html.Label("Route", style={"fontSize": "13px", "fontWeight": "600", "color": COLORS["muted"]}),
                                    dcc.Dropdown(
                                        id="route-filter",
                                        options=[{"label": "All Routes", "value": "ALL"}]
                                        + [{"label": r, "value": r} for r in ROUTE_OPTIONS],
                                        value="ALL",
                                        clearable=False,
                                        style={"marginTop": "6px"},
                                    ),
                                ],
                                style={"flex": "1", "minWidth": "220px"},
                            ),
                        ],
                        style={"display": "flex", "gap": "20px", "flexWrap": "wrap"},
                    ),
                ],
                style={**CARD_STYLE, "marginBottom": "24px"},
            ),

            # Route performance
            html.Div(
                [
                    section_header("Route Performance", "Flight traffic by route"),
                    dcc.Graph(id="route-chart", config={"displayModeBar": False}),
                ],
                style={**CARD_STYLE, "marginBottom": "24px"},
            ),

            # Airline performance
            html.Div(
                [
                    section_header("Airline Performance"),
                    html.Div(
                        [
                            chart_card([
                                html.Div("Flights by Airline", style={"fontWeight": "600", "marginBottom": "8px", "color": COLORS["text"]}),
                                dcc.Graph(id="airline-chart", config={"displayModeBar": False}),
                            ]),
                            chart_card([
                                html.Div("Avg Flight Duration by Airline", style={"fontWeight": "600", "marginBottom": "8px", "color": COLORS["text"]}),
                                dcc.Graph(id="duration-chart", config={"displayModeBar": False}),
                            ]),
                        ],
                        style={"display": "flex", "gap": "16px", "flexWrap": "wrap"},
                    ),
                ],
                style={"marginBottom": "24px"},
            ),

            # Overnight analysis + optional date trend
            html.Div(
                [
                    section_header("Overnight / Cross-Day Analysis"),
                    html.Div(
                        [
                            chart_card(
                                [
                                    html.Div("Overnight vs Same-Day Flights", style={"fontWeight": "600", "marginBottom": "8px", "color": COLORS["text"]}),
                                    dcc.Graph(id="overnight-chart", config={"displayModeBar": False}),
                                ],
                                min_width="280px",
                            ),
                        ]
                        + (
                            [
                                chart_card(
                                    [
                                        html.Div("Flights by Departure Date", style={"fontWeight": "600", "marginBottom": "8px", "color": COLORS["text"]}),
                                        dcc.Graph(id="date-chart", config={"displayModeBar": False}),
                                    ],
                                    min_width="320px",
                                )
                            ]
                            if has_dates
                            else []
                        ),
                        style={"display": "flex", "gap": "16px", "flexWrap": "wrap"},
                    ),
                ],
                style={"marginBottom": "24px"},
            ),

            # Data quality / anomalies
            html.Div(
                [
                    section_header(
                        "Data Quality / Anomaly Insights",
                        "Issues identified during the data-engineering pipeline (not operational delay metrics)",
                    ),
                    html.Div(id="anomaly-section"),
                ],
                style={**CARD_STYLE, "marginBottom": "24px"},
            ),
        ],
        style={
            "backgroundColor": COLORS["bg"],
            "minHeight": "100vh",
            "padding": "28px 32px",
            "fontFamily": "Inter, Segoe UI, Arial, sans-serif",
        },
    )


app.layout = build_error_layout() if LOAD_ERROR else build_dashboard_layout()


# --------------------------------------------------------------------------
# CALLBACKS
# --------------------------------------------------------------------------
if LOAD_ERROR is None:
    has_dates_global = df["departure_date"].notna().any()

    outputs = [
        Output("kpi-cards", "children"),
        Output("route-chart", "figure"),
        Output("airline-chart", "figure"),
        Output("duration-chart", "figure"),
        Output("overnight-chart", "figure"),
        Output("anomaly-section", "children"),
    ]
    if has_dates_global:
        outputs.append(Output("date-chart", "figure"))

    @app.callback(
        outputs,
        [Input("airline-filter", "value"), Input("route-filter", "value")],
    )
    def update_dashboard(selected_airline, selected_route):
        filtered = filter_data(selected_airline, selected_route)

        kpi_children = create_kpi_cards(filtered)
        route_fig = create_route_chart(filtered)
        airline_fig = create_airline_chart(filtered)
        duration_fig = create_duration_chart(filtered)
        overnight_fig = create_overnight_chart(filtered)
        anomaly_children = create_anomaly_section(filtered)

        result = [kpi_children, route_fig, airline_fig, duration_fig, overnight_fig, anomaly_children]
        if has_dates_global:
            result.append(create_date_chart(filtered))
        return tuple(result)


# --------------------------------------------------------------------------
# APP STARTUP
# --------------------------------------------------------------------------
if __name__ == "__main__":
    server.run(debug=True, port=5000)