import dash
from dash import Dash, html, dcc, Output, Input, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import json
import websocket
import threading
import time
from collections import deque
import pandas as pd
import numpy as np
from datetime import datetime, timedelta   # sin timezone

app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.title = "Dashboard BTC/USDT en Vivo - Binance"

# Variables globales
price_history = deque(maxlen=100000)
latest_price = {"price": None, "timestamp": None, "prev_price": None}
ws_thread = None
ws = None

threshold_value = 60000.0
alert_period_value = 5
window_sec_value = 30

alert_active = False
consecutive_above_threshold = 0

# WebSocket thread
def binance_websocket_thread():
    global ws

    def on_message(ws_obj, message):
        global consecutive_above_threshold, alert_active
        try:
            data = json.loads(message)
            if data.get('e') == 'trade':
                price = float(data['p'])
                ts_ms = data['T']
                ts = pd.to_datetime(ts_ms, unit='ms')

                with threading.Lock():
                    prev = latest_price["price"]
                    latest_price["price"] = price
                    latest_price["timestamp"] = ts
                    latest_price["prev_price"] = prev
                    price_history.append({"time": ts, "price": price})

                if price > threshold_value:
                    consecutive_above_threshold += 1
                    if consecutive_above_threshold >= alert_period_value:
                        alert_active = True
                else:
                    consecutive_above_threshold = 0
                    alert_active = False

        except Exception as e:
            print(f"Error en on_message: {e}")

    def on_error(ws_obj, error):
        print(f"WebSocket error: {error}")
        time.sleep(5)

    def on_close(ws_obj, close_status_code, close_msg):
        print("WebSocket cerrado → reconectando...")
        time.sleep(10)
        start_websocket()

    def on_open(ws_obj):
        print("Conectado a Binance WS")
        ws_obj.send(json.dumps({
            "method": "SUBSCRIBE",
            "params": ["btcusdt@trade"],
            "id": 1
        }))

    ws_url = "wss://stream.binance.com:9443/ws"
    ws = websocket.WebSocketApp(ws_url, on_open=on_open, on_message=on_message,
                                on_error=on_error, on_close=on_close)
    ws.run_forever(ping_interval=25, ping_timeout=10)

def start_websocket():
    global ws_thread
    if ws_thread is None or not ws_thread.is_alive():
        ws_thread = threading.Thread(target=binance_websocket_thread, daemon=True)
        ws_thread.start()
        print("Thread WebSocket iniciado")

start_websocket()

# Layout (mismo que antes)
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.Img(src="https://bitcoin.org/img/icons/opengraph.png?1646854750", height="80px", className="mb-3"),
            html.H1("Dashboard BTC/USDT - Binance", className="text-center text-warning mb-3"),
            html.P("Precios en tiempo real • Ventanas personalizables • Alertas por umbral",
                   className="text-center text-muted"),
        ], width=12)
    ], className="mb-5"),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Configuración"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Ventana (segundos)"),
                            dcc.Input(id="window-sec", type="number", value=30, min=5, max=600, step=5,
                                      className="form-control bg-dark text-white border-secondary"),
                        ], width=4),
                        dbc.Col([
                            dbc.Label("Umbral precio (USDT)"),
                            dcc.Input(id="threshold", type="number", value=60000.0, step=100,
                                      className="form-control bg-dark text-white border-secondary"),
                        ], width=4),
                        dbc.Col([
                            dbc.Label("Lecturas consecutivas alerta"),
                            dcc.Input(id="alert-period", type="number", value=5, min=1, max=50, step=1,
                                      className="form-control bg-dark text-white border-secondary"),
                        ], width=4),
                    ], className="g-4"),
                    dbc.Button("Aplicar cambios", id="apply-btn", color="warning", className="mt-3 w-100"),
                ])
            ], color="dark", outline=True),
        ], width=12)
    ], className="mb-5"),

    html.Div(id="alert-container", className="mb-4"),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H3("Precio Actual", className="text-center"),
                    html.Div(id="live-price", className="display-4 text-center fw-bold text-success"),
                    html.Div(id="price-delta", className="text-center fs-4 text-muted mt-2"),
                    html.Div(id="window-avg", className="text-center fs-4 text-info mt-3"),
                    html.Small(id="last-update", className="d-block text-center text-muted mt-2"),
                ])
            ], color="dark", outline=True),
        ], width=12)
    ], className="mb-5"),

    dbc.Row([
        dbc.Col([
            html.H4("Últimos 5 minutos - Precios crudos", className="text-center mb-3"),
            dcc.Graph(id="live-chart-raw", style={"height": "50vh"}),
        ], width=6),
        dbc.Col([
            html.H4("Última hora - Promedios por minuto (de promedios de ventanas)", className="text-center mb-3"),
            dcc.Graph(id="live-chart-avg", style={"height": "50vh"}),
        ], width=6),
    ]),

    dcc.Interval(id="interval-update", interval=800, n_intervals=0),
    dcc.Store(id="params-store", data={"window_sec": 30, "threshold": 60000.0, "alert_period": 5}),
], fluid=True, className="p-4", style={"backgroundColor": "#0d1117"})

@app.callback(
    Output("params-store", "data"),
    Input("apply-btn", "n_clicks"),
    [State("window-sec", "value"), State("threshold", "value"), State("alert-period", "value")],
    prevent_initial_call=True
)
def update_params(n_clicks, window_sec, threshold, alert_period):
    global threshold_value, alert_period_value, window_sec_value
    threshold_value = float(threshold) if threshold is not None else 60000.0
    alert_period_value = int(alert_period) if alert_period is not None else 5
    window_sec_value = int(window_sec) if window_sec is not None else 30
    return {"window_sec": window_sec_value, "threshold": threshold_value, "alert_period": alert_period_value}

@app.callback(
    [Output("live-price", "children"),
     Output("price-delta", "children"),
     Output("last-update", "children"),
     Output("window-avg", "children"),
     Output("live-chart-raw", "figure"),
     Output("live-chart-avg", "figure"),
     Output("alert-container", "children")],
    Input("interval-update", "n_intervals"),
    State("params-store", "data")
)
def update_dashboard(n, params):
    if latest_price["price"] is None:
        return "Conectando...", "", "Esperando...", "Calculando...", go.Figure(), go.Figure(), ""

    price = latest_price["price"]
    prev = latest_price["prev_price"]
    ts = latest_price["timestamp"]

    delta_str = f"{price - prev:+.2f} ({(price - prev)/prev*100:+.2f}%)" if prev else "—"
    price_str = f"{price:,.2f} USDT"
    update_str = f"Últ. act.: {ts.strftime('%H:%M:%S.%f')[:-3]}" if ts else ""

    # Snapshot seguro
    with threading.Lock():
        history_copy = list(price_history)

    if len(history_copy) < 5:
        return price_str, delta_str, update_str, "Insuficientes datos", go.Figure(), go.Figure(), ""

    df = pd.DataFrame(history_copy)
    df.set_index('time', inplace=True)
    df.sort_index(inplace=True)

    # Comparaciones usando tz-naive
    five_min_ago = datetime.now() - timedelta(minutes=5)
    df_5min = df[df.index >= five_min_ago]

    fig_raw = go.Figure(go.Scatter(
        x=df_5min.index,
        y=df_5min['price'],
        mode='lines+markers',
        line=dict(color='#00cc96', width=1.5),
        marker=dict(size=4),
        name='Precio'
    ))
    fig_raw.update_layout(
        title="Últimos 5 minutos - Precios crudos",
        xaxis_title="Tiempo",
        yaxis_title="Precio (USDT)",
        template="plotly_dark",
        height=450,
        hovermode="x unified"
    )

    avg_str = "Sin datos recientes"
    if not df_5min.empty:
        res_w = df_5min['price'].resample(f'{int(window_sec_value)}s').mean().dropna()
        if not res_w.empty:
            last_avg = res_w.iloc[-1]
            avg_str = f"Promedio ventana {window_sec_value}s: {last_avg:,.2f} USDT"

    # Gráfica 2
    one_hour_ago = datetime.now() - timedelta(hours=1)
    df_hour = df[df.index >= one_hour_ago]

    fig_avg = go.Figure()
    if not df_hour.empty:
        window_res = df_hour['price'].resample(f'{int(window_sec_value)}s').mean().dropna()

        if not window_res.empty:
            df_window = window_res.to_frame(name='price').reset_index()
            df_window.set_index('time', inplace=True)
            minute_avg = df_window.groupby(pd.Grouper(freq='min'))['price'].mean().reset_index()

            if not minute_avg.empty:
                fig_avg.add_trace(go.Scatter(
                    x=minute_avg['time'],
                    y=minute_avg['price'],
                    mode='lines',
                    line=dict(color='#ffcc00', width=2.5),
                    name='Promedio por minuto'
                ))

    fig_avg.update_layout(
        title="Última hora - Promedios por minuto (de promedios de ventanas)",
        xaxis_title="Tiempo",
        yaxis_title="Precio promedio (USDT)",
        template="plotly_dark",
        height=450,
        hovermode="x unified"
    )

    alert = dbc.Alert(
        f"¡ALERTA! Precio > {threshold_value:,.0f} USDT durante {alert_period_value} lecturas seguidas",
        color="danger",
        dismissable=True,
        is_open=alert_active,
        className="text-center fs-5"
    ) if alert_active else ""

    return price_str, delta_str, update_str, avg_str, fig_raw, fig_avg, alert


if __name__ == '__main__':
    print("Iniciando dashboard → http://127.0.0.1:8050")
    app.run(debug=True, port=8050)