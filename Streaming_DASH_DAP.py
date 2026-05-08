import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import dash_daq as daq
import websocket
import json
import threading

# --- Configuración del Streaming Directo ---
ultimo_precio = 0.0
tendencia = "gray"

def on_message(ws, message):
    global ultimo_precio, tendencia
    data = json.loads(message)
    # El stream 'btcusdt@ticker' envía el precio actual en la llave 'c'
    nuevo_precio = float(data['c'])
    
    if nuevo_precio > ultimo_precio:
        tendencia = "#00ff00"
    elif nuevo_precio < ultimo_precio:
        tendencia = "#ff0000"
        
    ultimo_precio = nuevo_precio

def run_ws():
    # Conexión directa al stream de Binance
    ws = websocket.WebSocketApp(
        "wss://stream.binance.com:9443/ws/btcusdt@ticker",
        on_message=on_message
    )
    ws.run_forever()

# Iniciamos el socket en un hilo para que no detenga el servidor Dash
threading.Thread(target=run_ws, daemon=True).start()

# --- Aplicación Dash ---
app = dash.Dash(__name__)

app.layout = html.Div(style={'backgroundColor': '#111111', 'color': '#e0e0e0', 'padding': '40px', 'height': '100vh'}, children=[
    html.H2("MONITOR BINANCE REAL-TIME (WSS)", style={'textAlign': 'center', 'color': '#00e5ff'}),
    
    html.Div(style={'display': 'flex', 'justifyContent': 'center', 'gap': '50px', 'marginTop': '60px'}, children=[
        
        # Indicador LED de actividad
        html.Div([
            html.P("Status"),
            daq.Indicator(id='status-led', value=True, color="#00ff00", label="ONLINE")
        ]),

        # Pantalla LED de precio
        html.Div([
            html.P("BTC / USDT"),
            daq.LEDDisplay(
                id='btc-price-led',
                value="0.00",
                color="#00e5ff",
                size=70,
                backgroundColor="#0a192f"
            )
        ]),

        # Gauge de volatilidad relativa
        # Rango ajustado para notar el movimiento de la aguja
        daq.Gauge(
            id='btc-gauge',
            min=60000,
            max=80000,
            value=70000,
            label="Rango de Mercado",
            showCurrentValue=True,
            units="USDT",
            color={"gradient":True,"ranges":{"red":[60000,65000],"yellow":[65000,75000],"green":[75000,80000]}}
        )
    ]),

    dcc.Interval(id='refresh', interval=300, n_intervals=0)
])

@app.callback(
    [Output('btc-price-led', 'value'),
     Output('btc-gauge', 'value'),
     Output('status-led', 'color')],
    [Input('refresh', 'n_intervals')]
)
def update_ui(n):
    # Formateamos el precio para el LED (debe ser string o número)
    precio_str = f"{ultimo_precio:.2f}"
    return precio_str, ultimo_precio, tendencia

if __name__ == '__main__':
    app.run(debug=True)