import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier

# ============================================================
# INDICADORES
# ============================================================

def calc_rsi(series, length=14):
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.rolling(length).mean()
    avg_loss = loss.rolling(length).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calc_atr(df, length=14):
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift()).abs()
    lc = (df["Low"]  - df["Close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(length).mean()


def calc_macd_hist(series):
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd - signal


def calc_cmf(df, length=20):
    mfm = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / (df["High"] - df["Low"])
    mfv = mfm * df["Volume"]
    return mfv.rolling(length).sum() / df["Volume"].rolling(length).sum()


# ============================================================
# CARGA Y LIMPIEZA DE MULTIINDEX (TU PROBLEMA REAL)
# ============================================================

def load_data(ticker):

    df = yf.download(ticker, period="2y")

    if df.empty:
        raise ValueError("No hay datos para el ticker.")

    df = df.reset_index()

    # Aplastar MultiIndex → dejarlo como 'Close_NVDA'
    newcols = []
    for col in df.columns:
        if isinstance(col, tuple):
            name = "_".join([str(c) for c in col if c])
        else:
            name = str(col)
        newcols.append(name)

    df.columns = newcols

    # Normalizar nombres → sacar columnas correctas
    rename_map = {}
    for c in df.columns:
        low = c.lower()
        if "close" in low:
            rename_map[c] = "Close"
        elif "open" in low:
            rename_map[c] = "Open"
        elif "high" in low:
            rename_map[c] = "High"
        elif "low" in low:
            rename_map[c] = "Low"
        elif "vol" in low:
            rename_map[c] = "Volume"
        elif "date" in low:
            rename_map[c] = "Date"

    df = df.rename(columns=rename_map)

    # QUEDARNOS SOLO CON LAS NECESARIAS
    columnas = ["Date", "Open", "High", "Low", "Close", "Volume"]
    df = df[[c for c in columnas if c in df.columns]]

    # Si no viene Close → FAIL
    if "Close" not in df.columns:
        raise ValueError(f"No se encontró columna Close. Columnas reales: {df.columns}")

    # Forzar numérico
    for col in ["Close", "Open", "High", "Low", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna()

    # Indicadores
    df["RSI"]        = calc_rsi(df["Close"])
    df["ATR"]        = calc_atr(df)
    df["MACD_hist"]  = calc_macd_hist(df["Close"])
    df["CMF"]        = calc_cmf(df)

    # Volatilidad asegurando SERIE
    vol = df["Close"].pct_change().rolling(20).std()
    vol = vol.replace([np.inf, -np.inf], np.nan).bfill().ffill()
    df["Vol"] = vol.astype(float)

    # Normalización segura
    norm = df["Close"] / df["Vol"]
    norm = norm.replace([np.inf, -np.inf], np.nan).bfill().ffill()
    df["Norm_Close"] = norm.astype(float)

    df = df.dropna()
    return df

# ============================================================
# CREAR SECUENCIAS
# ============================================================

def create_sequences(df, lookback=14):
    X, y = [], []
    feats = ["RSI", "ATR", "MACD_hist", "CMF", "Norm_Close"]
    target = df["Close"].shift(-2) > df["Close"]

    for i in range(len(df) - lookback - 2):
        window = df[feats].iloc[i:i+lookback].values.flatten()
        X.append(window)
        y.append(int(target.iloc[i+lookback]))

    return np.array(X), np.array(y)


# ============================================================
# MODELO
# ============================================================

def train_model(X, y):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        Xs, y, test_size=0.2, shuffle=False
    )

    model = GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05
    )

    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)

    return model, scaler, acc


# ============================================================
# MONTECARLO
# ============================================================

def montecarlo(df, runs=2000, steps=5):
    last = df["Close"].iloc[-1]
    sigma = df["Vol"].iloc[-1]
    if sigma <= 0 or np.isnan(sigma):
        sigma = 0.0001

    sims = []
    for _ in range(runs):
        price = last
        for _ in range(steps):
            price *= (1 + np.random.normal(0, sigma))
        sims.append(price)

    return float((np.array(sims) > last).mean())


# ============================================================
# TP / SL
# ============================================================

def get_tp_sl(df):
    price = df["Close"].iloc[-1]
    atr = df["ATR"].iloc[-1]
    return float(price + atr*2.2), float(price - atr*1.5)


# ============================================================
# GENERAR ANÁLISIS
# ============================================================

def generate_analysis(ticker):
    df = load_data(ticker)

    X, y = create_sequences(df)

    if len(X) < 20:
        raise ValueError("No hay suficientes datos para entrenar el modelo.")

    model, scaler, acc = train_model(X, y)
    prob = montecarlo(df)
    TP, SL = get_tp_sl(df)
    price = df["Close"].iloc[-1]

    signal = "ALCISTA" if prob > 0.55 else "BAJISTA" if prob < 0.45 else "NEUTRAL"

    return {
        "ticker": ticker,
        "accuracy": round(float(acc), 4),
        "probability": round(float(prob), 4),
        "price": round(float(price), 2),
        "TP": round(TP, 2),
        "SL": round(SL, 2),
        "signal": signal
    }