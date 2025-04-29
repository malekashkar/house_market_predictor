import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# --- Logging ---
def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

# --- Data Loading & Splitting ---
def load_and_filter(paths, top_n=5):
    df = None
    for p in paths:
        try:
            df = pd.read_csv(p, parse_dates=['Date'])
            logging.info(f"Loaded data from {p}")
            break
        except FileNotFoundError:
            continue
    if df is None:
        logging.error("CSV not found in provided paths.")
        raise FileNotFoundError
    df = df.dropna(subset=['SizeRank'])
    regions = df.nsmallest(top_n, 'SizeRank')['RegionName'].unique().tolist()
    logging.info(f"Top {top_n} regions: {regions}")
    return df[df.RegionName.isin(regions)].sort_values(['RegionName', 'Date']), regions


def split_by_date(df, train_end, val_end):
    return (
        df[df.Date <= train_end],
        df[(df.Date > train_end) & (df.Date <= val_end)],
        df[df.Date > val_end]
    )

# --- Metrics ---
def _calc_metrics(true, pred):
    mae = mean_absolute_error(true, pred)
    rmse = np.sqrt(mean_squared_error(true, pred))
    mape = (np.abs((true - pred) / true.replace(0, np.nan)).mean()) * 100
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

# --- Time-Series Baselines ---
def evaluate_ts(train, val):
    results = {}
    for name, args in [('ARIMA', {'order': (1, 1, 1)}), ('ETS', {'seasonal': 'add', 'seasonal_periods': 12})]:
        try:
            model = ARIMA(train, **args).fit() if name == 'ARIMA' else ExponentialSmoothing(train, **args).fit()
            pred = pd.Series(model.forecast(len(val)), index=val.index)
            m = _calc_metrics(val, pred)
            logging.info(f"{name} | MAE={m['MAE']:.2f}, RMSE={m['RMSE']:.2f}, MAPE={m['MAPE']:.2f}%")
        except Exception as e:
            logging.warning(f"{name} failed: {e}")
            m = {'MAE': np.nan, 'RMSE': np.nan, 'MAPE': np.nan}
        results[name] = m
    return results

# --- Machine Learning ---
def prepare_ml(df, train_end, val_end, lags):
    df = df.copy()
    df['month'] = df['Date'].dt.month
    df['quarter'] = df['Date'].dt.quarter
    df = pd.get_dummies(df, columns=['RegionName'], drop_first=True, dtype=float)
    features = lags + ['month', 'quarter'] + [c for c in df if c.startswith('RegionName_')] # type: ignore
    X_tr = df[df.Date <= train_end][features].dropna()
    y_tr = df.loc[X_tr.index, 'ZHVI']
    X_va = df[(df.Date > train_end) & (df.Date <= val_end)][features].dropna()
    y_va = df.loc[X_va.index, 'ZHVI']
    logging.info(f"ML shapes | Train: {X_tr.shape}, Val: {X_va.shape}")
    return (X_tr, y_tr, X_va, y_va), features


def train_ml(X_tr, y_tr, X_va, y_va):
    scaler = StandardScaler().fit(X_tr)
    Xt, Xv = scaler.transform(X_tr), scaler.transform(X_va)
    cv = TimeSeriesSplit(n_splits=3)
    rf_gs = GridSearchCV(
        RandomForestRegressor(random_state=42),
        {'n_estimators': [100, 200], 'max_depth': [5, 10]},
        cv=cv, scoring='neg_mean_absolute_error'
    )
    rf_gs.fit(X_tr, y_tr)
    logging.info(f"RF params: {rf_gs.best_params_}")
    rf_pred = rf_gs.predict(X_va)
    rf_m = _calc_metrics(y_va, rf_pred)
    svr = SVR().fit(Xt, y_tr)
    svr_pred = svr.predict(Xv)
    svr_m = _calc_metrics(y_va, svr_pred)
    logging.info(f"SVR | MAE={svr_m['MAE']:.2f}, RMSE={svr_m['RMSE']:.2f}, MAPE={svr_m['MAPE']:.2f}%")
    return {'RF': (rf_gs, rf_m), 'SVR': (svr, svr_m), 'scaler': scaler}

# --- Deep Learning (MLP) ---
def train_dl(X_tr, y_tr, X_va, y_va, device, epochs=20):
    dev = torch.device(device)
    tX = torch.tensor(X_tr.values, dtype=torch.float32).to(dev)
    ty = torch.tensor(y_tr.values, dtype=torch.float32).unsqueeze(1).to(dev)
    vX = torch.tensor(X_va.values, dtype=torch.float32).to(dev)
    train_loader = DataLoader(TensorDataset(tX, ty), batch_size=32, shuffle=True)

    class MLP(nn.Module):
        def __init__(self, d_in):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d_in, 64), nn.ReLU(),
                nn.Linear(64, 32), nn.ReLU(),
                nn.Linear(32, 1)
            )
        def forward(self, x):
            return self.net(x)

    model = MLP(X_tr.shape[1]).to(dev)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.L1Loss()
    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        pred_v = model(vX).cpu().numpy().squeeze()
    dl_m = _calc_metrics(y_va, pd.Series(pred_v, index=y_va.index))
    logging.info(f"DL MLP | MAE={dl_m['MAE']:.2f}, RMSE={dl_m['RMSE']:.2f}, MAPE={dl_m['MAPE']:.2f}%")
    return model, dl_m

# --- Test Evaluation ---
def evaluate_test(test, features, ml_models, scaler, dl_model, device):
    X_te = test[features].dropna()
    y_te = test.loc[X_te.index, 'ZHVI']
    rf, _ = ml_models['RF']
    svr, _ = ml_models['SVR']
    rf_p = rf.predict(X_te)
    svr_p = svr.predict(scaler.transform(X_te))
    rf_m = _calc_metrics(y_te, rf_p)
    svr_m = _calc_metrics(y_te, svr_p)
    logging.info(f"Test RF | {rf_m}")
    logging.info(f"Test SVR | {svr_m}")
    dev = torch.device(device)
    tX = torch.tensor(X_te.values, dtype=torch.float32).to(dev)
    with torch.no_grad():
        dl_p = dl_model(tX).cpu().numpy().squeeze()
    dl_m = _calc_metrics(y_te, pd.Series(dl_p, index=y_te.index))
    logging.info(f"Test DL MLP | {dl_m}")

# --- Plotting ---
def plot_validation(val, features, ml_models, scaler, dl_model, device):
    dates = val.index
    X_va = val[features].dropna()
    y_va = val.loc[X_va.index, 'ZHVI']
    rf, _ = ml_models['RF']
    svr, _ = ml_models['SVR']
    rf_p = rf.predict(X_va)
    svr_p = svr.predict(scaler.transform(X_va))
    dev = torch.device(device)
    tX = torch.tensor(X_va.values, dtype=torch.float32).to(dev)
    with torch.no_grad():
        dl_p = dl_model(tX).cpu().numpy().squeeze()
    plt.figure(figsize=(10, 6))
    plt.plot(dates, y_va, 'k.', label='Actual')
    plt.plot(dates, rf_p, 'r--', label='RF')
    plt.plot(dates, svr_p, 'b--', label='SVR')
    plt.plot(dates, dl_p, 'g--', label='MLP')
    plt.legend()
    plt.title('Validation Predictions')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_baseline(train, val, regions):
    plt.figure(figsize=(10, 4 * len(regions)))
    for i, r in enumerate(regions, 1):
        tr = train[train.RegionName == r].set_index('Date')['ZHVI']
        va = val[val.RegionName == r].set_index('Date')['ZHVI']
        preds = evaluate_ts(tr, va)
        plt.subplot(len(regions), 1, i)
        plt.plot(va.index, va, 'k.', label='Actual')
        for nm in preds:
            plt.plot(va.index, pd.Series(tr.values[:len(va)], index=va.index), label=nm)
        plt.title(f'{r} Baselines')
        plt.legend()
        plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# --- Forecasting ---
def forecast_future(df, train_end, val_end, ml_models, scaler, features, regions, horizon=12):
    idx = df.Date <= val_end
    X = df[idx][features].dropna()
    y = df.loc[X.index, 'ZHVI']
    rf, _ = ml_models['RF']
    rf_final = RandomForestRegressor(**rf.best_params_, random_state=42)
    rf_final.fit(X, y)
    last_date = df.Date.max()
    dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=horizon, freq='ME')
    forecasts = {}
    for r in regions:
        hist = df[df.RegionName == r].sort_values('Date')
        z = hist['ZHVI'].values
        cur = {'ZHVI_lag_1': z[-1], 'ZHVI_lag_3': z[-3], 'ZHVI_lag_6': z[-6]}
        preds = []
        for d in dates:
            feat = {**cur, 'month': d.month, 'quarter': d.quarter}
            for c in features:
                if c.startswith('RegionName_'):
                    feat[c] = 1.0 if c.split('_')[1] == r else 0.0
            xv = pd.DataFrame([feat])[features]
            p = rf_final.predict(xv)[0]
            preds.append(p)
            cur['ZHVI_lag_6'], cur['ZHVI_lag_3'], cur['ZHVI_lag_1'] = cur['ZHVI_lag_3'], cur['ZHVI_lag_1'], p
        forecasts[r] = pd.Series(preds, index=dates)
    plt.figure(figsize=(10, 4 * len(regions)))
    for i, r in enumerate(regions, 1):
        h = df[df.RegionName == r].set_index('Date')['ZHVI']
        plt.subplot(len(regions), 1, i)
        plt.plot(h.index, h, 'k-', label='Hist')
        plt.plot(forecasts[r].index, forecasts[r], 'r--', label='Forecast')
        plt.title(r)
        plt.legend()
        plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# --- Main ---
def main():
    configure_logging()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df, regions = load_and_filter(['./data/features.csv'])
    train, val, test = split_by_date(df, '2020-12-31', '2022-12-31')
    for r in regions:
        tr = train[train.RegionName == r].set_index('Date')['ZHVI']
        va = val[val.RegionName == r].set_index('Date')['ZHVI']
        if len(tr) > 15 and len(va) > 0:
            evaluate_ts(tr, va)
    (X_tr, y_tr, X_va, y_va), feats = prepare_ml(df, '2020-12-31', '2022-12-31', ['ZHVI_lag_1', 'ZHVI_lag_3', 'ZHVI_lag_6'])
    ml_models = train_ml(X_tr, y_tr, X_va, y_va)
    dl_model, _ = train_dl(X_tr, y_tr, X_va, y_va, str(device))
    evaluate_test(test, feats, ml_models, ml_models['scaler'], dl_model, str(device))
    plot_validation(val.set_index('Date'), feats, ml_models, ml_models['scaler'], dl_model, str(device))
    plot_baseline(train, val, regions)
    forecast_future(df, '2020-12-31', '2022-12-31', ml_models, ml_models['scaler'], feats, regions)

if __name__ == '__main__':
    main()
