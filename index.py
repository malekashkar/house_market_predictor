import pandas as pd
import numpy as np
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using PyTorch on device: {device}")

# 1) Load feature-enhanced data
try:
    df = pd.read_csv('./data/features.csv', parse_dates=['Date'])
except FileNotFoundError:
    try:
        df = pd.read_csv('Metro_zhvi_features.csv', parse_dates=['Date'])
        print("Loaded 'Metro_zhvi_features.csv' from local directory.")
    except FileNotFoundError:
        print("Error: Metro_zhvi_features.csv not found in /mnt/data/ or local directory.")
        print("Please ensure the file is in the correct directory or update the path.")
        exit()

if 'SizeRank' not in df.columns or 'RegionName' not in df.columns:
    print("Error: 'SizeRank' or 'RegionName' column not found in the CSV.")
    exit()

df_ranked = df.dropna(subset=['SizeRank'])
df_ranked = df_ranked[df_ranked['SizeRank'] > 0]

if df_ranked.empty:
     print("Warning: No regions found with SizeRank > 0. Using all regions or stopping.")
     top5 = df['RegionName'].unique().tolist()[:5]
     if not top5:
         print("Error: No regions found at all.")
         exit()
     print(f"Using first 5 available regions: {top5}")
else:
     top5 = df_ranked.drop_duplicates('RegionName').sort_values('SizeRank')['RegionName'].tolist()[:5]

if not top5:
    print("Error: Could not determine top 5 regions.")
    exit()

print(f"Focusing on regions: {top5}")
df = df[df['RegionName'].isin(top5)].copy()
df.sort_values(by=['RegionName', 'Date'], inplace=True)

# 2) Train/validation/test split by date
train_end = '2020-12-31'
val_end   = '2022-12-31'

train_df = df[df['Date'] <= train_end].copy()
val_df   = df[(df['Date'] > train_end) & (df['Date'] <= val_end)].copy()
test_df  = df[df['Date'] > val_end].copy()

# 3) Baseline time-series models per region
ts_metrics = {}
print("\n--- Training Baseline Time Series Models ---")
for region in top5:
    print(f"Region: {region}")
    ser_train_region = train_df[train_df['RegionName']==region]
    ser_val_region   = val_df[val_df['RegionName']==region]

    if ser_train_region.empty or ser_val_region.empty:
        print(f"Skipping region {region} due to missing data in train or validation set.")
        continue

    ser_train = ser_train_region.set_index('Date')['ZHVI']
    ser_val   = ser_val_region.set_index('Date')['ZHVI']

    if len(ser_train) < 15:
         print(f"Skipping region {region} due to insufficient training data ({len(ser_train)} points).")
         continue
    if len(ser_val) == 0:
         print(f"Skipping region {region} due to no validation data.")
         continue

    arima_fore = None
    ets_fore = None
    try:
        try:
            arima_model = ARIMA(ser_train, order=(1,1,1)).fit()
            arima_fore_raw = arima_model.forecast(len(ser_val))
            arima_fore = pd.Series(arima_fore_raw.values, index=ser_val.index)
        except Exception as e:
            print(f"  ARIMA failed for {region}: {e}")
            arima_fore = pd.Series(np.nan, index=ser_val.index)

        try:
            ets_model = ExponentialSmoothing(ser_train, seasonal='add', seasonal_periods=12).fit()
            ets_fore_raw  = ets_model.forecast(len(ser_val))
            ets_fore = pd.Series(ets_fore_raw.values, index=ser_val.index)
        except Exception as e:
            print(f"  ETS failed for {region}: {e}")
            ets_fore = pd.Series(np.nan, index=ser_val.index)

        for name, pred in [('ARIMA', arima_fore), ('ETS', ets_fore)]:
             if pred is not None and not pred.isnull().all():
                mae  = mean_absolute_error(ser_val, pred)
                rmse = np.sqrt(mean_squared_error(ser_val, pred))
                safe_ser_val = ser_val.replace(0, np.nan)
                mape = np.mean(np.abs((safe_ser_val - pred) / safe_ser_val)) * 100

                ts_metrics[(region, name)] = {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}
                print(f"  {name} - MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")
             else:
                 print(f"  {name} - Skipped (Prediction failed or all NaN)")
                 ts_metrics[(region, name)] = {'MAE': np.nan, 'RMSE': np.nan, 'MAPE': np.nan}

    except Exception as e:
        print(f"An unexpected error occurred during time series modeling for {region}: {e}")
        if (region, 'ARIMA') not in ts_metrics:
            ts_metrics[(region, 'ARIMA')] = {'MAE': np.nan, 'RMSE': np.nan, 'MAPE': np.nan}
        if (region, 'ETS') not in ts_metrics:
            ts_metrics[(region, 'ETS')] = {'MAE': np.nan, 'RMSE': np.nan, 'MAPE': np.nan}


# 4) Machine-learning models (global across regions)
print("\n--- Preparing Data for ML Models ---")
df_ml = pd.get_dummies(df, columns=['RegionName'], prefix='Region', drop_first=True, dtype=float)

lag_cols = ['ZHVI_lag_1', 'ZHVI_lag_3', 'ZHVI_lag_6']
if not all(col in df.columns for col in lag_cols):
    print(f"Error: One or more lag columns {lag_cols} not found.")
    print("Make sure the CSV contains pre-calculated lag features or calculate them.")
    if not all(col in df.columns for col in lag_cols):
        print("Error: Lag columns still missing after attempting calculation.")
        exit()

if 'month' not in df.columns:
    df['month'] = df['Date'].dt.month
if 'quarter' not in df.columns:
    df['quarter'] = df['Date'].dt.quarter
df_ml = pd.get_dummies(df, columns=['RegionName'], prefix='Region', drop_first=True, dtype=float)

features_time = ['month', 'quarter']
region_dummies = [c for c in df_ml.columns if c.startswith('Region_')]
features_ml = lag_cols + features_time + region_dummies
target = 'ZHVI'

print(f"ML Features: {features_ml}")

train_ml_idx = df_ml['Date'] <= train_end
val_ml_idx   = (df_ml['Date'] > train_end) & (df_ml['Date'] <= val_end)

X_train_ml = df_ml.loc[train_ml_idx, features_ml]
y_train_ml = df_ml.loc[train_ml_idx, target]
X_val_ml   = df_ml.loc[val_ml_idx, features_ml]
y_val_ml   = df_ml.loc[val_ml_idx, target]

train_nan_mask = ~X_train_ml.isna().any(axis=1)
X_train_ml = X_train_ml.loc[train_nan_mask]
y_train_ml = y_train_ml.loc[train_nan_mask]

val_nan_mask   = ~X_val_ml.isna().any(axis=1)
X_val_ml   = X_val_ml.loc[val_nan_mask]
y_val_ml   = y_val_ml.loc[val_nan_mask]

print(f"Shape before scaling: X_train={X_train_ml.shape}, X_val={X_val_ml.shape}")

if X_train_ml.empty or X_val_ml.empty:
    print("Error: No data remaining for ML models after handling NaNs. Check lag generation and date ranges.")
    exit()

print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_ml)
X_val_scaled = scaler.transform(X_val_ml)

X_train_scaled_df = pd.DataFrame(X_train_scaled, index=X_train_ml.index, columns=X_train_ml.columns)
X_val_scaled_df = pd.DataFrame(X_val_scaled, index=X_val_ml.index, columns=X_val_ml.columns)

y_train_ml = y_train_ml.astype(np.float32)
y_val_ml = y_val_ml.astype(np.float32)

X_train_final = X_train_scaled_df.astype(np.float32).values
X_val_final = X_val_scaled_df.astype(np.float32).values
print(f"Final array types: X_train={X_train_final.dtype}, y_train={y_train_ml.dtype}")

tscv = TimeSeriesSplit(n_splits=3)

print("\n--- Training Random Forest ---")
rf = RandomForestRegressor(random_state=42, n_jobs=-1)
param_rf = {'n_estimators': [100, 200], 'max_depth': [5, 10]}
grid_rf = GridSearchCV(rf, param_rf, cv=tscv, scoring='neg_mean_absolute_error', n_jobs=1)

try:
    grid_rf.fit(X_train_ml, y_train_ml)
    print(f"Best RF Params: {grid_rf.best_params_}")
    rf_pred = grid_rf.predict(X_val_ml)
    rf_metrics = {
        'MAE': mean_absolute_error(y_val_ml, rf_pred),
        'RMSE': np.sqrt(mean_squared_error(y_val_ml, rf_pred)),
        'MAPE': np.mean(np.abs((y_val_ml - rf_pred) / y_val_ml.replace(0, np.nan))) * 100
    }
    print(f"RF Val Metrics: {rf_metrics}")
except ValueError as e:
     print(f"Error during RF training/prediction: {e}")
     print("Check data shapes, NaN values, or TimeSeriesSplit compatibility.")
     rf_metrics = {'MAE': np.nan, 'RMSE': np.nan, 'MAPE': np.nan}

print("\n--- Training SVR ---")
svr = SVR()
svr.fit(X_train_scaled_df, y_train_ml)
svr_pred = svr.predict(X_val_scaled_df)
svr_metrics = {
    'MAE': mean_absolute_error(y_val_ml, svr_pred),
    'RMSE': np.sqrt(mean_squared_error(y_val_ml, svr_pred)),
    'MAPE': np.mean(np.abs((y_val_ml - svr_pred) / y_val_ml.replace(0, np.nan))) * 100
}
print(f"SVR Val Metrics (scaled): {svr_metrics}")


# 5) Deep-learning models with PyTorch
dl_metrics = {}
print("\n--- Training Deep Learning Models (PyTorch) ---")

print("Training MLP...")
class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.layer_1 = nn.Linear(input_dim, 64)
        self.relu_1 = nn.ReLU()
        self.layer_2 = nn.Linear(64, 32)
        self.relu_2 = nn.ReLU()
        self.layer_3 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.relu_1(self.layer_1(x))
        x = self.relu_2(self.layer_2(x))
        x = self.layer_3(x)
        return x

X_train_tensor = torch.tensor(X_train_final).to(device)
y_train_tensor = torch.tensor(y_train_ml.values).unsqueeze(1).to(device)
X_val_tensor = torch.tensor(X_val_final).to(device)
y_val_tensor = torch.tensor(y_val_ml.values).unsqueeze(1).to(device)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=len(val_dataset))

mlp_model = MLP(X_train_final.shape[1]).to(device)
criterion_mlp = nn.L1Loss()
optimizer_mlp = optim.Adam(mlp_model.parameters(), lr=0.001)
n_epochs_mlp = 20

for epoch in range(n_epochs_mlp):
    mlp_model.train()
    train_loss = 0.0
    for inputs, targets in train_loader:
        optimizer_mlp.zero_grad()
        outputs = mlp_model(inputs)
        loss = criterion_mlp(outputs, targets)
        loss.backward()
        optimizer_mlp.step()
        train_loss += loss.item() * inputs.size(0)
    train_loss /= len(train_loader.dataset)

    mlp_model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = mlp_model(inputs)
            loss = criterion_mlp(outputs, targets)
            val_loss += loss.item() * inputs.size(0)
    val_loss /= len(val_loader.dataset)

    if (epoch + 1) % 5 == 0:
         print(f'MLP Epoch {epoch+1}/{n_epochs_mlp}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

mlp_model.eval()
with torch.no_grad():
    mlp_pred_tensor = mlp_model(X_val_tensor)
mlp_pred = mlp_pred_tensor.squeeze().cpu().numpy()

dl_metrics['MLP'] = {
    'MAE': mean_absolute_error(y_val_ml, mlp_pred),
    'RMSE': np.sqrt(mean_squared_error(y_val_ml, mlp_pred)),
    'MAPE': np.mean(np.abs((y_val_ml - mlp_pred) / y_val_ml.replace(0, np.nan))) * 100
}
print(f"MLP Val Metrics: {dl_metrics['MLP']}")

print("\nTraining LSTM...")
def make_lstm_inputs_torch(df_subset, feature_cols):
    if not all(col in df_subset.columns for col in feature_cols):
         print(f"Error: LSTM features {feature_cols} not found in subset.")
         return None, None

    sequences = df_subset[feature_cols].values
    targets = df_subset['ZHVI'].values

    nan_mask = ~np.isnan(sequences).any(axis=1)
    sequences = sequences[nan_mask]
    targets = targets[nan_mask]

    if sequences.size == 0:
        print("Warning: No valid sequences remain after handling NaNs for LSTM.")
        return None, None

    sequences_reshaped = sequences.reshape(sequences.shape[0], sequences.shape[1], 1)
    targets = targets.astype(np.float32)

    return torch.tensor(sequences_reshaped, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32).unsqueeze(1)

lstm_feature_cols = ['ZHVI_lag_6', 'ZHVI_lag_3', 'ZHVI_lag_1']
X_train_lstm_tensor, y_train_lstm_tensor = make_lstm_inputs_torch(train_df, lstm_feature_cols)
X_val_lstm_tensor, y_val_lstm_tensor = make_lstm_inputs_torch(val_df, lstm_feature_cols)

if X_train_lstm_tensor is None or X_val_lstm_tensor is None:
    print("Skipping LSTM due to data preparation issues (likely NaNs or missing columns).")
    dl_metrics['LSTM'] = {'MAE': np.nan, 'RMSE': np.nan, 'MAPE': np.nan}
else:
    print(f"LSTM Train shapes: X={X_train_lstm_tensor.shape}, y={y_train_lstm_tensor.shape}")
    print(f"LSTM Val shapes: X={X_val_lstm_tensor.shape}, y={y_val_lstm_tensor.shape}")

    X_train_lstm_tensor = X_train_lstm_tensor.to(device)
    y_train_lstm_tensor = y_train_lstm_tensor.to(device)
    X_val_lstm_tensor = X_val_lstm_tensor.to(device)
    y_val_lstm_tensor = y_val_lstm_tensor.to(device)

    train_lstm_dataset = TensorDataset(X_train_lstm_tensor, y_train_lstm_tensor)
    val_lstm_dataset = TensorDataset(X_val_lstm_tensor, y_val_lstm_tensor)

    train_lstm_loader = DataLoader(train_lstm_dataset, batch_size=16, shuffle=True)
    val_lstm_loader = DataLoader(val_lstm_dataset, batch_size=len(val_lstm_dataset))

    class LSTMModel(nn.Module):
         def __init__(self, input_size, hidden_size, num_layers=1):
            super(LSTMModel, self).__init__()
            self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
            self.linear = nn.Linear(hidden_size, 1)
         def forward(self, x):
            lstm_out, _ = self.lstm(x)
            last_time_step_out = lstm_out[:, -1, :]
            out = self.linear(last_time_step_out)
            return out

    lstm_model = LSTMModel(input_size=1, hidden_size=50).to(device)
    criterion_lstm = nn.L1Loss()
    optimizer_lstm = optim.Adam(lstm_model.parameters(), lr=0.001)
    n_epochs_lstm = 20

    for epoch in range(n_epochs_lstm):
        lstm_model.train()
        train_loss = 0.0
        for inputs, targets in train_lstm_loader:
            optimizer_lstm.zero_grad()
            outputs = lstm_model(inputs)
            loss = criterion_lstm(outputs, targets)
            loss.backward()
            optimizer_lstm.step()
            train_loss += loss.item() * inputs.size(0)
        train_loss /= len(train_lstm_loader.dataset)

        lstm_model.eval()
        val_loss = 0.0
        with torch.no_grad():
             for inputs, targets in val_lstm_loader:
                outputs = lstm_model(inputs)
                loss = criterion_lstm(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
        val_loss /= len(val_lstm_loader.dataset)

        if (epoch + 1) % 5 == 0:
            print(f'LSTM Epoch {epoch+1}/{n_epochs_lstm}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    lstm_model.eval()
    with torch.no_grad():
        lstm_pred_tensor = lstm_model(X_val_lstm_tensor)
    lstm_pred = lstm_pred_tensor.squeeze().cpu().numpy()
    y_val_lstm_np = y_val_lstm_tensor.squeeze().cpu().numpy()

    dl_metrics['LSTM'] = {
        'MAE': mean_absolute_error(y_val_lstm_np, lstm_pred),
        'RMSE': np.sqrt(mean_squared_error(y_val_lstm_np, lstm_pred)),
        'MAPE': np.mean(np.abs((y_val_lstm_np - lstm_pred) / np.where(y_val_lstm_np == 0, np.nan, y_val_lstm_np))) * 100
    }
    print(f"LSTM Val Metrics: {dl_metrics['LSTM']}")

# 6) Aggregate and display all results
results_list = []

for (region, model_name), metrics in ts_metrics.items():
    results_list.append({'Group': 'TS Baseline', 'Model': f"{model_name}_{region}", **metrics})

results_list.append({'Group': 'ML Global', 'Model': 'Random Forest', **rf_metrics})
results_list.append({'Group': 'ML Global', 'Model': 'SVR (Scaled)', **svr_metrics})

for model_name, metrics in dl_metrics.items():
    results_list.append({'Group': 'DL Global', 'Model': model_name, **metrics})

results_df = pd.DataFrame(results_list)
numeric_cols = ['MAE', 'RMSE', 'MAPE']
for col in numeric_cols:
    if col in results_df.columns:
        results_df[col] = pd.to_numeric(results_df[col], errors='coerce')

pd.set_option('display.float_format', '{:.2f}'.format)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)

print("\n\n--- Final Results ---")
print(results_df.fillna('N/A').to_string(index=False))

# 7) Evaluation & Visualization
print("\n--- Step 7: Evaluation & Visualization ---")

# 7a) Evaluate on Test Set
print("Evaluating models on the Test Set...")
test_metrics_list = []

test_ml_idx = df_ml['Date'] > val_end
X_test_ml = df_ml.loc[test_ml_idx, features_ml]
y_test_ml = df_ml.loc[test_ml_idx, target]

test_nan_mask = ~X_test_ml.isna().any(axis=1)
X_test_ml = X_test_ml.loc[test_nan_mask]
y_test_ml = y_test_ml.loc[test_nan_mask]

if not X_test_ml.empty:
    X_test_scaled = scaler.transform(X_test_ml)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, index=X_test_ml.index, columns=X_test_ml.columns)
    X_test_final = X_test_scaled_df.astype(np.float32).values
    y_test_ml = y_test_ml.astype(np.float32)

    rf_test_pred = grid_rf.predict(X_test_ml)
    test_metrics_list.append({
        'Model': 'Random Forest',
        'MAE_test': mean_absolute_error(y_test_ml, rf_test_pred),
        'RMSE_test': np.sqrt(mean_squared_error(y_test_ml, rf_test_pred)),
        'MAPE_test': np.mean(np.abs((y_test_ml - rf_test_pred) / y_test_ml.replace(0, np.nan))) * 100
    })

    svr_test_pred = svr.predict(X_test_scaled_df)
    test_metrics_list.append({
        'Model': 'SVR (Scaled)',
        'MAE_test': mean_absolute_error(y_test_ml, svr_test_pred),
        'RMSE_test': np.sqrt(mean_squared_error(y_test_ml, svr_test_pred)),
        'MAPE_test': np.mean(np.abs((y_test_ml - svr_test_pred) / y_test_ml.replace(0, np.nan))) * 100
    })

    X_test_tensor = torch.tensor(X_test_final).to(device)
    mlp_model.eval()
    with torch.no_grad():
        mlp_test_pred_tensor = mlp_model(X_test_tensor)
    mlp_test_pred = mlp_test_pred_tensor.squeeze().cpu().numpy()
    test_metrics_list.append({
        'Model': 'MLP',
        'MAE_test': mean_absolute_error(y_test_ml, mlp_test_pred),
        'RMSE_test': np.sqrt(mean_squared_error(y_test_ml, mlp_test_pred)),
        'MAPE_test': np.mean(np.abs((y_test_ml - mlp_test_pred) / y_test_ml.replace(0, np.nan))) * 100
    })

    X_test_lstm_tensor, y_test_lstm_tensor = make_lstm_inputs_torch(test_df, lstm_feature_cols)
    if X_test_lstm_tensor is not None and not X_test_lstm_tensor.nelement() == 0:
         X_test_lstm_tensor = X_test_lstm_tensor.to(device)
         y_test_lstm_np = y_test_lstm_tensor.squeeze().cpu().numpy()

         lstm_model.eval()
         with torch.no_grad():
             lstm_test_pred_tensor = lstm_model(X_test_lstm_tensor)
         lstm_test_pred = lstm_test_pred_tensor.squeeze().cpu().numpy()

         test_metrics_list.append({
            'Model': 'LSTM',
            'MAE_test': mean_absolute_error(y_test_lstm_np, lstm_test_pred),
            'RMSE_test': np.sqrt(mean_squared_error(y_test_lstm_np, lstm_test_pred)),
            'MAPE_test': np.mean(np.abs((y_test_lstm_np - lstm_test_pred) / np.where(y_test_lstm_np == 0, np.nan, y_test_lstm_np))) * 100
         })
    else:
         print("Skipping LSTM test evaluation due to lack of valid test sequences.")
         test_metrics_list.append({'Model': 'LSTM', 'MAE_test': np.nan, 'RMSE_test': np.nan, 'MAPE_test': np.nan})

    test_metrics_df = pd.DataFrame(test_metrics_list)
    results_df = pd.merge(results_df, test_metrics_df, on='Model', how='left')

else:
    print("Skipping test set evaluation as no test data remained after processing.")
    results_df['MAE_test'] = np.nan
    results_df['RMSE_test'] = np.nan
    results_df['MAPE_test'] = np.nan

# 7b) Plot Predicted vs. Actual (Validation Set)
print("Generating validation plots...")

val_dates = df_ml.loc[y_val_ml.index, 'Date']

plt.figure(figsize=(15, 12))
plt.suptitle('Predicted vs. Actual ZHVI (Validation Set: 2021-2022)', fontsize=16)

plt.subplot(2, 2, 1)
plt.plot(val_dates, y_val_ml, label='Actual', marker='.', linestyle='-', alpha=0.7)
plt.plot(val_dates, rf_pred, label='RF Predicted', marker='x', linestyle='--', alpha=0.7)
plt.title(f'Random Forest (MAPE: {rf_metrics["MAPE"]:.2f}%)')
plt.ylabel('ZHVI')
plt.legend()
plt.xticks(rotation=45)

plt.subplot(2, 2, 2)
plt.plot(val_dates, y_val_ml, label='Actual', marker='.', linestyle='-', alpha=0.7)
plt.plot(val_dates, svr_pred, label='SVR Predicted', marker='x', linestyle='--', alpha=0.7)
plt.title(f'SVR Scaled (MAPE: {svr_metrics["MAPE"]:.2f}%)')
plt.ylabel('ZHVI')
plt.legend()
plt.xticks(rotation=45)

plt.subplot(2, 2, 3)
plt.plot(val_dates, y_val_ml, label='Actual', marker='.', linestyle='-', alpha=0.7)
plt.plot(val_dates, mlp_pred, label='MLP Predicted', marker='x', linestyle='--', alpha=0.7)
plt.title(f'MLP (MAPE: {dl_metrics["MLP"]["MAPE"]:.2f}%)')
plt.ylabel('ZHVI')
plt.legend()
plt.xticks(rotation=45)

if 'LSTM' in dl_metrics and isinstance(dl_metrics['LSTM'], dict) and not np.isnan(dl_metrics['LSTM']['MAPE']):
     _, y_val_lstm_tensor_plot = make_lstm_inputs_torch(val_df, lstm_feature_cols)
     lstm_val_mask = ~val_df[lstm_feature_cols].isna().any(axis=1)
     lstm_val_dates = val_df.loc[lstm_val_mask, 'Date']

     if len(lstm_val_dates) == len(lstm_pred):
         plt.subplot(2, 2, 4)
         plt.plot(lstm_val_dates, y_val_lstm_np, label='Actual', marker='.', linestyle='-', alpha=0.7)
         plt.plot(lstm_val_dates, lstm_pred, label='LSTM Predicted', marker='x', linestyle='--', alpha=0.7)
         plt.title(f'LSTM (MAPE: {dl_metrics["LSTM"]["MAPE"]:.2f}%)')
         plt.ylabel('ZHVI')
         plt.legend()
         plt.xticks(rotation=45)
     else:
         print("Warning: LSTM plot skipped due to index misalignment.")
         plt.subplot(2, 2, 4)
         plt.text(0.5, 0.5, 'LSTM Index Mismatch', ha='center', va='center')
         plt.title('LSTM')

else:
     plt.subplot(2, 2, 4)
     plt.text(0.5, 0.5, 'LSTM Skipped', horizontalalignment='center', verticalalignment='center')
     plt.title('LSTM')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

print("Generating validation plots for baseline models (per region)...")
n_regions = len(top5)
plt.figure(figsize=(15, 4 * n_regions))
plt.suptitle('Baseline Model Predictions vs. Actual (Validation Set)', fontsize=16)
plot_num = 1
for region in top5:
     ser_val_region = val_df[val_df['RegionName']==region].set_index('Date')['ZHVI']
     if ser_val_region.empty:
         continue

     arima_metrics = ts_metrics.get((region, 'ARIMA'), {})
     ets_metrics = ts_metrics.get((region, 'ETS'), {})

     try:
          ser_train_region = train_df[train_df['RegionName']==region].set_index('Date')['ZHVI']
          arima_model_plot = ARIMA(ser_train_region, order=(1,1,1)).fit()
          arima_fore_plot_raw = arima_model_plot.forecast(len(ser_val_region))
          arima_fore_plot = pd.Series(arima_fore_plot_raw.values, index=ser_val_region.index)

          ets_model_plot = ExponentialSmoothing(ser_train_region, seasonal='add', seasonal_periods=12).fit()
          ets_fore_plot_raw = ets_model_plot.forecast(len(ser_val_region))
          ets_fore_plot = pd.Series(ets_fore_plot_raw.values, index=ser_val_region.index)
     except Exception as e:
          print(f"  Warning: Could not re-generate baseline forecasts for plotting {region}: {e}")
          arima_fore_plot = pd.Series(np.nan, index=ser_val_region.index)
          ets_fore_plot = pd.Series(np.nan, index=ser_val_region.index)

     plt.subplot(n_regions, 1, plot_num)
     plt.plot(ser_val_region.index, ser_val_region, label='Actual', marker='.', linestyle='-')
     if not arima_fore_plot.isnull().all():
        plt.plot(ser_val_region.index, arima_fore_plot, label=f'ARIMA (MAPE: {arima_metrics.get("MAPE", np.nan):.2f}%)', marker='x', linestyle='--')
     if not ets_fore_plot.isnull().all():
        plt.plot(ser_val_region.index, ets_fore_plot, label=f'ETS (MAPE: {ets_metrics.get("MAPE", np.nan):.2f}%)', marker='^', linestyle=':')
     plt.title(f'Region: {region}')
     plt.ylabel('ZHVI')
     plt.legend()
     plt.xticks(rotation=45)
     plot_num += 1

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# 7c) Summarize Performance
print("Performance summary table generated (see final output).")

# 8) Forecasting Future Periods
print("\n--- Step 8: Forecasting Future Periods ---")
# 8a) Pick best model
best_model_name = 'Random Forest'
print(f"Selected best model based on validation performance: {best_model_name}")

# 8b) Retrain the best model on combined Train + Validation data
if best_model_name == 'Random Forest':
    print("Retraining Random Forest on Train+Validation data...")
    combined_ml_idx = df_ml['Date'] <= val_end
    X_combined_ml = df_ml.loc[combined_ml_idx, features_ml]
    y_combined_ml = df_ml.loc[combined_ml_idx, target]

    combined_nan_mask = ~X_combined_ml.isna().any(axis=1)
    X_combined_ml = X_combined_ml.loc[combined_nan_mask]
    y_combined_ml = y_combined_ml.loc[combined_nan_mask]
    y_combined_ml = y_combined_ml.astype(np.float32)

    best_rf_params = grid_rf.best_params_
    final_rf_model = RandomForestRegressor(**best_rf_params, random_state=42, n_jobs=-1)
    final_rf_model.fit(X_combined_ml, y_combined_ml)
    print("Final RF model retrained.")

    # 8c) Generate 12-month ahead forecasts
    forecast_horizon = 12
    last_date = df['Date'].max()
    forecast_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=forecast_horizon, freq='ME')

    forecasts_all_regions = {}

    print(f"Generating {forecast_horizon}-month forecasts for regions: {top5}...")

    latest_known_data = df[df['Date'] <= val_end].copy()

    for region in top5:
        print(f"  Forecasting for {region}...")
        region_forecasts = []
        history_df = latest_known_data[latest_known_data['RegionName'] == region].sort_values('Date').copy()

        if len(history_df) < 6:
             print(f"    Skipping {region}: Insufficient history ({len(history_df)} points) for lags.")
             forecasts_all_regions[region] = pd.Series(np.nan, index=forecast_dates)
             continue

        current_lags = {
            'ZHVI_lag_1': history_df['ZHVI'].iloc[-1],
            'ZHVI_lag_3': history_df['ZHVI'].iloc[-3],
            'ZHVI_lag_6': history_df['ZHVI'].iloc[-6]
        }

        for f_date in forecast_dates:
            features_step = {}
            features_step.update(current_lags)
            features_step['month'] = f_date.month
            features_step['quarter'] = f_date.quarter
            for dummy_col in region_dummies:
                dummy_region = dummy_col.replace('Region_', '')
                features_step[dummy_col] = 1.0 if dummy_region == region else 0.0

            X_step = pd.DataFrame([features_step], columns=features_ml)
            X_step = X_step.astype(float)

            try:
                 forecast_value = final_rf_model.predict(X_step)[0]
            except Exception as e:
                 print(f"    Error predicting step for {region} on {f_date}: {e}")
                 forecast_value = np.nan

            region_forecasts.append(forecast_value)

            new_lag_1 = forecast_value
            new_lag_3 = current_lags['ZHVI_lag_1']
            new_lag_6 = current_lags['ZHVI_lag_3']

            current_lags['ZHVI_lag_6'] = new_lag_6
            current_lags['ZHVI_lag_3'] = new_lag_3
            current_lags['ZHVI_lag_1'] = new_lag_1

        forecasts_all_regions[region] = pd.Series(region_forecasts, index=forecast_dates)

    # 8d) Visualize forecast bands vs. history
    print("Generating forecast plots...")
    plt.figure(figsize=(15, 5 * n_regions))
    plt.suptitle(f'{forecast_horizon}-Month ZHVI Forecast using {best_model_name}', fontsize=16)
    plot_num = 1
    for region in top5:
         plt.subplot(n_regions, 1, plot_num)
         history_plot = df[(df['RegionName'] == region) & (df['Date'] <= val_end)].set_index('Date')['ZHVI']
         plt.plot(history_plot.index, history_plot, label='Historical ZHVI', color='blue')

         forecast_plot = forecasts_all_regions.get(region)
         if forecast_plot is not None and not forecast_plot.isnull().all():
             plt.plot(forecast_plot.index, forecast_plot, label='Forecast', color='red', linestyle='--')
         else:
              if forecast_plot is not None:
                  plt.text(history_plot.index[-1], history_plot.iloc[-1]*0.95 , 'Forecast Failed/Skipped', color='grey')

         plt.title(f'Region: {region}')
         plt.ylabel('ZHVI')
         plt.legend()
         plt.xticks(rotation=45)
         plot_num += 1

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

else:
    print(f"Forecasting logic not implemented for model: {best_model_name}")

print("\n\n--- Final Results (Including Test Metrics) ---")
numeric_cols = ['MAE', 'RMSE', 'MAPE', 'MAE_test', 'RMSE_test', 'MAPE_test']
for col in numeric_cols:
    if col in results_df.columns:
        results_df[col] = pd.to_numeric(results_df[col], errors='coerce')

pd.set_option('display.float_format', '{:.2f}'.format)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)

print(results_df.fillna('N/A').to_string(index=False))