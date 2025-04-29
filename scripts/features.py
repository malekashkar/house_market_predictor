import pandas as pd

df = pd.read_csv('../data/cleanedata.csv', parse_dates=['Date'])
df = df.sort_values(['RegionID', 'Date'])

for lag in [1, 3, 6]:
    df[f'ZHVI_lag_{lag}'] = df.groupby('RegionID')['ZHVI'].shift(lag)

df['month']   = df['Date'].dt.month
df['quarter'] = df['Date'].dt.quarter
df['year']    = df['Date'].dt.year

df.to_csv('../data/features.csv', index=False)