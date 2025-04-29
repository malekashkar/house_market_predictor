import pandas as pd

raw_df = pd.read_csv('/data/original.csv')
id_vars = ['RegionID', 'SizeRank', 'RegionName', 'RegionType', 'StateName']

df_long = raw_df.melt(
    id_vars=id_vars,
    var_name='Date',
    value_name='ZHVI'
)

df_long['Date'] = pd.to_datetime(df_long['Date'])
df_long['ZHVI']  = pd.to_numeric(df_long['ZHVI'], errors='coerce')

df_clean = (
    df_long
    .dropna(subset=['ZHVI'])
    .sort_values(['RegionID', 'Date'])
    .reset_index(drop=True)
)

df_clean.to_csv('/data/cleanedata.csv', index=False)