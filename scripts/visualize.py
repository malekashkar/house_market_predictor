import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('/data/cleanedata.csv', parse_dates=['Date'])

top5_regions = (
    df[df['SizeRank'] > 0]
    .drop_duplicates('RegionName')
    .sort_values('SizeRank')['RegionName']
    .tolist()[:5]
)

states = df[df['RegionName'].isin(top5_regions)]['StateName'].unique().tolist()

# 1) Line chart: ZHVI over time for top 5 metros
plt.figure(figsize=(12, 6))
for region in top5_regions:
    subset = df[df['RegionName'] == region]
    plt.plot(subset['Date'], subset['ZHVI'], label=region)
plt.legend()
plt.xlabel('Date')
plt.ylabel('ZHVI')
plt.title('ZHVI Over Time for Top 5 Metro Regions')
plt.tight_layout()
plt.show()

# 2) Box plot: distribution of ZHVI for top 5 regions
plt.figure(figsize=(10, 6))
data_regions = [df[df['RegionName'] == r]['ZHVI'].dropna() for r in top5_regions]
plt.boxplot(data_regions, label=top5_regions, notch=True)
plt.xticks(rotation=45)
plt.ylabel('ZHVI')
plt.title('ZHVI Distribution for Top 5 Metro Regions')
plt.tight_layout()
plt.show()

# 3) Box plot: distribution of ZHVI by state for those top metros
plt.figure(figsize=(10, 6))
data_states = [df[df['StateName'] == s]['ZHVI'].dropna() for s in states]
plt.boxplot(data_states, label=states, notch=True)
plt.xticks(rotation=45)
plt.ylabel('ZHVI')
plt.title('ZHVI Distribution by State (Top 5 Metros)')
plt.tight_layout()
plt.show()
