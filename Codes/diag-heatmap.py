import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_path = 'mydata.csv'
df = pd.read_csv(file_path)
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

correlation_matrix = df.corr()

plt.figure(figsize=(16, 12))
sns.heatmap(correlation_matrix, cmap='YlGnBu', annot=True, fmt='.2f', cbar=True)
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.show()