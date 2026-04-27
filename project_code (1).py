import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns, os
from sklearn.ensemble import IsolationForest

os.makedirs('/tmp/dmt/out', exist_ok=True)
np.random.seed(42)

# 1) Generate synthetic banking transaction dataset
n = 1000
locations = ['Chennai','Mumbai','Delhi','Bangalore','Kolkata','Coimbatore','Hyderabad']
types = ['POS','ATM','Online','Transfer']
normal = pd.DataFrame({
    'TransactionID': range(1, n+1),
    'Amount': np.random.gamma(2, 1500, n).round(2),
    'Hour': np.random.choice(range(7,22), n),
    'Location': np.random.choice(locations, n),
    'Type': np.random.choice(types, n, p=[0.4,0.25,0.25,0.10]),
})
# Inject ~30 fraudulent / unusual transactions
fraud_n = 30
fraud = pd.DataFrame({
    'TransactionID': range(n+1, n+fraud_n+1),
    'Amount': np.random.uniform(50000, 200000, fraud_n).round(2),
    'Hour': np.random.choice([0,1,2,3,4], fraud_n),
    'Location': np.random.choice(['Unknown','Foreign','Lagos','Moscow'], fraud_n),
    'Type': np.random.choice(['Online','Transfer'], fraud_n),
})
df = pd.concat([normal, fraud], ignore_index=True).sample(frac=1, random_state=1).reset_index(drop=True)
df.to_csv('/tmp/dmt/out/transactions.csv', index=False)

# 2) Outlier detection using Isolation Forest on Amount + Hour
X = df[['Amount','Hour']].values
model = IsolationForest(contamination=0.03, random_state=42)
df['Anomaly'] = model.fit_predict(X)
df['Status'] = np.where(df['Anomaly']==-1, 'Suspicious', 'Normal')

flagged = df[df['Status']=='Suspicious']
flagged.to_csv('/tmp/dmt/out/suspicious.csv', index=False)
print(f"Total: {len(df)}, Flagged suspicious: {len(flagged)}")

sns.set_style('whitegrid')

# Chart 1: Scatter Amount vs Hour
plt.figure(figsize=(8,5))
colors = df['Status'].map({'Normal':'#3b82f6','Suspicious':'#ef4444'})
plt.scatter(df['Hour'], df['Amount'], c=colors, alpha=0.6, edgecolors='white', s=40)
plt.xlabel('Transaction Hour (0-23)'); plt.ylabel('Amount (₹)')
plt.title('Transaction Amount vs Hour — Suspicious in Red')
plt.tight_layout(); plt.savefig('/tmp/dmt/out/chart_scatter.png', dpi=130); plt.close()

# Chart 2: Amount distribution
plt.figure(figsize=(8,5))
sns.histplot(df[df['Amount']<60000]['Amount'], bins=40, color='#3b82f6', label='Normal range')
plt.axvline(df['Amount'].quantile(0.99), color='#ef4444', linestyle='--', label='99th percentile')
plt.title('Distribution of Transaction Amounts'); plt.xlabel('Amount (₹)'); plt.legend()
plt.tight_layout(); plt.savefig('/tmp/dmt/out/chart_dist.png', dpi=130); plt.close()

# Chart 3: Suspicious by location
plt.figure(figsize=(8,5))
flagged['Location'].value_counts().plot(kind='bar', color='#ef4444')
plt.title('Suspicious Transactions by Location'); plt.ylabel('Count'); plt.xticks(rotation=30)
plt.tight_layout(); plt.savefig('/tmp/dmt/out/chart_loc.png', dpi=130); plt.close()

print("Charts saved.")
