

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# NOTE: Load data
sell_recs = pd.read_csv("sell_recommendations.csv")

price_gaps = sell_recs["price_gap_usd"]

# NOTE: Plot
fig, ax = plt.subplots(figsize=(10, 7))

ax.hist(price_gaps, bins=20, color="#1f77b4", edgecolor="#1f77b4")

ax.set_xlabel("Price Gap (USD)", fontsize=12)
ax.set_ylabel("Frequency", fontsize=12)
ax.set_title("Price Gap Distribution (Arbitrage Opportunities)", fontsize=14)

ax.set_xticks(np.arange(-10, 35, 5))

plt.tight_layout()
plt.show()