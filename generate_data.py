import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# Generate date range
date_range = pd.date_range(start='2023-01-01', periods=180, freq='D')

# Generate features
price = np.random.randint(100, 300, size=len(date_range))
discount = np.random.randint(0, 50, size=len(date_range))
month = date_range.month

# Generate sales using a formula + noise
sales = (
    (500 - price) + 
    (2 * discount) + 
    (month * 3) +
    np.random.normal(0, 10, size=len(date_range))
).astype(int)

# Create DataFrame
df = pd.DataFrame({
    'Date': date_range,
    'Sales': sales,
    'Price': price,
    'Discount': discount,
    'Month': month
})

# Save to CSV (optional)
df.to_csv('sales_dataset.csv', index=False)

print(df.head())
