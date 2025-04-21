import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# Generate test date range (next 30 days after training period)
test_date_range = pd.date_range(start='2023-07-01', periods=30, freq='D')

# Generate features
test_price = np.random.randint(100, 300, size=len(test_date_range))
test_discount = np.random.randint(0, 50, size=len(test_date_range))
test_month = test_date_range.month

# Generate sales using synthetic formula + noise
test_sales = (
    (500 - test_price) +
    (2 * test_discount) +
    (test_month * 3) +
    np.random.normal(0, 10, size=len(test_date_range))
).astype(int)

# Create the test DataFrame
test_df = pd.DataFrame({
    'Date': test_date_range,
    'Sales': test_sales,
    'Price': test_price,
    'Discount': test_discount,
    'Month': test_month
})

# Save to CSV (optional)
test_df.to_csv('sales_test_dataset.csv', index=False)

# Display first few rows
print("Test Dataset:")
print(test_df.head())
