"""Weekly Orders Report
For each week, find the total number of orders. Include only the orders that are from the first quarter of 2023.
The output should contain 'week' and 'quantity'."""

# Import necessary libraries
import pandas as pd

# Sample data for demonstration
data = {
    'week': ['2022-12-30', '2023-01-02', '2023-01-06', '2023-02-15', '2023-03-20'],
    'quantity': [10, 15, 20, 5, 30]
}

# Create DataFrame
orders_analysis = pd.DataFrame(data)

# Convert 'week' column to datetime
orders_analysis['week'] = pd.to_datetime(orders_analysis['week'])

# Filter for first quarter of 2023
orders_q1_2023 = orders_analysis[(orders_analysis['week'] >= '2023-01-01') & (orders_analysis['week'] <= '2023-03-31')]

# Group by 'week' and sum 'quantity'
result = orders_q1_2023.groupby('week')['quantity'].sum().reset_index()

# Rename columns
result.columns = ['week', 'quantity']

print(result)
