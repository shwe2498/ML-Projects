import pandas as pd
import datetime

def calculate_active_user_penetration_rate(data):
    # Create DataFrame
    penetration_analysis = pd.DataFrame(data)

    # Convert last_active_date to datetime
    penetration_analysis['last_active_date'] = pd.to_datetime(penetration_analysis['last_active_date'])

    # Set the current date to 2024-01-31
    current_date = datetime.datetime(2024, 1, 31)

    # Calculate the date 30 days prior
    date_30_days_prior = current_date - datetime.timedelta(days=30)

    # Create a new column to mark active users
    penetration_analysis['is_active'] = (
        (penetration_analysis['last_active_date'] >= date_30_days_prior) &
        (penetration_analysis['sessions'] >= 5) &
        (penetration_analysis['listening_hours'] >= 10)
    )

    # Group by country and count total and active users
    grouped = penetration_analysis.groupby('country').agg(
        total_users=('user_id', 'count'),
        active_users=('is_active', 'sum')
    ).reset_index()

    # Compute the penetration rate for each country
    grouped['active_user_penetration_rate'] = (grouped['active_users'] / grouped['total_users'] * 100).round(2)

    # Format the result with relevant columns
    result = grouped[['country', 'active_user_penetration_rate']]

    return result

# Sample data
data = {
    'user_id': [1, 2, 3, 4, 5],
    'country': ['USA', 'USA', 'Japan', 'USA', 'USA'],
    'last_active_date': ['2024-01-29 00:00:00', '2024-01-21 00:00:00', '2024-01-28 00:00:00', '2024-01-04 00:00:00', '2024-01-04 00:00:00'],
    'listening_hours': [87, 62, 39, 16, 73],
    'sessions': [29, 20, 11, 8, 23]
}

# Calculate the active user penetration rate
result = calculate_active_user_penetration_rate(data)
print(result)

