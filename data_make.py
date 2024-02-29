import numpy as np
import pandas as pd

# Read CSV files
df = pd.read_csv(r"C:\\Users\\hyowo\\OneDrive\\Documents\\programming\\2024 winter\\CxC hackathon\\wild_fire_cleaned.csv")
test = pd.read_csv(r"C:\\Users\\hyowo\\OneDrive\\Documents\\programming\\2024 winter\\CxC hackathon\\wild_fire_mean.csv")

# Define columns to get and their squared versions
columns_to_get = ['current_size', 'fire_spread_rate', 'temperature', 'relative_humidity', 'wind_speed', 'fire_fighting_start_size', 'duration_from_reported_to_dispatch']
columns_to_get_sqr = ['current_size_sqr', 'fire_spread_rate_sqr', 'temperature_sqr', 'relative_humidity_sqr', 'wind_speed_sqr', 'fire_fighting_start_size_sqr', 'duration_from_reported_to_dispatch_sqr']

# Create a new DataFrame with selected columns from 'test' DataFrame
new_df = pd.DataFrame()
new_df[columns_to_get] = test[columns_to_get].copy()

# Convert 'duration_from_reported_to_dispatch' to seconds
new_df['duration_from_reported_to_dispatch'] = pd.to_timedelta(new_df['duration_from_reported_to_dispatch']).dt.total_seconds() / 60

# Convert DataFrame to NumPy array
test_array = new_df.to_numpy()

# Compute squared values
test_array_squared = np.square(test_array[:, 1:])

# Create a structured array with squared values and named columns
test_array2 = np.zeros(test_array_squared.shape, dtype=[(name, float) for name in columns_to_get_sqr])

# Assign squared values to the respective columns
for idx, name in enumerate(columns_to_get_sqr):
    test_array2[name] = test_array_squared[:, idx]

# Concatenate original and squared arrays
test_array = np.c_[test_array, test_array2]

# Take the natural logarithm of 'current_size' in 'df'
df['current_size'] = np.log(df['current_size'])

# Save the processed data to a new CSV file
np.savetxt("wild_fire_data_v2.csv", test_array, delimiter=",")
