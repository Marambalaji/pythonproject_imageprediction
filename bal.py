import pandas as pd

# Read the CSV file
data = pd.read_csv("D:\data1.csv")

# Display the contents of the CSV file
print(data)

# Optional: Display specific columns
print("\nAQ and USS:")
print(data[['AQ', 'USS']])
