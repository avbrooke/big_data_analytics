import arff
import pandas as pd
import csv

# Load the ARFF file as raw lines
with open('gcse_cleaned.arff', 'r') as f:
    lines = f.readlines()

# Separate header and data
data_start = lines.index("@data\n")
header = lines[:data_start]
data = lines[data_start + 1:]

# Write a cleaned CSV version
with open('gcse_data.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    for line in data:
        # Remove surrounding quotes and split cleanly
        cleaned = [entry.strip(" '\"\n") for entry in line.split(",")]
        writer.writerow(cleaned)

print("Cleaned CSV written to gcse_data.csv")

# If you know the actual column names, define them here:
column_names = [
    "firstname", "lastname", "gender", "dob", "auth",
    "subject1", "grade1", "subject2", "grade2",
    "subject3", "grade3", "subject4", "grade4",
    "subject5", "grade5"
]

df = pd.read_csv("gcse_data.csv", names=column_names)

# Replace '?' with NaN
df.replace('?', pd.NA, inplace=True)

# Show clean output
print(df.head())
print(df.info())