import pandas as pd
from analysis.profiler import profile_data


df = pd.read_csv("data/Order_delivery.csv")
profile = profile_data(df)
print("Data Description")
print("\nTotal Rows")
print(profile["Total Rows"])
print("\nTotal Columns")
print(profile["Total Columns"])
print("\nColumn Types")
print(profile["Column Types"])
print("\nTop 5 Rows")
print(profile["Top 5 Rows"])
print("\nLast 5 Rows")
print(profile["Last 5 Rows"])
print("\nDescription")
print(profile["Description"])
print("\nColumn Names: ")
print(profile["Column Names"])
print("\nMissing Values")
print(profile["Missing Values"])