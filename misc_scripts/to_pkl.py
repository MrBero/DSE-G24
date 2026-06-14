import pandas as pd

f = pd.read_csv('inputs/CSV_WIND_TURBINE_5X.csv', skipinitialspace=True)
f.to_pickle('inputs/CSV_WIND_TURBINE_5X.pkl')
# print(f.columns)