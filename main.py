import pandas as pd

db = pd.read_excel("base-mkt.xlsx")

print(f'shape: {db.shape}\n head: {db.head()}')
