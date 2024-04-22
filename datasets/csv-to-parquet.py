import pandas as pd
df = pd.read_csv('dataset_train.csv')
df.to_parquet('dataset_train.parquet')

df = pd.read_csv('dataset_test.csv')
df.to_parquet('dataset_test.parquet')