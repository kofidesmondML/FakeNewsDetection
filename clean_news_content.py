import pandas as pd
file_path='./data/NewContent.csv'

df=pd.read_csv(file_path)
print(df.head(10))