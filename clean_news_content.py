import pandas as pd
from datetime import datetime
import ast

file_path = './data/NewsContent.csv'

df = pd.read_csv(file_path)
print(df.head(10))

def clean_publish_date(df):
    def convert_date(date_string):
        try:
            if pd.isna(date_string):
                return None
            date_dict = ast.literal_eval(date_string)
            date = datetime.fromtimestamp(date_dict['$date'] / 1000)
            return date.strftime("%m-%d-%Y")
        except Exception as e:
            print(f"Error processing date: {e} with entry: {date_string}")
            return None
    
    df['publish_date'] = df['publish_date'].apply(convert_date)
    return df

df = clean_publish_date(df)

def process_authors(df):
    def convert_authors(row):
        try:
            if pd.isna(row['authors']) or row['authors'] == '':
                return None
            list_items = ast.literal_eval(row['authors'])
            return ', '.join(list_items)
        except Exception as e:
            print(f"Error processing authors: {e}")
            return None

    df['authors'] = df.apply(convert_authors, axis=1)
    return df

df = process_authors(df)

def extract_summary(df):
    def get_summary(row):
        try:
            if pd.isna(row['meta_data']):
                return None
            meta = ast.literal_eval(row['meta_data'])
            summ = meta.get('og', {}).get('description', None)
            return summ
        except Exception as e:
            print(f"Error processing meta_data: {e}")
            return None

    df['summary'] = df.apply(get_summary, axis=1)
    return df

df = extract_summary(df)

print(df.head(10))
print(df['summary'])

df.to_csv(file_path, index=False)

  






df.to_csv(file_path)