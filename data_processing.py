import pandas as pd 
import os
import json
import re 

file_path = './PolitiFact/PolitiFactNewsUser.txt'
with open(file_path, 'r') as file:
    content = file.read()
    #print(content)
column_names = ['NewsID', 'UserID', 'Shares']
df = pd.read_csv(file_path, delimiter='\t', names=column_names)
print(df.head())
df.to_csv('./data/News_User.csv', index=False)


user_user_path='./PolitiFact/PolitiFactUserUser.txt'
with open(user_user_path, 'r') as file:
    content=file.read()
column_names = ['follower_id', 'followed_id']
df=pd.read_csv(user_user_path,delimiter='\t', names=column_names)
print(df.head())
df.to_csv('./data/User_User.csv', index=False)


news_path = './PolitiFact/News.txt'
with open(news_path, 'r') as file:
    content = file.read()
column_names = ['filename']
df = pd.read_csv(news_path, delimiter='\t', names=column_names)
df.reset_index(drop=True, inplace=True)
df.index += 1
df.rename_axis('NewsID', inplace=True)
print(df.head())
df.to_csv('./data/News.csv', index=True)


def json_folder_to_csv(folder_path, output_csv):
    data = []
    all_keys = set()
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as f:
                json_data = json.load(f)
                result = re.sub(r'(\d+).*', r'\1', filename)
                json_data['filename'] = result
                all_keys.update(json_data.keys())
                data.append(json_data)

    df = pd.DataFrame(data)
    df = df.reindex(columns=all_keys)

    if 'filename' in df.columns:
        filename_column = df.pop('filename')
        df.insert(0, 'filename', filename_column)

    df.to_csv(output_csv, index=False)
    print(f"Data successfully saved to {output_csv}")


json_folder_to_csv('./PolitiFact/FakeNewsContent/', './data/FakeNewsContent.csv')
json_folder_to_csv('./PolitiFact/RealNewsContent/', './data/RealNewsContent.csv')

csv1_path = './data/FakeNewsContent.csv'
csv2_path = './data/RealNewsContent.csv'
df1 = pd.read_csv(csv1_path)
df2 = pd.read_csv(csv2_path)
print("Columns in FakeNewsContent:", df1.columns)
print("Columns in RealNewsContent:", df2.columns)
merged_df = pd.concat([df1, df2], ignore_index=True)
merged_df.to_csv('./data/MergedNewsContent.csv', index=False)




merged_csv_path = './data/MergedNewsContent.csv'
news_csv_path = './data/News.csv'
output_path = './data/NewsContent.csv'
merged_df = pd.read_csv(merged_csv_path)
news_df = pd.read_csv(news_csv_path)
updated_df = pd.merge(merged_df, news_df[['filename', 'NewsID']], on='filename', how='left')
column_order = ['NewsID'] + [col for col in updated_df.columns if col != 'NewsID']
updated_df = updated_df[column_order]
updated_df = updated_df.sort_values(by='NewsID')
updated_df.to_csv(output_path, index=False)




