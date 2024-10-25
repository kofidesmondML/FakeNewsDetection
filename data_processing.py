import pandas as pd 
import os
import json
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


def json_folder_to_csv(folder_path, output_csv):
    data = []
    all_keys = set()
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as f:
                json_data = json.load(f)
                all_keys.update(json_data.keys())
                data.append(json_data)

    df = pd.DataFrame(data)
    df = df.reindex(columns=all_keys)
    df.to_csv(output_csv, index=True)
    print(f"Data successfully saved to {output_csv}")

json_folder_to_csv('./PolitiFact/FakeNewsContent/', './data/FakeNewsContent.csv')
json_folder_to_csv('./PolitiFact/RealNewsContent/','./data/RealNewsContent.csv')