import pandas as pd 

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