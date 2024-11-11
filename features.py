import pandas as pd
import textstat
import string
import networkx as nx
from itertools import combinations
import ssl
import ast
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import re
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


ssl._create_default_https_context = ssl._create_unverified_context 
emotion_lexicon_path = 'NRC-Emotion-Lexicon-Wordlevel-v0.92.txt'
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

csv_path = './data/NewsContent.csv'
news_user_path = './data/News_User.csv'
user_user_path='./data/User_user.csv'
graph_path='./PolitiFact/PolitiFactUserUser.txt'
df = pd.read_csv(csv_path)

def calculate_total_shares(filepath=news_user_path):
    try:
        df_shares = pd.read_csv(filepath)
        shares_per_news = df_shares.groupby('NewsID')['Shares'].sum().reset_index()
        shares_per_news.columns = ['NewsID', 'TotalShares']
        return shares_per_news
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
    except pd.errors.EmptyDataError:
        print("Error: The file is empty.")
    except pd.errors.ParserError:
        print("Error: The file is not formatted correctly.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def mark_top_img_presence(df=df):
    try:
        df['image_present'] = df['top_img'].apply(lambda x: 1 if pd.notna(x) and x.strip() else 0)
        return df[['NewsID', 'image_present']]
    except KeyError:
        print("Error: 'top_img' or 'NewsID' column not found in the DataFrame.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def count_images(df):
    try:
        if df.empty:
            print("Error: The DataFrame is empty.")
            return pd.DataFrame(columns=['NewsID', 'image_count'])
        
        if 'images' not in df.columns or 'NewsID' not in df.columns:
            print("Error: 'images' or 'NewsID' column not found in the DataFrame.")
            return pd.DataFrame(columns=['NewsID', 'image_count'])
        
        image_counts = []
        news_ids = []
        for index, row in df.iterrows():
            if pd.isna(row['images']) or row['images'] == '':
                image_counts.append(0)
            else:
                img = row['images']
                img = ast.literal_eval(img)
                count_of_images = len(img)
                image_counts.append(count_of_images)
            
            news_ids.append(row['NewsID'])
        
        result_df = pd.DataFrame({'NewsID': news_ids, 'image_count': image_counts})
        return result_df
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return pd.DataFrame(columns=['NewsID', 'image_count'])

def count_unique_user_shares(filepath=news_user_path):
    try:
        df = pd.read_csv(filepath)
        unique_user_shares = df.groupby('NewsID')['UserID'].nunique().reset_index()
        unique_user_shares.columns = ['NewsID', 'UniqueUserShares']
        return unique_user_shares
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
    except pd.errors.EmptyDataError:
        print("Error: The file is empty.")
    except pd.errors.ParserError:
        print("Error: The file is not formatted correctly.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

df['summary'] = df['summary'].where(df['summary'].notna(), None)

def extract_pos_features(text):
    tokens = nltk.word_tokenize(text)
    tagged_sent = nltk.pos_tag(tokens)
    pos_dict = {
        'num_nouns': len([word for word, pos in tagged_sent if pos == 'NN']),
        'num_propernouns': len([word for word, pos in tagged_sent if pos == 'NNP']),
        'num_determinants': len([word for word, pos in tagged_sent if pos == 'DT']),
        'num_adverbs': len([word for word, pos in tagged_sent if pos == 'RB']),
        'num_interjections': len([word for word, pos in tagged_sent if pos == 'UH']),
        'num_verbs': len([word for word, pos in tagged_sent if pos.startswith('VB')]),
        'num_adjectives': len([word for word, pos in tagged_sent if pos == 'JJ'])
    }
    return pos_dict

def extract_named_entities(text):
    blob = TextBlob(text)
    named_entities = [entity for entity, tag in blob.tags if tag == 'NNP']
    return ', '.join(named_entities)

def count_punctuations(text):
    try:
        return sum(1 for char in text if char in string.punctuation)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return 0 

def get_follower_counts(filepath):
    try:
        df = pd.read_csv(filepath)
        follower_counts = df.groupby('followed_id')['follower_id'].nunique().reset_index()
        follower_counts.columns = ['UserID', 'FollowerCount']
        return follower_counts
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        return pd.DataFrame(columns=['UserID', 'FollowerCount'])
    except pd.errors.EmptyDataError:
        print("Error: The file is empty.")
        return pd.DataFrame(columns=['UserID', 'FollowerCount'])
    except pd.errors.ParserError:
        print("Error: The file is not formatted correctly.")
        return pd.DataFrame(columns=['UserID', 'FollowerCount'])
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return pd.DataFrame(columns=['UserID', 'FollowerCount'])

def get_unique_users_per_news(df):
    try:
        unique_users_per_news = df.groupby('NewsID')['UserID'].unique().reset_index()
        unique_users_per_news.columns = ['NewsID', 'UniqueUserList']
        unique_users_per_news = unique_users_per_news.sort_values(by='NewsID').reset_index(drop=True)
        return unique_users_per_news
    except KeyError as e:
        print(f"Error: Missing column in the DataFrame - {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def get_avg_followers_per_news(news_df, follower_df):
    try:
        unique_users_per_news = get_unique_users_per_news(news_df)
        merged = unique_users_per_news.explode('UniqueUserList')
        merged = merged.merge(follower_df, how='left', left_on='UniqueUserList', right_on='UserID')
        avg_followers_per_news = merged.groupby('NewsID')['FollowerCount'].mean().reset_index()
        avg_followers_per_news.columns = ['NewsID', 'AvgFollowers']
        return avg_followers_per_news
    except KeyError as e:
        print(f"Error: Missing column in the DataFrame - {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")  

    
def jaccard_similarity(node1, node2, G):
    neighbors1 = set(G.neighbors(int(node1)))  
    neighbors2 = set(G.neighbors(int(node2)))  
    intersection = len(neighbors1.intersection(neighbors2))
    union = len(neighbors1.union(neighbors2))
    if union == 0:
        return 0
    return intersection / union
def average_node_similarity(G,df):
    results = []
    for news_id in df['NewsID'].unique():
        users = df[df['NewsID'] == news_id]['UniqueUserList'].values[0]
        similarities = []
        for user1, user2 in combinations(users, 2):
            similarity = jaccard_similarity(user1, user2, G)
            similarities.append(similarity)
        if similarities:
            avg_similarity = sum(similarities) / len(similarities)
        else:
            avg_similarity = 0
        results.append([news_id, avg_similarity])
    
    return pd.DataFrame(results, columns=['NewsID', 'AverageSimilarity'])

def analyze_sentiment_vader(text):
    if isinstance(text, str):
        sid = SentimentIntensityAnalyzer()
        sentiment = sid.polarity_scores(text)
        return sentiment
    else:
        return {'neg': 0.0, 'neu': 0.0, 'pos': 0.0, 'compound': 0.0}

def get_sentiment_for_titles(df):
    sentiment_scores = df['title'].apply(analyze_sentiment_vader).apply(pd.Series)
    sentiment_scores.columns = [f'title_{col}' for col in sentiment_scores.columns]
    df_sentiment = pd.concat([df['NewsID'], sentiment_scores], axis=1)
    return df_sentiment

def get_sentiment_for_text(df):
    sentiment_scores = df['text'].apply(analyze_sentiment_vader).apply(pd.Series)
    sentiment_scores.columns = [f'text_{col}' for col in sentiment_scores.columns]
    df_sentiment = pd.concat([df['NewsID'], sentiment_scores], axis=1)
    return df_sentiment

lmtzr = WordNetLemmatizer()
stops = list(set(stopwords.words('english') + list(set(ENGLISH_STOP_WORDS)) + ["http"]))

def emotion_NRC(df, emo_dic_path, title_or_text):
    emotion_dic = NRC_dict(emo_dic_path)
    results_emo = pd.DataFrame()
    
    for x, row in tqdm(df.iterrows(), total=df.shape[0]):
        extracted_emotion = getEmotionVector(row[title_or_text], emotion_dic)
        extracted_emotion = {f"{title_or_text}_{key}": value for key, value in extracted_emotion.items()}
        temp = pd.DataFrame(extracted_emotion, index=[x])
        results_emo = pd.concat([results_emo, temp], ignore_index=True)
    
    df_emo = pd.concat([df[['NewsID']], results_emo], axis=1)
    return df_emo

def NRC_dict(fileEmotion):
    emotion_df = pd.read_csv(fileEmotion, names=["word", "emotion", "intensity"], sep='\t')
    emotion_dic = {}
    
    for index, row in emotion_df.iterrows():
        word = str(row['word'])
        emotion = str(row['emotion'])
        temp_key = word + '#' + emotion
        emotion_dic[temp_key] = row['intensity']
        
        temp_key_n = str(lmtzr.lemmatize(word)) + '#' + emotion
        emotion_dic[temp_key_n] = row['intensity']
        temp_key_v = str(lmtzr.lemmatize(word, 'v')) + '#' + emotion
        emotion_dic[temp_key_v] = row['intensity']
    
    return emotion_dic

def getEmotionItensity(word, emotion, emotion_dic):
    key = word + "#" + emotion
    return emotion_dic.get(key, 0.0)

def isWordInEmotionFile(word, emotion_dic):
    result = [(key) for key in emotion_dic.keys() if key.startswith(word + "#")]
    return len(result) > 0

def isStopWord(word):
    return word in stops

def calculateEmotion(emotions, word, emotion_dic):
    emotions["Anger"] += getEmotionItensity(word, "anger", emotion_dic)
    emotions["Anticipation"] += getEmotionItensity(word, "anticipation", emotion_dic)
    emotions["Disgust"] += getEmotionItensity(word, "disgust", emotion_dic)
    emotions["Fear"] += getEmotionItensity(word, "fear", emotion_dic)
    emotions["Joy"] += getEmotionItensity(word, "joy", emotion_dic)
    emotions["Sadness"] += getEmotionItensity(word, "sadness", emotion_dic)
    emotions["Surprise"] += getEmotionItensity(word, "surprise", emotion_dic)
    emotions["Trust"] += getEmotionItensity(word, "trust", emotion_dic)

def getEmotionVector(text, emotion_dic):
    emotions = {"Anger": 0.0, "Anticipation": 0.0, "Disgust": 0.0, "Fear": 0.0, "Joy": 0.0, "Sadness": 0.0, "Surprise": 0.0, "Trust": 0.0, "Objective": 0.0}
    str_ = re.sub("[^a-zA-Z]+", " ", str(text))
    str_ = re.sub(r'[^a-zA-Z ]+', '', str_).lower()
    words = str_.split()
    
    for word in words:
        if not isStopWord(word):
            if isWordInEmotionFile(word, emotion_dic): 
                calculateEmotion(emotions, word, emotion_dic)
            elif isWordInEmotionFile(lmtzr.lemmatize(word), emotion_dic):
                calculateEmotion(emotions, lmtzr.lemmatize(word), emotion_dic)
            elif isWordInEmotionFile(lmtzr.lemmatize(word, 'v'), emotion_dic):
                calculateEmotion(emotions, lmtzr.lemmatize(word, 'v'), emotion_dic)
            else:
                emotions["Objective"] += 1
    
    total = sum(emotions.values())
    for key in sorted(emotions.keys()):
        emotions[key] = (1.0 / total) * emotions[key] if total > 0 else 0
    return emotions


def extract_features(df):
    features = {
        'NewsID': [],
        'label': [],
        'Text Length': [],
        'Summary Length': [],
        'Readability Score': [],
        'Keyword Frequency': [],
        'Title Length': [],
        'Named Entities': [],
        'num_nouns': [],
        'num_propernouns': [],
        'num_determinants': [],
        'num_adverbs': [],
        'num_interjections': [],
        'num_verbs': [],
        'num_adjectives': [],
        'punctuation_count': []
    }

    for _, row in df.iterrows():
        text = str(row.get('text', "")) if pd.notna(row.get('text')) else ""
        summary = str(row.get('summary', "")) if pd.notna(row.get('summary')) else ""
        title = str(row.get('title', "")) if pd.notna(row.get('title')) else ""
        keywords = str(row.get('keywords', "")) if pd.notna(row.get('keywords')) else ""

        features['NewsID'].append(row.get('NewsID', None))
        features['label'].append(row.get('filename', None))
        features['Text Length'].append(len(text.split()) if text else 0)
        features['Summary Length'].append(len(summary.split()) if summary else 0)
        features['Readability Score'].append(textstat.flesch_kincaid_grade(text) if text else 0)
        keyword_count = sum(text.lower().count(keyword.lower()) for keyword in keywords.split(',')) if keywords else 0
        features['Keyword Frequency'].append(keyword_count)
        features['Title Length'].append(len(title.split()) if title else 0)
        named_entities = extract_named_entities(text) if text else ""
        features['Named Entities'].append(named_entities)

        pos_features = extract_pos_features(text)
        features['num_nouns'].append(pos_features['num_nouns'])
        features['num_propernouns'].append(pos_features['num_propernouns'])
        features['num_determinants'].append(pos_features['num_determinants'])
        features['num_adverbs'].append(pos_features['num_adverbs'])
        features['num_interjections'].append(pos_features['num_interjections'])
        features['num_verbs'].append(pos_features['num_verbs'])
        features['num_adjectives'].append(pos_features['num_adjectives'])
        punctuation_count = count_punctuations(text)
        features['punctuation_count'].append(punctuation_count)

    for key, value in features.items():
        print(f"{key}: {len(value)}")

    return pd.DataFrame(features)


G = nx.read_edgelist(graph_path, nodetype=int)
title_sentiment_df=get_sentiment_for_titles(df)
text_sentiment_df=get_sentiment_for_text(df)
news_user_df=pd.read_csv(news_user_path)
unique_users=get_unique_users_per_news(news_user_df)
follower_counts_df = get_follower_counts(user_user_path)
news_user_df = pd.read_csv(news_user_path)
avg_followers_df = get_avg_followers_per_news(news_user_df, follower_counts_df)
features_df = extract_features(df)
total_shares_df = calculate_total_shares(news_user_path)
unique_user_shares_df = count_unique_user_shares(news_user_path)
image_presence_df = mark_top_img_presence(df)
image_count_df = count_images(df)
node_similarity_df=average_node_similarity(G,df=unique_users)
title_emotion_df = emotion_NRC(df, emotion_lexicon_path, 'title')
text_emotion_df=emotion_NRC(df,emotion_lexicon_path, 'text')



if total_shares_df is not None:
    features_df = features_df.merge(total_shares_df, on='NewsID', how='left')
if text_emotion_df is not None:
    features_df=features_df.merge(text_emotion_df, on='NewsID', how='left' )
if unique_user_shares_df is not None:
    features_df = features_df.merge(unique_user_shares_df, on='NewsID', how='left')
if image_presence_df is not None:
    features_df = features_df.merge(image_presence_df, on='NewsID', how='left')
if image_count_df is not None:
    features_df = features_df.merge(image_count_df, on='NewsID', how='left')
if avg_followers_df is not None:
    features_df=features_df.merge(avg_followers_df, on='NewsID', how='left')
if node_similarity_df is not None:
    features_df=features_df.merge(node_similarity_df, on='NewsID', how='left')
if title_sentiment_df is not None:
    features_df=features_df.merge(title_sentiment_df, on='NewsID',how='left')
if text_sentiment_df is not None:
    features_df=features_df.merge(text_sentiment_df, on='NewsID', how='left')
if title_emotion_df is not None:
    features_df=features_df.merge(title_emotion_df,on='NewsID', how='left')
print(features_df.columns)
features_df.to_csv('./data/ExtractedFeatures.csv', index=False)
print('All features extracted successfully')
