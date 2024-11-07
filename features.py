import pandas as pd
import textstat
import string
import ast
import nltk
from textblob import TextBlob
 
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

csv_path = './data/NewsContent.csv'
news_user_path = './data/News_user.csv'
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
        'Total Shares': [],
        'Unique User Shares': [],
        'image_present': [],
        #'punctuation_count': []
    }

    for _, row in df.iterrows():
        text = row.get('text', None)
        summary = row.get('summary', None)
        title = row.get('title', None)
        keywords = row.get('keywords', None)

        if not text or not title or (isinstance(keywords, str) and not keywords.strip()):
            continue

        if isinstance(text, float):
            text = str(text)

        features['NewsID'].append(row.get('NewsID', None))
        features['label'].append(row.get('filename', None))
        features['Text Length'].append(len(text.split()) if isinstance(text, str) else 0)
        features['Summary Length'].append(len(summary.split()) if summary is not None else 0)
        features['Readability Score'].append(textstat.flesch_kincaid_grade(text) if text else None)
        keyword_count = sum(text.lower().count(keyword.lower()) for keyword in keywords.split(',')) if isinstance(keywords, str) else 0
        features['Keyword Frequency'].append(keyword_count)
        features['Title Length'].append(len(title.split()) if title else 0)
        named_entities = extract_named_entities(text)
        features['Named Entities'].append(named_entities)
        pos_features = extract_pos_features(text)
        features['num_nouns'].append(pos_features['num_nouns'])
        features['num_propernouns'].append(pos_features['num_propernouns'])
        features['num_determinants'].append(pos_features['num_determinants'])
        features['num_adverbs'].append(pos_features['num_adverbs'])
        features['num_interjections'].append(pos_features['num_interjections'])
        features['num_verbs'].append(pos_features['num_verbs'])
        features['num_adjectives'].append(pos_features['num_adjectives'])
        punctuation_count=count_punctuations(text)
        #features['punctuation_count'].append(punctuation_count)

    return pd.DataFrame(features)

features_df = extract_features(df)
total_shares_df = calculate_total_shares(news_user_path)
unique_user_shares_df = count_unique_user_shares(news_user_path)
image_presence_df = mark_top_img_presence(df)
image_count_df = count_images(df)

if total_shares_df is not None:
    features_df = features_df.merge(total_shares_df, on='NewsID', how='left')
if unique_user_shares_df is not None:
    features_df = features_df.merge(unique_user_shares_df, on='NewsID', how='left')
if image_presence_df is not None:
    features_df = features_df.merge(image_presence_df, on='NewsID', how='left')
if image_count_df is not None:
    features_df = features_df.merge(image_count_df, on='NewsID', how='left')
features_df.to_csv('ExtractedFeatures.csv', index=False)
