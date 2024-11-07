import pandas as pd
import textstat
import nltk
from textblob import TextBlob

# Download necessary NLTK data if not done already
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Load data
csv_path = './data/NewsContent.csv'
df = pd.read_csv(csv_path)

# Ensure missing values in 'summary' are set to None
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
        'num_adjectives': []
    }

    for _, row in df.iterrows():
        text = row.get('text', None)
        summary = row.get('summary', None)
        title = row.get('title', None)
        keywords = row.get('keywords', None)

        # Check for valid text, title, and keywords (ensure keywords is a string)
        if not text or not title or (isinstance(keywords, str) and not keywords.strip()):
            continue

        # Ensure text is a valid string before processing
        if isinstance(text, float):  # This handles NaN or float types
            text = str(text)

        features['NewsID'].append(row.get('NewsID', None))
        features['label'].append(row.get('label', None))

        features['Text Length'].append(len(text.split()) if isinstance(text, str) else 0)
        features['Summary Length'].append(len(summary.split()) if summary is not None else 0)
        
        # Handle non-string or NaN text for readability score
        if isinstance(text, str):
            features['Readability Score'].append(textstat.flesch_kincaid_grade(text) if text else None)
        else:
            features['Readability Score'].append(None)

        # Calculate keyword frequency only if keywords is a valid string
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

    return pd.DataFrame(features)

# Extract features from the DataFrame
features_df = extract_features(df)

# Save extracted features to a CSV file
features_df.to_csv('./data/ExtractedFeatures.csv', index=False)

print("Features extracted and saved to 'ExtractedFeatures.csv'.")
