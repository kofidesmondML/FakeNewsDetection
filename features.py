import pandas as pd
import textstat
import nltk
import spacy

nlp = spacy.load("en_core_web_sm")

csv_path = './data/NewsContent.csv'
df = pd.read_csv(csv_path)

df['summary'] = df['summary'].where(df['summary'].notna(), None)

def extract_pos_features(text):
    tagged_sent = nltk.pos_tag(nltk.word_tokenize(text))
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

def extract_features(df):
    features = {
        'NewsID': [],
        'filename': [],
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

        if not text or not title or (keywords and not keywords.strip()):
            continue

        features['NewsID'].append(row.get('NewsID', None))
        features['filename'].append(row.get('filename', None))

        features['Text Length'].append(len(text.split()) if text else 0)
        features['Summary Length'].append(len(summary.split()) if summary is not None else 0)
        features['Readability Score'].append(textstat.flesch_kincaid_grade(text) if text else None)

        keyword_count = sum(text.lower().count(keyword.lower()) for keyword in keywords.split(',')) if keywords else 0
        features['Keyword Frequency'].append(keyword_count)
        features['Title Length'].append(len(title.split()) if title else 0)

        doc = nlp(text)
        named_entities = [ent.text for ent in doc.ents]
        features['Named Entities'].append(', '.join(named_entities))

        pos_features = extract_pos_features(text)
        features['num_nouns'].append(pos_features['num_nouns'])
        features['num_propernouns'].append(pos_features['num_propernouns'])
        features['num_determinants'].append(pos_features['num_determinants'])
        features['num_adverbs'].append(pos_features['num_adverbs'])
        features['num_interjections'].append(pos_features['num_interjections'])
        features['num_verbs'].append(pos_features['num_verbs'])
        features['num_adjectives'].append(pos_features['num_adjectives'])

    return pd.DataFrame(features)

features_df = extract_features(df)
features_df.to_csv('./data/ExtractedFeatures.csv', index=False)

print("Features extracted and saved to 'ExtractedFeatures.csv'.")



