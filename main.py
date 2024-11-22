import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab') 

def load_and_clean_data(file_path):
    """Loads and cleans Sentiment140 dataset."""
    columns = ['sentiment', 'id', 'date', 'query', 'user', 'text']
    df = pd.read_csv(file_path, encoding='latin-1', names=columns)

    df['sentiment'] = df['sentiment'].map({0: 'Negative', 2: 'Neutral', 4: 'Positive'})

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    def clean_text(text):
        """Cleans the text data by removing URLs, mentions, hashtags, and punctuation."""
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"@\w+", "", text)
        text = re.sub(r"#\w+", "", text)
        text = re.sub(r"[^\w\s]", "", text) 
        text = text.lower()                
        tokens = word_tokenize(text)        
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
        return " ".join(tokens)

    df['cleaned_text'] = df['text'].apply(clean_text)
    return df

def visualize_data(df):
    """Generates visualizations for sentiment distribution and word clouds."""
    sentiment_counts = df['sentiment'].value_counts()
    plt.figure(figsize=(8, 6))
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette='coolwarm')
    plt.title("Sentiment Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.show()

    for sentiment in ['Positive', 'Negative', 'Neutral']:
        text = " ".join(tweet for tweet in df[df['sentiment'] == sentiment]['cleaned_text'])
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
        plt.figure(figsize=(10, 8))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f"Word Cloud for {sentiment} Sentiment")
        plt.show()

    plt.figure(figsize=(10, 8))
    all_sentiments = df.groupby('sentiment')['cleaned_text'].apply(lambda x: ' '.join(x))
    for sentiment in all_sentiments.index:
        words = ' '.join(all_sentiments[sentiment].split())
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(words)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f"Most Frequent Words in {sentiment} Sentiment")
        plt.show()

def train_model(df):
    """Trains a sentiment classification model using Naive Bayes."""
    X = df['cleaned_text'] 
    y = df['sentiment']

    vectorizer = CountVectorizer(max_features=5000)
    X_vectorized = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

    model = MultinomialNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred, labels=['Positive', 'Negative', 'Neutral'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Positive', 'Negative', 'Neutral'], yticklabels=['Positive', 'Negative', 'Neutral'])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

def save_preprocessed_data(df):
    """Saves the cleaned dataset to a new CSV file."""
    df.to_csv('./data/cleaned_sentiment140.csv', index=False)
    print("Cleaned data saved to 'cleaned_sentiment140.csv'.")

if __name__ == "__main__":
    file_path = './data/sentiment140.csv'

    print("Loading and cleaning data...")
    df = load_and_clean_data(file_path)

    print("\nVisualizing data...")
    visualize_data(df)

    print("\nTraining sentiment classifier...")
    train_model(df)

    print("\nSaving preprocessed data...")
    save_preprocessed_data(df)
