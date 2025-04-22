import nltk
from transformers import pipeline
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

class FinancialSentimentAnalyzer:
    def __init__(self):
        # Download required NLTK data
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        
        # Initialize the sentiment analysis pipeline
        self.sentiment_analyzer = pipeline("sentiment-analysis", model="ProsusAI/finbert")
        
        # Initialize text processing components
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        # Add financial-specific stop words
        self.stop_words.update(['stock', 'market', 'price', 'share', 'shares', 'company', 'inc', 'corp'])
    
    def preprocess_text(self, text):
        """Preprocess the input text for sentiment analysis."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stop words and lemmatize
        processed_tokens = [self.lemmatizer.lemmatize(token) 
                          for token in tokens 
                          if token not in self.stop_words]
        
        return ' '.join(processed_tokens)
    
    def analyze_sentiment(self, text):
        """Analyze the sentiment of the given text."""
        # Preprocess the text
        processed_text = self.preprocess_text(text)
        
        # Get sentiment prediction
        result = self.sentiment_analyzer(processed_text)[0]
        
        return {
            'sentiment': result['label'],
            'confidence': result['score']
        }
    
    def analyze_batch(self, texts):
        """Analyze sentiment for multiple texts."""
        results = []
        for text in texts:
            result = self.analyze_sentiment(text)
            results.append(result)
        return results 