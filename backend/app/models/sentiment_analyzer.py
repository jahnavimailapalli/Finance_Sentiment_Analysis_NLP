import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from ..utils.model_loader import CustomModelLoader

class FinancialSentimentAnalyzer:
    def __init__(self):
        # Download required NLTK data
        self._download_nltk_data()
        
        # Initialize the custom model
        self.model_loader = CustomModelLoader()
        
        # Initialize text processing components
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        # Add financial-specific stop words
        self._add_financial_stop_words()
    
    def _download_nltk_data(self):
        """Download required NLTK data."""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')
    
    def _add_financial_stop_words(self):
        """Add financial-specific stop words."""
        financial_stop_words = {
            'stock', 'market', 'price', 'share', 'shares', 'company', 'inc', 'corp',
            'financial', 'investment', 'investor', 'trading', 'trade', 'portfolio',
            'asset', 'assets', 'security', 'securities', 'index', 'indices', 'fund',
            'funds', 'etf', 'etfs', 'dividend', 'dividends', 'yield', 'returns',
            'return', 'risk', 'risks', 'volatility', 'volatile', 'sector', 'sectors',
            'industry', 'industries', 'revenue', 'earnings', 'profit', 'profits',
            'loss', 'losses', 'margin', 'margins', 'ratio', 'ratios', 'valuation',
            'value', 'growth', 'decline', 'increase', 'decrease', 'trend', 'trends'
        }
        self.stop_words.update(financial_stop_words)
    
    def preprocess_text(self, text):
        """Preprocess the input text for sentiment analysis."""
        if not isinstance(text, str):
            raise ValueError("Input must be a string")
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stop words and lemmatize
        processed_tokens = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token not in self.stop_words and len(token) > 1
        ]
        
        return ' '.join(processed_tokens)
    
    def analyze_sentiment(self, text):
        """Analyze the sentiment of the given text."""
        if not text or not isinstance(text, str):
            raise ValueError("Text must be a non-empty string")
            
        # Preprocess the text
        processed_text = self.preprocess_text(text)
        
        if not processed_text:
            return {
                'sentiment': 'neutral',
                'confidence': 0.5,
                'original_text': text,
                'raw_scores': {'positive': 0.33, 'neutral': 0.34, 'negative': 0.33}
            }
        
        # Get sentiment prediction using custom model
        try:
            result = self.model_loader.predict_sentiment(processed_text)
            return {
                'sentiment': result['sentiment'],
                'confidence': result['confidence'],
                'original_text': text,
                'processed_text': processed_text,
                'raw_scores': result['raw_scores']
            }
        except Exception as e:
            raise RuntimeError(f"Error in sentiment analysis: {str(e)}")
    
    def analyze_batch(self, texts):
        """Analyze sentiment for multiple texts."""
        if not isinstance(texts, list):
            raise ValueError("Input must be a list of strings")
            
        results = []
        for text in texts:
            try:
                result = self.analyze_sentiment(text)
                results.append(result)
            except Exception as e:
                results.append({
                    'error': str(e),
                    'text': text
                })
        return results 