import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os

class CustomModelLoader:
    def __init__(self, model_path='sentiment_model.h5', 
                 tokenizer_path='tokenizer.pkl',
                 label_encoder_path='label_encoder.pkl'):
        self.model = None
        self.tokenizer = None
        self.label_encoder = None
        self.max_length = 100  # Adjust based on your model's requirements
        
        # Load the model
        if os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path)
        else:
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        # Load the tokenizer
        if os.path.exists(tokenizer_path):
            with open(tokenizer_path, 'rb') as handle:
                self.tokenizer = pickle.load(handle)
        else:
            raise FileNotFoundError(f"Tokenizer file not found at {tokenizer_path}")
            
        # Load the label encoder
        if os.path.exists(label_encoder_path):
            with open(label_encoder_path, 'rb') as handle:
                self.label_encoder = pickle.load(handle)
        else:
            raise FileNotFoundError(f"Label encoder file not found at {label_encoder_path}")
    
    def preprocess_text(self, text):
        """Preprocess text for the model."""
        # Convert text to sequence
        sequence = self.tokenizer.texts_to_sequences([text])
        # Pad sequence
        padded = pad_sequences(sequence, maxlen=self.max_length, padding='post')
        return padded
    
    def predict_sentiment(self, text):
        """Predict sentiment for a given text."""
        # Preprocess the text
        processed_text = self.preprocess_text(text)
        
        # Get prediction
        prediction = self.model.predict(processed_text, verbose=0)[0]
        
        # Get the predicted class index
        predicted_class = np.argmax(prediction)
        
        # Convert prediction to sentiment label using label encoder
        sentiment = self.label_encoder.inverse_transform([predicted_class])[0]
        
        # Get confidence score
        confidence = float(prediction[predicted_class])
        
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'raw_scores': {label: float(score) for label, score in zip(self.label_encoder.classes_, prediction)}
        }
    
    def predict_batch(self, texts):
        """Predict sentiment for multiple texts."""
        results = []
        for text in texts:
            try:
                result = self.predict_sentiment(text)
                results.append(result)
            except Exception as e:
                results.append({
                    'error': str(e),
                    'text': text
                })
        return results 