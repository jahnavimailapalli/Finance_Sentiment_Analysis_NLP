from flask import Flask, request, jsonify
from flask_cors import CORS
from sentiment_analyzer import FinancialSentimentAnalyzer
import yfinance as yf
from datetime import datetime, timedelta

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
analyzer = FinancialSentimentAnalyzer()

@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    try:
        data = request.get_json()
        text = data.get('text')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
            
        result = analyzer.analyze_sentiment(text)
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analyze_batch', methods=['POST'])
def analyze_batch():
    try:
        data = request.get_json()
        texts = data.get('texts')
        
        if not texts or not isinstance(texts, list):
            return jsonify({'error': 'Invalid input format'}), 400
            
        results = analyzer.analyze_batch(texts)
        return jsonify({'results': results})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/stock_news', methods=['GET'])
def get_stock_news():
    try:
        symbol = request.args.get('symbol')
        if not symbol:
            return jsonify({'error': 'No stock symbol provided'}), 400
            
        stock = yf.Ticker(symbol)
        news = stock.news
        
        # Analyze sentiment for each news item
        analyzed_news = []
        for item in news:
            title = item.get('title', '')
            sentiment = analyzer.analyze_sentiment(title)
            analyzed_news.append({
                'title': title,
                'link': item.get('link', ''),
                'publisher': item.get('publisher', ''),
                'published_date': item.get('providerPublishTime', ''),
                'sentiment': sentiment
            })
            
        return jsonify({'news': analyzed_news})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000) 