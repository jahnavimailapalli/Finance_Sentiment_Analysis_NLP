from flask import Blueprint, request, jsonify
from ..models.sentiment_analyzer import FinancialSentimentAnalyzer
import yfinance as yf
from datetime import datetime, timedelta

api_bp = Blueprint('api', __name__)
analyzer = FinancialSentimentAnalyzer()

@api_bp.route('/analyze', methods=['POST'])
def analyze_sentiment():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        text = data.get('text')
        if not text:
            return jsonify({'error': 'No text provided'}), 400
            
        result = analyzer.analyze_sentiment(text)
        return jsonify(result)
    
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/analyze_batch', methods=['POST'])
def analyze_batch():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        texts = data.get('texts')
        if not texts or not isinstance(texts, list):
            return jsonify({'error': 'Invalid input format. Expected a list of texts.'}), 400
            
        results = analyzer.analyze_batch(texts)
        return jsonify({'results': results})
    
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/stock_news', methods=['GET'])
def get_stock_news():
    try:
        symbol = request.args.get('symbol')
        if not symbol:
            return jsonify({'error': 'No stock symbol provided'}), 400
            
        # Get news from Yahoo Finance
        stock = yf.Ticker(symbol)
        news = stock.news
        
        if not news:
            return jsonify({'error': 'No news found for this symbol'}), 404
        
        # Analyze sentiment for each news item
        analyzed_news = []
        for item in news:
            title = item.get('title', '')
            if title:
                try:
                    sentiment = analyzer.analyze_sentiment(title)
                    analyzed_news.append({
                        'title': title,
                        'link': item.get('link', ''),
                        'publisher': item.get('publisher', ''),
                        'published_date': item.get('providerPublishTime', ''),
                        'sentiment': sentiment
                    })
                except Exception as e:
                    analyzed_news.append({
                        'title': title,
                        'link': item.get('link', ''),
                        'publisher': item.get('publisher', ''),
                        'published_date': item.get('providerPublishTime', ''),
                        'error': str(e)
                    })
        
        return jsonify({
            'symbol': symbol,
            'news_count': len(analyzed_news),
            'news': analyzed_news
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500 