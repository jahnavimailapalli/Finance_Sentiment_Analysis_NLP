# Financial Sentiment Analysis System

This project provides a machine learning-based sentiment analysis system specifically designed for financial texts. It uses the FinBERT model, which is pre-trained on financial texts, to analyze sentiment in financial news, reports, and social media content.

## Features

- Real-time sentiment analysis of financial texts
- Batch processing of multiple texts
- Integration with Yahoo Finance API for stock news analysis
- RESTful API for easy integration
- Preprocessing pipeline optimized for financial texts

## Setup

1. Clone the repository
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Start the backend server:
   ```bash
   cd backend
   python app.py
   ```

The API will be available at `http://localhost:5000`

## API Endpoints

### Analyze Single Text
- **Endpoint**: `/analyze`
- **Method**: POST
- **Request Body**:
  ```json
  {
    "text": "Your financial text here"
  }
  ```
- **Response**:
  ```json
  {
    "sentiment": "positive/negative/neutral",
    "confidence": 0.95
  }
  ```

### Analyze Multiple Texts
- **Endpoint**: `/analyze_batch`
- **Method**: POST
- **Request Body**:
  ```json
  {
    "texts": ["Text 1", "Text 2", "Text 3"]
  }
  ```

### Get Stock News with Sentiment
- **Endpoint**: `/stock_news`
- **Method**: GET
- **Query Parameter**: `symbol` (e.g., AAPL, GOOGL)
- **Example**: `/stock_news?symbol=AAPL`

## Technologies Used

- Python
- Flask
- Transformers (FinBERT)
- NLTK
- yfinance
- Pandas
- NumPy

## License

MIT License 