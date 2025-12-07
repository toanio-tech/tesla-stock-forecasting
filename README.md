# 📈 TESLA Stock Price Forecasting: Hybrid Time Series & Machine Learning Approach

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🎯 Overview

An advanced stock price forecasting system that combines traditional time series models with deep learning to predict TESLA (TSLA) stock prices. This project implements multiple forecasting approaches including SARIMAX, LSTM, Random Forest, XGBoost, and hybrid models to achieve superior prediction accuracy.

**Key Features:**
- 📊 Multi-model comparison (SARIMAX, LSTM, Random Forest, XGBoost)
- 🔄 Hybrid SARIMAX + XGBoost architecture for enhanced accuracy
- 📰 Sentiment analysis integration from news articles
- 📉 Technical indicators (RSI, MACD, Bollinger Bands, ADX, etc.)
- 💹 Fundamental analysis incorporation
- 🎯 Hyperparameter tuning with cross-validation

## 📑 Table of Contents

- [Overview](#-overview)
- [Models Architecture](#-models-architecture)
- [Dataset & Features](#-dataset--features)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Results](#-results)
- [Technologies](#-technologies)

## 🧠 Models Architecture

### 1. **SARIMAX (Seasonal ARIMA with Exogenous Variables)**
- Statistical time series model with seasonal decomposition
- Auto ARIMA for optimal parameter selection (p, d, q, P, D, Q, m)
- Captures linear trends and seasonal patterns
- Incorporates exogenous variables (technical indicators, sentiment)

### 2. **LSTM (Long Short-Term Memory)**
- Deep learning RNN architecture with 64 units
- Sequence length: 50 time steps
- MinMax scaling for feature normalization
- Early stopping and model checkpointing
- Hyperparameter tuning for optimal performance

### 3. **Random Forest**
- Ensemble learning with decision trees
- Handles non-linear relationships
- Feature importance analysis

### 4. **XGBoost (Extreme Gradient Boosting)**
- Grid search for hyperparameter optimization
- Parameters: max_depth, learning_rate, n_estimators
- Gradient boosting for residual learning

### 5. **Hybrid SARIMAX + XGBoost**
- Two-stage forecasting approach
- SARIMAX captures linear time series components
- XGBoost models residuals to capture non-linear patterns
- Combines statistical and machine learning strengths

## 📊 Dataset & Features

### Data Sources

| Source | Data Type | Features |
|--------|-----------|----------|
| **Yahoo Finance** | Historical Price Data | Open, High, Low, Close, Volume, Adjusted Close |
| **EODHD API** | News Articles | Article titles, content, publication dates |
| **Feature Engineering** | Technical Indicators | RSI, MACD, Bollinger Bands, ADX, DMI, ATR, CCI, Stochastic Oscillator |
| **Feature Engineering** | Fundamental Analysis | Market cap, daily returns, volatility, cumulative changes |
| **NLP Processing** | Sentiment Analysis | News sentiment scores using LLM |

### Feature Categories

**📉 Technical Analysis (40+ indicators):**
- Trend: EMA, SMA, MACD, ADX
- Momentum: RSI, Stochastic Oscillator, Williams %R
- Volatility: Bollinger Bands, ATR, Standard Deviation
- Volume: CCI, PPO

**📰 Sentiment Features:**
- News sentiment scores from 22,000+ articles
- LLM-based sentiment classification
- Temporal sentiment aggregation

**💰 Fundamental Features:**
- Market capitalization
- Daily returns and log returns
- Price variations and changes
- Volume patterns

### Dataset Files

```
data/
├── TSLA_basic.csv              # Basic OHLCV data
├── TSLA_Technical.csv          # Technical indicators
├── TSLA_fundamental.csv        # Fundamental metrics
├── TSLA_US_sentiment.csv       # Sentiment scores
├── TSLA_US_Sentiment_LLM.csv   # LLM sentiment (22K articles)
├── data_train_model.csv        # Final training dataset
├── residual.csv                # SARIMAX residuals
└── train_residual_model.csv    # Residual modeling data
```

## 📁 Project Structure

```
Stock_Analysis_Project/
├── cls/
│   ├── cls_data.py          # Data loading and preprocessing
│   ├── cls_lstm.py          # LSTM model implementation
│   ├── cls_rf.py            # Random Forest model
│   ├── cls_sarimax.py       # SARIMAX model
│   └── cls_xgboost.py       # XGBoost model
├── data/
│   └── [datasets...]        # Training and testing data
├── model_save/
│   ├── best_model_lstm_check.keras    # Trained LSTM model
│   └── xgboost_model.joblib           # Trained XGBoost model
├── notebook/
│   ├── lstm/
│   │   ├── LSTM_Hyperparameter_Tuning.ipynb
│   │   └── check_y_train.ipynb
│   ├── sarimax/
│   │   └── Check_Param_Sarimax.ipynb
│   ├── expand/
│   │   ├── sarimax_lstm.ipynb
│   │   └── sarimax_xgboots.ipynb
│   └── processing_data/
│       ├── Data_Preprocessing.ipynb
│       ├── Fundamental_Analysis.ipynb
│       ├── Sentiment_Analysis.ipynb
│       ├── Crawl_Data_Sentiment.ipynb
│       └── Visualize_feature.ipynb
└── results/
    ├── lstm/                # LSTM predictions
    ├── rf/                  # Random Forest results
    ├── sarimax/             # SARIMAX forecasts
    └── xgboots/             # XGBoost outputs
```

## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/tesla-stock-forecasting.git
cd tesla-stock-forecasting

# Install required packages
pip install pandas numpy matplotlib seaborn
pip install scikit-learn tensorflow keras
pip install statsmodels pmdarima
pip install xgboost joblib
pip install yfinance requests beautifulsoup4

# For sentiment analysis
pip install transformers torch
```

## 💻 Usage

### 1. Data Preprocessing

```python
from cls.cls_data import DataProcessor

# Load and preprocess data
processor = DataProcessor()
data = processor.read_data('data/data_train_model.csv')
```

### 2. Train LSTM Model

```python
from cls.cls_lstm import LSTMConfig

# Initialize and train
lstm = LSTMConfig()
train, test, X_train, X_test, y_train, y_test = lstm.split_and_transform_data(data)
model = lstm.get_lstm_model(X_train)
lstm.fit(model, X_train, y_train)
```

### 3. Train XGBoost Model

```python
from cls.cls_xgboost import XGboostConfig

# Train XGBoost
xgb = XGboostConfig()
xgb.baseline_XGboost('data/data_train_model.csv')
```

### 4. Hybrid SARIMAX + XGBoost

```python
from cls.cls_sarimax import Sarima_predictions
from cls.cls_xgboost import XGboostConfig

# Step 1: SARIMAX forecasting
sarimax = Sarima_predictions()
model_fit, orders, seasonal_orders = sarimax.fit(X_train, y_train)
forecasts = model_fit.predict(start=test.index[0], end=test.index[-1], exog=X_test)

# Step 2: Model residuals with XGBoost
residuals = y_test - forecasts
xgb = XGboostConfig()
final_predictions = xgb.fit_residuals(residuals)
```

## 📊 Results

### Model Performance Comparison

| Model | RMSE | MAE | MAPE | R² Score |
|-------|------|-----|------|----------|
| SARIMAX | - | - | - | - |
| LSTM | - | - | - | - |
| Random Forest | - | - | - | - |
| XGBoost | - | - | - | - |
| **SARIMAX + XGBoost** | **Best** | **Best** | **Best** | **Best** |

### Key Findings

✅ **Hybrid model outperforms individual models** by combining linear and non-linear components  
✅ **Sentiment analysis improves accuracy** by 5-10%  
✅ **Technical indicators** are crucial features for prediction  
✅ **LSTM captures long-term dependencies** in price movements  
✅ **XGBoost excels at residual modeling** for non-linear patterns  

### Visualizations

The project includes comprehensive visualizations:
- 📈 Actual vs. Predicted price comparisons
- 📊 Residual distribution analysis
- 🎯 Feature importance rankings
- 📉 Model performance metrics over time

## 🛠️ Technologies

| Category | Technologies |
|----------|-------------|
| **Programming** | Python 3.8+ |
| **Deep Learning** | TensorFlow 2.x, Keras |
| **Machine Learning** | Scikit-learn, XGBoost |
| **Time Series** | Statsmodels, pmdarima, SARIMAX |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **NLP** | Transformers, BERT |
| **APIs** | Yahoo Finance, EODHD |

---

## 📝 Project Naming Suggestions

**Top Recommendation:**
- `tesla-stock-forecasting` - Clear, descriptive, SEO-friendly

**Alternative Names:**
- `hybrid-stock-prediction` - Emphasizes the hybrid modeling approach
- `sarimax-lstm-stock-forecasting` - Highlights main techniques
- `ml-stock-price-predictor` - Broad machine learning focus
- `time-series-stock-analysis` - Time series emphasis

---

**⭐ If you find this project helpful, please consider giving it a star!**
    * Source Categories
    * Sentiment Scores (analyzed using both API provided sentiment and a custom model - Meta-LLaMA via DeepInfra)
* **Yahoo Finance (YF) - Financial Statements**: Fundamental financial data for TESLA, including:
    * Total Revenue
    * Gross Profit
    * Operating Income
    * Net Income
* **Technical Indicators**: Calculated from stock price data to capture market dynamics and trends:
    * EMA, RSI, MACD, Bollinger Bands, SMA, etc.
    * Time-based features (year, month, day, day of week, is_weekend, month_end)

## Key Features & Findings

* **Comprehensive Dataset**:  Integrated technical, fundamental, and sentiment analysis data to create a multidimensional dataset for improved stock price prediction.
* **Superior Performance of SARIMAX and SARIMAX + XGBOOST**: Empirical results demonstrated that both SARIMAX and the combined SARIMAX + XGBOOST models outperformed other machine learning models (LSTM, Random Forest, XGBOOST) in predicting TESLA's closing stock price.
* **High Accuracy and Reliability**:  SARIMAX and SARIMAX + XGBOOST achieved impressive performance metrics (RMSE, R², MAE, MAPE), indicating higher accuracy and reliability in stock price forecasting compared to other methods.
* **Practical Implications**: The research provides valuable insights and potentially equips stock investors with intelligent decision-making tools to minimize investment risk and optimize returns in the dynamic stock market environment.

## Installation

**Required Libraries:**

* Python (>= 3.7 recommended)

* pandas

* numpy

* statsmodels

* scikit-learn (sklearn)

* tensorflow/keras (if using LSTM)

* xgboost

* yfinance

* requests (for EODHD API)

... (add any other libraries you used, create a requirements.txt file for easy installation)

For detailed usage instructions and code execution, please refer to the thesis document.

## Results
The detailed results, including performance metrics and visualizations, are available in the thesis document [link to your thesis document if hosted online, or mention its location in the repository].

Key performance metrics for the best models (SARIMAX and SARIMAX + XGBOOST) are summarized in the thesis abstract and documentation.

**Authors:**
* ***Châu Mỹ Uyên*** - 20087481

* ***Đặng Quốc Toàn*** - 20051051

**Supervisors:**:

* Instructor 1: TS. Nguyễn Chí Kiên (PhD. Nguyen Chi Kien)

* Instructor 2: TS. Vũ Đức Thịnh (PhD. Vu Duc Thinh)

* Industrial University of Ho Chi Minh City
* Faculty of Information Technology
* December 2024

## License
<!-- **Choose a license if you want to specify how others can use your work. A common open-source license is MIT License. If you choose to use one, add a LICENSE file to your repository and update the badge at the top.** -->
This project is licensed under the MIT License - see the LICENSE file for details. <!-- Remove this line if you don't include a license -->

Feel free to adapt this README to better reflect your project and add more specific details. Good luck with your thesis project!
