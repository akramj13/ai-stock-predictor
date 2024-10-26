import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import precision_score
from ta import add_all_ta_features
from ta.utils import dropna
from ta.momentum import StochasticOscillator
from ta.volatility import BollingerBands
from ta.volume import OnBalanceVolumeIndicator

# Obtain stock data for any desired stock
ticker = "JNJ" 
stock = yf.Ticker(ticker)
stock_data = stock.history(start="2000-01-01")

# Handle cases where stock data might be empty
# E.g., if the stock was listed after 2000
if stock_data.empty:
    stock_data = stock.history(period="max")

# Drop unnecessary columns for this analysis
columns_to_drop = ['Dividends', 'Stock Splits']
stock_data.drop(columns=[col for col in columns_to_drop if col in stock_data.columns], inplace=True)

# Technical analysis metrics and adding them to the dataframe
stock_data = dropna(stock_data)  # Ensuring no null values
stock_data = add_all_ta_features(
    stock_data, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True
)

# Advanced technical indicators
stochastic = StochasticOscillator(
    high=stock_data['High'], low=stock_data['Low'], close=stock_data['Close'], window=14, smooth_window=3
)
bollinger = BollingerBands(close=stock_data['Close'], window=20, window_dev=2)
obv = OnBalanceVolumeIndicator(close=stock_data['Close'], volume=stock_data['Volume'])

# Combine all additional columns at once to avoid fragmentation
additional_columns = pd.DataFrame({
    'Stochastic %K': stochastic.stoch(),
    'Stochastic %D': stochastic.stoch_signal(),
    'Bollinger MAVG': bollinger.bollinger_mavg(),
    'Bollinger High': bollinger.bollinger_hband(),
    'Bollinger Low': bollinger.bollinger_lband(),
    'OBV': obv.on_balance_volume(),
    'Day of Week': stock_data.index.dayofweek,
    'Month': stock_data.index.month,
    'Quarter': stock_data.index.quarter
}, index=stock_data.index)

# Concatenate the new columns
stock_data = pd.concat([stock_data, additional_columns], axis=1)

# Create target variable for predicting if the next day's close will be greater
stock_data['Next Day'] = stock_data['Close'].shift(-1)
stock_data['Next Day Greater?'] = (stock_data['Next Day'] > stock_data['Close']).astype(int)
stock_data.dropna(inplace=True)

# Define the model
model = RFC(n_estimators=100, min_samples_split=100, random_state=1)

# Split the data into training and testing sets
train = stock_data.iloc[:-100]
test = stock_data.iloc[-100:]

# List of predictors including the newly added ones
predictors = [
    'Open', 'High', 'Low', 'Close', 'Volume', 'momentum_rsi', 'trend_macd',
    'volatility_bbm', 'volatility_bbl', 'Stochastic %K', 'Stochastic %D', 
    'Bollinger MAVG', 'Bollinger High', 'Bollinger Low', 'OBV', 
    'Day of Week', 'Month', 'Quarter'
]

# Function to train and predict
def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Next Day Greater?"])
    predictions = model.predict(test[predictors])
    return pd.DataFrame({
        'Actual': test['Next Day Greater?'],
        'Predictions': predictions
    }, index=test.index)

# Modified backtesting function
def backtest(data, model, predictors, start_fraction=0.2, step=250):
    if len(data) < step:
        print("Not enough data for backtesting.")
        return pd.DataFrame()  # Return an empty DataFrame if insufficient data

    start = int(len(data) * start_fraction)
    all_predictions = []

    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i + step)].copy()
        if len(test) == 0:
            break
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)
    
    return pd.concat(all_predictions) if all_predictions else pd.DataFrame()

# Backtest the model with initial predictors
predictions = backtest(stock_data, model, predictors)

if not predictions.empty:
    print(predictions['Predictions'].value_counts())
    print("Precision Score:", precision_score(predictions['Actual'], predictions['Predictions']))
else:
    print("No predictions were generated. Try with a different stock or adjust parameters.")

# Create new predictors based on different horizons
horizons = [2, 5, 60, 250, 1000]
new_columns = {}

for horizon in horizons:
    new_columns[f"Close Ratio {horizon} Days"] = stock_data["Close"] / stock_data['Close'].rolling(horizon).mean()
    new_columns[f"Trend {horizon} Days"] = stock_data['Next Day Greater?'].shift(1).rolling(horizon).sum()

# Combine new columns at once to prevent fragmentation
stock_data = pd.concat([stock_data, pd.DataFrame(new_columns, index=stock_data.index)], axis=1)

# Drop NaN values resulting from rolling operations
stock_data.dropna(inplace=True)

# Update model with new parameters
model = RFC(n_estimators=450, min_samples_split=50, random_state=1, max_depth=10)

# Update the predict function for a probability-based approach
def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Next Day Greater?"])
    predictions = model.predict_proba(test[predictors])[:, 1]
    predictions = (predictions >= 0.5750).astype(int)
    return pd.DataFrame({
        'Actual': test['Next Day Greater?'],
        'Predictions': predictions
    }, index=test.index)

# Backtest the model with the new predictors
predictions = backtest(stock_data, model, predictors)

if not predictions.empty:
    print(predictions['Predictions'].value_counts())
    print("Precision Score:", precision_score(predictions['Actual'], predictions['Predictions']))
else:
    print("No predictions were generated. Try with a different stock or adjust parameters.")