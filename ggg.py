import ccxt
import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Tuple
from sklearn.linear_model import LinearRegression

exchange = ccxt.coinbase()

def fetch_order_book(symbol: str) -> dict:
    return exchange.fetch_order_book(symbol)

def fetch_current_price(symbol: str) -> float:
    try:
        ticker = exchange.fetch_ticker(symbol)
        current_price = ticker['last']
        print(f"Current price of {symbol}: {current_price}")
        return current_price
    except Exception as e:
        print(f"An error occurred while fetching the current price: {e}")
        return None

def fetch_historical_prices(symbol: str, limit: int = 100) -> np.ndarray:
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe='1m', limit=limit)
        prices = np.array([x[4] for x in ohlcv])  # Closing prices
        return prices
    except Exception as e:
        print(f"An error occurred while fetching historical prices: {e}")
        return np.array([])

def identify_stop_loss_clusters(order_book: dict, num_bins: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    bids = np.array(order_book['bids'])
    asks = np.array(order_book['asks'])

    bid_prices = bids[:, 0]
    bid_volumes = bids[:, 1]
    ask_prices = asks[:, 0]
    ask_volumes = asks[:, 1]

    bid_bin_edges = np.linspace(bid_prices.min(), bid_prices.max(), num_bins + 1)
    ask_bin_edges = np.linspace(ask_prices.min(), ask_prices.max(), num_bins + 1)

    bid_clusters, _ = np.histogram(bid_prices, bins=bid_bin_edges, weights=bid_volumes)
    ask_clusters, _ = np.histogram(ask_prices, bins=ask_bin_edges, weights=ask_volumes)

    return bid_clusters, ask_clusters, bid_bin_edges, ask_bin_edges

def print_clusters(clusters: np.ndarray, bin_edges: np.ndarray, cluster_type: str) -> None:
    print(f'\n{cluster_type} clusters:')
    for i in range(len(clusters)):
        print(f'Price level: {bin_edges[i]:.2f} - {bin_edges[i+1]:.2f}, Volume: {clusters[i]:.2f}')

def plot_clusters(bid_clusters: np.ndarray, ask_clusters: np.ndarray, bid_bin_edges: np.ndarray, ask_bin_edges: np.ndarray, current_price: float, predicted_price: float, bid_liquidity_price: float, ask_liquidity_price: float, direction: str) -> None:
    plt.clf()

    plt.subplot(2, 1, 1)
    plt.bar(bid_bin_edges[:-1], bid_clusters, width=np.diff(bid_bin_edges), edgecolor='black', align='edge')
    plt.title('Bid Clusters')
    plt.xlabel('Price')
    plt.ylabel('Volume')
    plt.axvline(x=current_price - 150, color='red', linestyle='--', label='-$150')
    plt.axvline(x=current_price + 150, color='green', linestyle='--', label='+$150')
    plt.axvline(x=predicted_price, color='blue', linestyle='--', label='Predicted Price')
    plt.axvline(x=bid_liquidity_price, color='purple', linestyle='-', linewidth=2, label='Bid Liquidity Cluster')

    if direction == 'LONG':
        plt.annotate('LONG', xy=(current_price, max(bid_clusters) / 2), xytext=(current_price + 20, max(bid_clusters) / 2 + 10),
                     arrowprops=dict(facecolor='green', shrink=0.05))

    plt.legend()

    plt.subplot(2, 1, 2)
    plt.bar(ask_bin_edges[:-1], ask_clusters, width=np.diff(ask_bin_edges), edgecolor='black', align='edge')
    plt.title('Ask Clusters')
    plt.xlabel('Price')
    plt.ylabel('Volume')
    plt.axvline(x=current_price - 150, color='red', linestyle='--', label='-$150')
    plt.axvline(x=current_price + 150, color='green', linestyle='--', label='+$150')
    plt.axvline(x=predicted_price, color='blue', linestyle='--', label='Predicted Price')
    plt.axvline(x=ask_liquidity_price, color='orange', linestyle='-', linewidth=2, label='Ask Liquidity Cluster')

    if direction == 'SHORT':
        plt.annotate('SHORT', xy=(current_price, max(ask_clusters) / 2), xytext=(current_price - 20, max(ask_clusters) / 2 + 10),
                     arrowprops=dict(facecolor='red', shrink=0.05))

    plt.legend()

    plt.tight_layout()
    plt.pause(0.1)

def calculate_volume_percentages(bid_clusters: np.ndarray, ask_clusters: np.ndarray) -> Tuple[float, float]:
    total_bid_volume = bid_clusters.sum()
    total_ask_volume = ask_clusters.sum()
    total_volume = total_bid_volume + total_ask_volume

    if total_volume == 0:
        return 0.0, 0.0

    bid_percentage = (total_bid_volume / total_volume) * 100
    ask_percentage = (total_ask_volume / total_volume) * 100

    return bid_percentage, ask_percentage

def predict_price(prices: np.ndarray) -> float:
    x = np.arange(len(prices)).reshape(-1, 1)
    y = prices.reshape(-1, 1)

    model = LinearRegression()
    model.fit(x, y)

    next_index = np.array([[len(prices)]])
    predicted_price = model.predict(next_index)
    
    return predicted_price[0][0]

def identify_liquidity_clusters(bid_clusters: np.ndarray, ask_clusters: np.ndarray, bid_bin_edges: np.ndarray, ask_bin_edges: np.ndarray) -> Tuple[float, float]:
    # Find the price levels with the highest liquidity
    max_bid_volume_index = np.argmax(bid_clusters)
    max_ask_volume_index = np.argmax(ask_clusters)

    bid_liquidity_price = bid_bin_edges[max_bid_volume_index]  # Price level with max bid volume
    ask_liquidity_price = ask_bin_edges[max_ask_volume_index]  # Price level with max ask volume

    return bid_liquidity_price, ask_liquidity_price

symbol = 'ETH/USD'
plt.ion()
fig = plt.figure(figsize=(14, 7))

while True:
    order_book = fetch_order_book(symbol)
    current_price = fetch_current_price(symbol)
    if current_price is None:
        continue

    historical_prices = fetch_historical_prices(symbol)

    if len(historical_prices) < 2:
        print("Not enough historical prices to make a prediction.")
        continue

    predicted_price = predict_price(historical_prices)

    bid_clusters, ask_clusters, bid_bin_edges, ask_bin_edges = identify_stop_loss_clusters(order_book)

    print_clusters(bid_clusters, bid_bin_edges, 'Bid')
    print_clusters(ask_clusters, ask_bin_edges, 'Ask')

    bid_percentage, ask_percentage = calculate_volume_percentages(bid_clusters, ask_clusters)
    print(f'\nBid volume percentage: {bid_percentage:.2f}%')
    print(f'Ask volume percentage: {ask_percentage:.2f}%')

    # Identify liquidity clusters
    bid_liquidity_price, ask_liquidity_price = identify_liquidity_clusters(bid_clusters, ask_clusters, bid_bin_edges, ask_bin_edges)
    print(f'\nLiquidity cluster (Bid): {bid_liquidity_price:.2f}')
    print(f'Liquidity cluster (Ask): {ask_liquidity_price:.2f}')

    # Predict price direction
    if current_price <= bid_liquidity_price:
        direction = 'LONG'
    elif current_price >= ask_liquidity_price:
        direction = 'SHORT'
    else:
        direction = 'NEUTRAL'
    
    print(f'Predicted direction: {direction}')

    plot_clusters(bid_clusters, ask_clusters, bid_bin_edges, ask_bin_edges, current_price, predicted_price, bid_liquidity_price, ask_liquidity_price, direction)

    time.sleep(5)
