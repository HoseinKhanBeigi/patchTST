"""
Download Bitcoin 5-minute OHLCV data using CCXT

This script downloads 5 years of Bitcoin 5-minute candlestick data
and saves it to btc_5m_5years.csv for use with the PatchTST training script.
"""

import ccxt
import pandas as pd
from datetime import datetime, timedelta
import time
import os


def download_btc_data(exchange_name='binance', timeframe='5m', years=5):
    """
    Download Bitcoin OHLCV data from a cryptocurrency exchange.
    
    Args:
        exchange_name: Name of the exchange (default: 'binance')
        timeframe: Candlestick timeframe (default: '5m' for 5 minutes)
        years: Number of years of historical data to download (default: 5)
        
    Returns:
        DataFrame with OHLCV data
    """
    # Initialize exchange
    print(f"Connecting to {exchange_name}...")
    exchange_class = getattr(ccxt, exchange_name)
    exchange = exchange_class({
        'enableRateLimit': True,
        'options': {
            'defaultType': 'spot',  # Use spot market
        }
    })
    
    # Load markets first (required for some exchanges)
    print("Loading markets...")
    try:
        exchange.load_markets()
    except Exception as e:
        print(f"Warning: Could not load markets: {e}")
        # Some exchanges work without explicit load_markets
    
    # Check if exchange supports the timeframe
    if exchange.timeframes is None or not isinstance(exchange.timeframes, dict):
        print("Warning: Exchange timeframes not available. Proceeding with default timeframe '5m'.")
    else:
        if timeframe not in exchange.timeframes:
            print(f"Warning: {timeframe} not available. Available timeframes: {list(exchange.timeframes.keys())[:10]}")
            # Try to use a similar timeframe
            if '5m' in exchange.timeframes:
                timeframe = '5m'
            elif len(exchange.timeframes) > 0:
                timeframe = list(exchange.timeframes.keys())[0]
                print(f"Using {timeframe} instead.")
    
    # Symbol for Bitcoin (usually BTC/USDT or BTC/USD)
    symbol = 'BTC/USDT'
    if not hasattr(exchange, 'markets') or exchange.markets is None or len(exchange.markets) == 0:
        # Try to load markets if not loaded
        try:
            exchange.load_markets()
        except:
            pass
    
    # Check if symbol exists in markets
    if hasattr(exchange, 'markets') and exchange.markets and symbol not in exchange.markets:
        # Try alternative symbols
        alternatives = ['BTC/USD', 'BTC/USDC', 'BTC/BUSD', 'BTC/EUR']
        symbol_found = False
        for alt_symbol in alternatives:
            if exchange.markets and alt_symbol in exchange.markets:
                symbol = alt_symbol
                symbol_found = True
                break
        
        if not symbol_found:
            # Try to find any BTC pair
            if exchange.markets:
                btc_pairs = [s for s in exchange.markets.keys() if s.startswith('BTC/')]
                if btc_pairs:
                    symbol = btc_pairs[0]
                    print(f"Using {symbol} instead of BTC/USDT")
                else:
                    raise ValueError(f"Could not find BTC trading pair. Available pairs: {list(exchange.markets.keys())[:10]}")
            else:
                # If markets still not loaded, try common symbols
                print("Markets not loaded, trying common symbol...")
    
    print(f"Downloading {symbol} data with {timeframe} timeframe...")
    
    # Calculate start date (5 years ago)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)
    
    print(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Convert to milliseconds timestamp
    since = int(start_date.timestamp() * 1000)
    
    all_ohlcv = []
    current_since = since
    batch_count = 0
    
    print("Downloading data in batches...")
    
    while current_since < int(end_date.timestamp() * 1000):
        try:
            # Fetch OHLCV data
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=current_since, limit=1000)
            
            if not ohlcv:
                print("No more data available.")
                break
            
            all_ohlcv.extend(ohlcv)
            
            # Update since timestamp to the last candle's timestamp + 1
            current_since = ohlcv[-1][0] + 1
            
            batch_count += 1
            if batch_count % 10 == 0:
                latest_date = datetime.fromtimestamp(ohlcv[-1][0] / 1000)
                print(f"Downloaded {len(all_ohlcv)} candles... Latest: {latest_date.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Rate limiting
            time.sleep(exchange.rateLimit / 1000)
            
        except ccxt.NetworkError as e:
            print(f"Network error: {e}. Retrying...")
            time.sleep(5)
        except ccxt.ExchangeError as e:
            print(f"Exchange error: {e}")
            break
        except Exception as e:
            print(f"Unexpected error: {e}")
            break
    
    print(f"\nTotal candles downloaded: {len(all_ohlcv)}")
    
    # Convert to DataFrame
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    # Convert timestamp from milliseconds to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Remove duplicates and sort
    df = df.drop_duplicates(subset=['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Remove any future timestamps (just in case)
    df = df[df['timestamp'] <= datetime.now()]
    
    print(f"Final dataset: {len(df)} candles")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    return df


def main():
    """
    Main function to download and save Bitcoin data.
    """
    print("=" * 60)
    print("Bitcoin OHLCV Data Downloader")
    print("=" * 60)
    print()
    
    # Try multiple exchanges in order of preference
    exchanges_to_try = ['binance', 'coinbasepro', 'kraken', 'bitfinex', 'okx', 'bybit']
    
    df = None
    last_error = None
    for exchange_name in exchanges_to_try:
        try:
            print(f"\nAttempting to use {exchange_name}...")
            df = download_btc_data(exchange_name=exchange_name, timeframe='5m', years=5)
            if df is not None and len(df) > 0:
                break
        except Exception as e:
            last_error = e
            print(f"Failed to use {exchange_name}: {e}")
            import traceback
            traceback.print_exc()
            print("Trying next exchange...")
            continue
    
    if df is None or len(df) == 0:
        print("\nError: Could not download data from any exchange.")
        return
    
    # Save to CSV
    output_file = 'btc_5m_5years.csv'
    print(f"\nSaving data to {output_file}...")
    df.to_csv(output_file, index=False)
    
    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"\n✓ Successfully saved {len(df)} candles to {output_file}")
    print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"  File size: {file_size_mb:.2f} MB")
    print("\nYou can now use this file with train_patchtst.py")


if __name__ == "__main__":
    main()

