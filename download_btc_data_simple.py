"""
Simple Bitcoin 5-minute OHLCV data downloader using CCXT

This is a simplified version that focuses on Binance API.
"""

import ccxt
import pandas as pd
from datetime import datetime, timedelta
import time
import os


def download_from_exchange(exchange_name, years=5):
    """Download Bitcoin data from specified exchange."""
    print(f"Connecting to {exchange_name}...")
    
    # Get exchange class
    exchange_class = getattr(ccxt, exchange_name)
    
    # Initialize exchange with common settings
    exchange_config = {
        'enableRateLimit': True,
    }
    
    # Add exchange-specific options
    if exchange_name == 'binance':
        exchange_config['options'] = {'defaultType': 'spot'}
    elif exchange_name == 'coinbasepro':
        exchange_config['options'] = {'sandbox': False}
    
    exchange = exchange_class(exchange_config)
    
    # Load markets
    print("Loading markets...")
    try:
        exchange.load_markets()
    except Exception as e:
        print(f"Warning: Could not load markets: {e}")
        raise
    
    # Determine symbol and timeframe
    symbol = 'BTC/USDT'
    if exchange_name == 'coinbasepro':
        symbol = 'BTC/USD'
    elif exchange_name == 'kraken':
        symbol = 'BTC/USD'
    
    # Check available timeframes
    timeframe = '5m'
    if hasattr(exchange, 'timeframes') and exchange.timeframes:
        if '5m' not in exchange.timeframes:
            # Try alternatives
            alternatives = ['5m', '1m', '15m']
            for alt in alternatives:
                if alt in exchange.timeframes:
                    timeframe = alt
                    break
            else:
                # Use first available timeframe
                timeframe = list(exchange.timeframes.keys())[0]
                print(f"Using {timeframe} instead of 5m")
    
    print(f"Downloading {symbol} data with {timeframe} timeframe...")
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)
    since = int(start_date.timestamp() * 1000)
    
    print(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    all_ohlcv = []
    current_since = since
    batch_count = 0
    max_iterations = 10000  # Safety limit
    retry_count = 0
    max_retries = 5
    
    print("Downloading data in batches...")
    
    while current_since < int(end_date.timestamp() * 1000) and batch_count < max_iterations:
        try:
            # Fetch OHLCV data (most exchanges allow up to 1000 candles per request)
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=current_since, limit=1000)
            
            if not ohlcv or len(ohlcv) == 0:
                print("No more data available.")
                break
            
            # Add to collection
            all_ohlcv.extend(ohlcv)
            
            # Update timestamp for next batch
            last_timestamp = ohlcv[-1][0]
            if last_timestamp <= current_since:
                # No progress, break to avoid infinite loop
                print("No progress in download, stopping.")
                break
            current_since = last_timestamp + 1
            
            # Reset retry count on success
            retry_count = 0
            
            batch_count += 1
            if batch_count % 10 == 0:
                latest_date = datetime.fromtimestamp(ohlcv[-1][0] / 1000)
                progress_pct = ((current_since - since) / (int(end_date.timestamp() * 1000) - since)) * 100
                print(f"Downloaded {len(all_ohlcv)} candles... Latest: {latest_date.strftime('%Y-%m-%d %H:%M:%S')} ({progress_pct:.1f}% complete)")
            
            # Rate limiting - adjust based on exchange
            time.sleep(0.2)  # Slightly slower to avoid timeouts
            
        except ccxt.NetworkError as e:
            retry_count += 1
            if retry_count <= max_retries:
                wait_time = min(5 * retry_count, 30)  # Exponential backoff, max 30 seconds
                print(f"Network error (attempt {retry_count}/{max_retries}): {e}. Waiting {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"Max retries reached. Continuing with downloaded data so far...")
                break
        except ccxt.ExchangeError as e:
            error_msg = str(e).lower()
            # Check if it's a timeout error
            if 'timeout' in error_msg or 'timed out' in error_msg or '51054' in str(e):
                retry_count += 1
                if retry_count <= max_retries:
                    wait_time = min(10 * retry_count, 60)  # Longer wait for timeouts
                    print(f"Timeout error (attempt {retry_count}/{max_retries}): {e}")
                    print(f"Waiting {wait_time} seconds before retrying...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"Max retries reached for timeout. Continuing with downloaded data so far...")
                    print(f"Downloaded {len(all_ohlcv)} candles up to {datetime.fromtimestamp(current_since/1000).strftime('%Y-%m-%d %H:%M:%S')}")
                    break
            else:
                print(f"Exchange error: {e}")
                # For non-timeout errors, continue if we have some data
                if len(all_ohlcv) > 0:
                    print(f"Continuing with downloaded data so far...")
                    break
                else:
                    raise
        except Exception as e:
            retry_count += 1
            if retry_count <= max_retries:
                wait_time = min(5 * retry_count, 30)
                print(f"Unexpected error (attempt {retry_count}/{max_retries}): {e}. Waiting {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"Max retries reached. Error: {e}")
                if len(all_ohlcv) > 0:
                    print(f"Continuing with downloaded data so far...")
                    break
                else:
                    import traceback
                    traceback.print_exc()
                    raise
    
    if len(all_ohlcv) == 0:
        raise ValueError("No data downloaded!")
    
    print(f"\nTotal candles downloaded: {len(all_ohlcv)}")
    
    # Check if we got the full date range
    if current_since < int(end_date.timestamp() * 1000):
        remaining_days = (int(end_date.timestamp() * 1000) - current_since) / (1000 * 60 * 60 * 24)
        print(f"Warning: Download incomplete. Still need {remaining_days:.0f} days of data.")
        print(f"Last downloaded: {datetime.fromtimestamp(current_since/1000).strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Target end date: {end_date.strftime('%Y-%m-%d %H:%M:%S')}")
        print("You can run the script again to continue downloading, or use the partial dataset.")
    
    # Convert to DataFrame
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    # Convert timestamp from milliseconds to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Remove duplicates and sort
    df = df.drop_duplicates(subset=['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Remove any future timestamps
    df = df[df['timestamp'] <= datetime.now()]
    
    print(f"Final dataset: {len(df)} candles")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    return df


def main():
    """Main function."""
    print("=" * 60)
    print("Bitcoin OHLCV Data Downloader")
    print("=" * 60)
    print()
    
    output_file = 'btc_5m_5years.csv'
    existing_df = None
    
    # Check if file already exists and try to resume
    if os.path.exists(output_file):
        try:
            print(f"Found existing file: {output_file}")
            existing_df = pd.read_csv(output_file)
            existing_df['timestamp'] = pd.to_datetime(existing_df['timestamp'])
            print(f"  Existing data: {len(existing_df)} candles")
            print(f"  Date range: {existing_df['timestamp'].min()} to {existing_df['timestamp'].max()}")
            
            # Check if we need more data
            latest_timestamp = existing_df['timestamp'].max()
            target_end = datetime.now()
            days_needed = (target_end - latest_timestamp).days
            
            if days_needed < 30:
                print(f"\n✓ File already has recent data (only {days_needed} days old).")
                print("  Using existing file. Delete it if you want to re-download.")
                return 0
            else:
                print(f"\n  File is {days_needed} days old. Will continue downloading...")
        except Exception as e:
            print(f"  Warning: Could not read existing file: {e}")
            print("  Will download fresh data...")
    
    # Try multiple exchanges in order of preference
    exchanges_to_try = [
        'bybit',      # Usually works from most locations
        'okx',        # OKX (formerly OKEx)
        'kraken',     # Kraken
        'coinbasepro', # Coinbase Pro
        'bitfinex',   # Bitfinex
        'gate',       # Gate.io
        'huobi',      # Huobi
        'binance',    # Binance (last resort due to geo restrictions)
    ]
    
    df = None
    last_error = None
    
    for exchange_name in exchanges_to_try:
        try:
            print(f"\n{'='*60}")
            print(f"Trying {exchange_name}...")
            print(f"{'='*60}\n")
            df = download_from_exchange(exchange_name, years=5)
            if df is not None and len(df) > 0:
                print(f"\n✓ Successfully downloaded data from {exchange_name}!")
                break
        except ccxt.ExchangeNotAvailable as e:
            print(f"✗ {exchange_name} not available: {e}")
            last_error = e
            continue
        except ccxt.NetworkError as e:
            print(f"✗ Network error with {exchange_name}: {e}")
            last_error = e
            continue
        except Exception as e:
            print(f"✗ Error with {exchange_name}: {e}")
            last_error = e
            import traceback
            traceback.print_exc()
            continue
    
    if df is None or len(df) == 0:
        print(f"\n✗ Could not download data from any exchange.")
        if last_error:
            print(f"Last error: {last_error}")
        # If we have existing data, use that
        if existing_df is not None and len(existing_df) > 0:
            print(f"\nUsing existing data file instead...")
            df = existing_df
        else:
            return 1
    
    # Merge with existing data if available
    if existing_df is not None and len(existing_df) > 0:
        print(f"\nMerging with existing data...")
        # Combine and remove duplicates
        combined_df = pd.concat([existing_df, df], ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=['timestamp'])
        combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
        df = combined_df
        print(f"  Combined dataset: {len(df)} candles")
        print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    try:
        
        # Save to CSV
        print(f"\nSaving data to {output_file}...")
        df.to_csv(output_file, index=False)
        
        file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
        print(f"\n✓ Successfully saved {len(df)} candles to {output_file}")
        print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"  File size: {file_size_mb:.2f} MB")
        print("\nYou can now use this file with train_patchtst.py")
        
    except Exception as e:
        print(f"\n✗ Error saving file: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

