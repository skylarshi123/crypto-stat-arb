"""
Data fetcher for cryptocurrency historical data using CCXT.
Fetches 2 years of hourly OHLCV data for statistical arbitrage strategy.
"""

import ccxt
import pandas as pd
import numpy as np
import time
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CryptoDataFetcher:
    """Fetches and validates cryptocurrency historical data from Kraken."""
    
    def __init__(self, exchange_name: str = 'kraken'):
        """Initialize the data fetcher with exchange connection."""
        self.exchange = getattr(ccxt, exchange_name)({
            'apiKey': '',  # Not needed for public data
            'secret': '',
            'enableRateLimit': True,  # Built-in rate limiting
            'sandbox': False,
        })
        
        # Symbols to fetch (adapted for Kraken availability)
        self.symbols = ['BTC/USD', 'ETH/USD', 'SOL/USD', 'AVAX/USD']
        
        # Data directory
        self.data_dir = 'data'
        os.makedirs(self.data_dir, exist_ok=True)
    
    def calculate_date_range(self) -> tuple:
        """Calculate start and end dates for available data."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=60)  # 60 days (what Kraken provides)
        
        # Convert to timestamps (milliseconds for CCXT)
        start_timestamp = int(start_date.timestamp() * 1000)
        end_timestamp = int(end_date.timestamp() * 1000)
        
        logger.info(f"Fetching data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        return start_timestamp, end_timestamp
    
    def fetch_ohlcv_data(self, symbol: str, start_timestamp: int, 
                    end_timestamp: int, timeframe: str = '1h') -> pd.DataFrame:
        """
        Fetch OHLCV data for a symbol between start and end timestamps.
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Timeframe for candles ('1h' for hourly)
            start_timestamp: Start timestamp in milliseconds
            end_timestamp: End timestamp in milliseconds
            
        Returns:
            DataFrame with OHLCV data
        """
        all_data = []
        current_timestamp = start_timestamp
        
        # Progress bar for this symbol
        total_hours = (end_timestamp - start_timestamp) // (1000 * 60 * 60)
        pbar = tqdm(total=total_hours, desc=f"Fetching {symbol}", unit="hours")
        
        while current_timestamp < end_timestamp:
            try:
                # Fetch data in chunks (max 1000 candles per request for Binance)
                ohlcv = self.exchange.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    since=current_timestamp,
                    limit=1000
                )
                
                if not ohlcv:
                    logger.warning(f"No data received for {symbol} at timestamp {current_timestamp}")
                    break
                
                all_data.extend(ohlcv)
                
                # Update timestamp for next batch
                last_timestamp = ohlcv[-1][0]
                current_timestamp = last_timestamp + (60 * 60 * 1000)  # Add 1 hour
                
                # Update progress bar
                hours_fetched = len(ohlcv)
                pbar.update(hours_fetched)
                
                # Rate limiting - sleep between requests
                time.sleep(0.1)  # 100ms delay
                
            except ccxt.NetworkError as e:
                logger.error(f"Network error for {symbol}: {e}")
                time.sleep(5)  # Wait 5 seconds before retry
                continue
                
            except ccxt.ExchangeError as e:
                logger.error(f"Exchange error for {symbol}: {e}")
                time.sleep(2)
                continue
                
            except Exception as e:
                logger.error(f"Unexpected error for {symbol}: {e}")
                break
        
        pbar.close()
        
        # Convert to DataFrame
        if not all_data:
            logger.error(f"No data fetched for {symbol}")
            return pd.DataFrame()
        
        df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('datetime')
        
        # Remove duplicates based on timestamp
        df = df.drop_duplicates(subset=['timestamp'])
        
        logger.info(f"Fetched {len(df)} candles for {symbol}")
        return df
    
    def validate_data(self, df: pd.DataFrame, symbol: str) -> bool:
        """
        Validate fetched data for quality and completeness.
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Symbol name for logging
            
        Returns:
            True if data passes validation, False otherwise
        """
        if df.empty:
            logger.error(f"No data for {symbol}")
            return False
        
        # Check for missing values
        if df.isnull().any().any():
            logger.warning(f"Missing values found in {symbol} data")
        
        # Check for non-positive prices
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if (df[col] <= 0).any():
                logger.error(f"Non-positive prices found in {symbol} {col} column")
                return False
        
        # Check for negative volume
        if (df['volume'] < 0).any():
            logger.error(f"Negative volume found in {symbol}")
            return False
        
        # Check for reasonable price ranges (high >= low, etc.)
        if (df['high'] < df['low']).any():
            logger.error(f"Invalid OHLC data: high < low in {symbol}")
            return False
            
        if (df['high'] < df['open']).any() or (df['high'] < df['close']).any():
            logger.error(f"Invalid OHLC data: high < open/close in {symbol}")
            return False
            
        if (df['low'] > df['open']).any() or (df['low'] > df['close']).any():
            logger.error(f"Invalid OHLC data: low > open/close in {symbol}")
            return False
        
        # Check data continuity (no large gaps)
        df_sorted = df.sort_values('timestamp')
        time_diffs = df_sorted['timestamp'].diff().dropna()
        expected_diff = 60 * 60 * 1000  # 1 hour in milliseconds
        
        large_gaps = time_diffs > expected_diff * 2  # Allow some tolerance
        if large_gaps.any():
            gap_count = large_gaps.sum()
            logger.warning(f"Found {gap_count} large time gaps in {symbol} data")
        
        logger.info(f"Data validation passed for {symbol}: {len(df)} records")
        return True
    
    def save_to_csv(self, df: pd.DataFrame, symbol: str) -> str:
        """Save DataFrame to CSV file."""
        # Clean symbol name for filename
        clean_symbol = symbol.replace('/', '_')
        filename = f"{clean_symbol}_hourly.csv"
        filepath = os.path.join(self.data_dir, filename)
        
        # Save with timestamp as column (not index) for easier loading
        df_save = df.reset_index()
        df_save.to_csv(filepath, index=False)
        
        logger.info(f"Saved {len(df)} records to {filepath}")
        return filepath
    
    def fetch_all_symbols(self) -> Dict[str, str]:
        """
        Fetch data for all symbols and save to CSV files.
        
        Returns:
            Dictionary mapping symbols to their CSV file paths
        """
        start_timestamp, end_timestamp = self.calculate_date_range()
        results = {}
        
        logger.info(f"Starting data fetch for {len(self.symbols)} symbols")
        
        for symbol in self.symbols:
            logger.info(f"Processing {symbol}...")
            
            # Check if file already exists
            clean_symbol = symbol.replace('/', '_')
            filename = f"{clean_symbol}_hourly.csv"
            filepath = os.path.join(self.data_dir, filename)
            
            if os.path.exists(filepath):
                logger.info(f"File {filepath} already exists. Loading existing data...")
                try:
                    existing_df = pd.read_csv(filepath)
                    if len(existing_df) > 0:
                        logger.info(f"Using existing data for {symbol}: {len(existing_df)} records")
                        results[symbol] = filepath
                        continue
                except Exception as e:
                    logger.warning(f"Error reading existing file {filepath}: {e}")
            
            # Fetch new data
            try:
                df = self.fetch_ohlcv_data(symbol, start_timestamp, end_timestamp)
                
                if not df.empty and self.validate_data(df, symbol):
                    filepath = self.save_to_csv(df, symbol)
                    results[symbol] = filepath
                else:
                    logger.error(f"Failed to fetch valid data for {symbol}")
                    
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                continue
            
            # Pause between symbols to be respectful to the API
            time.sleep(1)
        
        return results
    
    def print_summary(self, results: Dict[str, str]) -> None:
        """Print summary statistics of fetched data."""
        logger.info("=" * 60)
        logger.info("DATA FETCH SUMMARY")
        logger.info("=" * 60)
        
        total_records = 0
        
        for symbol, filepath in results.items():
            if os.path.exists(filepath):
                df = pd.read_csv(filepath)
                records = len(df)
                total_records += records
                
                if records > 0:
                    start_date = pd.to_datetime(df['datetime'].iloc[0]).strftime('%Y-%m-%d')
                    end_date = pd.to_datetime(df['datetime'].iloc[-1]).strftime('%Y-%m-%d')
                    logger.info(f"{symbol:>10}: {records:>6,} records ({start_date} to {end_date})")
                else:
                    logger.info(f"{symbol:>10}: {records:>6,} records (EMPTY)")
        
        logger.info("-" * 60)
        logger.info(f"{'TOTAL':>10}: {total_records:>6,} records")
        logger.info(f"Target for strategy: ~2,880 records (60 days × 24 hours × 2 months)")
        
        if total_records > 2500:  # Allow some tolerance for 60 days
            logger.info("✅ Sufficient data for statistical arbitrage strategy")
        else:
            logger.warning("⚠️  May need more data for robust backtesting")


def main():
    """Main function to fetch all cryptocurrency data."""
    fetcher = CryptoDataFetcher()
    
    logger.info("Starting cryptocurrency data fetch...")
    logger.info(f"Target symbols: {fetcher.symbols}")
    
    # Fetch all data
    results = fetcher.fetch_all_symbols()
    
    # Print summary
    fetcher.print_summary(results)
    
    if len(results) == len(fetcher.symbols):
        logger.info("🎉 All symbols fetched successfully!")
    else:
        logger.warning(f"⚠️  Only {len(results)}/{len(fetcher.symbols)} symbols fetched successfully")
    
    logger.info("Data fetch complete. Files saved in 'data/' directory.")


if __name__ == "__main__":
    main()