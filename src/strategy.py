"""
Mean-reversion strategy for cryptocurrency statistical arbitrage.
Generates entry signals based on price drops + volume spikes.
"""

import pandas as pd
import numpy as np
import os
import glob
from typing import List, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MeanReversionStrategy:
    """
    Short-term mean-reversion strategy that identifies oversold conditions.
    
    Entry Signal:
    - Price drops > 2% in 1 hour AND
    - Volume > 1.5x recent (24h) average
    """
    
    def __init__(self, data_dir: str = 'data', results_dir: str = 'results'):
        """Initialize strategy with data and results directories."""
        self.data_dir = data_dir
        self.results_dir = results_dir
        
        # Strategy parameters (as per PROJECT_SPEC.md)
        self.price_drop_threshold = -0.02  # -2% price drop
        self.volume_ratio_threshold = 1.5   # 1.5x volume spike
        self.volume_lookback_hours = 24     # 24-hour rolling average
        
        # Ensure results directory exists
        os.makedirs(self.results_dir, exist_ok=True)
    
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load all cryptocurrency data from CSV files.
        
        Returns:
            Dictionary mapping symbol names to DataFrames
        """
        data = {}
        
        # Find all CSV files in data directory
        csv_files = glob.glob(os.path.join(self.data_dir, "*_hourly.csv"))
        
        if not csv_files:
            raise ValueError(f"No hourly CSV files found in {self.data_dir}")
        
        logger.info(f"Loading data from {len(csv_files)} files...")
        
        for filepath in csv_files:
            # Extract symbol from filename (e.g., "BTC_USD_hourly.csv" -> "BTC_USD")
            filename = os.path.basename(filepath)
            symbol = filename.replace("_hourly.csv", "")
            
            try:
                df = pd.read_csv(filepath)
                
                # Convert datetime column and set as index
                df['datetime'] = pd.to_datetime(df['datetime'])
                df = df.set_index('datetime').sort_index()
                
                # Validate required columns
                required_cols = ['open', 'high', 'low', 'close', 'volume']
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    logger.error(f"Missing columns in {symbol}: {missing_cols}")
                    continue
                
                data[symbol] = df
                logger.info(f"Loaded {symbol}: {len(df)} records from {df.index[0]} to {df.index[-1]}")
                
            except Exception as e:
                logger.error(f"Error loading {filepath}: {e}")
                continue
        
        if not data:
            raise ValueError("No valid data files loaded")
            
        return data
    
    def calculate_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate 1-hour returns (percentage change).
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with additional 'return_1h' column
        """
        # Calculate 1-hour return using close prices
        df = df.copy()
        df['return_1h'] = df['close'].pct_change()
        
        return df
    
    def calculate_volume_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volume metrics: rolling average and ratio.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with additional volume columns
        """
        df = df.copy()
        
        # Calculate 24-hour rolling average volume
        df['volume_24h_avg'] = df['volume'].rolling(
            window=self.volume_lookback_hours, 
            min_periods=1
        ).mean()
        
        # Calculate volume ratio (current / average)
        df['volume_ratio'] = df['volume'] / df['volume_24h_avg']
        
        # Handle division by zero or NaN
        df['volume_ratio'] = df['volume_ratio'].fillna(1.0)
        df.loc[df['volume_24h_avg'] == 0, 'volume_ratio'] = 1.0
        
        return df
    
    def generate_signals(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Generate entry signals based on price drop + volume spike conditions.
        
        Args:
            df: DataFrame with OHLCV and calculated metrics
            symbol: Symbol name for logging
            
        Returns:
            DataFrame with additional 'signal' column
        """
        df = df.copy()
        
        # Entry signal: price drop >= 2% AND volume ratio >= 1.5x
        price_drop_condition = df['return_1h'] <= self.price_drop_threshold
        volume_spike_condition = df['volume_ratio'] >= self.volume_ratio_threshold
        
        # Combined signal (both conditions must be true)
        df['signal'] = (price_drop_condition & volume_spike_condition).astype(int)
        
        # Add symbol column for combined analysis
        df['symbol'] = symbol
        
        # Log signal statistics
        total_signals = df['signal'].sum()
        signal_rate = total_signals / len(df) * 100
        
        logger.info(f"{symbol}: {total_signals} signals ({signal_rate:.2f}% of hours)")
        
        return df
    
    def process_symbol(self, symbol: str, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process a single symbol through the complete strategy pipeline.
        
        Args:
            symbol: Symbol name
            df: Raw OHLCV data
            
        Returns:
            DataFrame with all calculated metrics and signals
        """
        logger.info(f"Processing {symbol}...")
        
        # Step 1: Calculate returns
        df = self.calculate_returns(df)
        
        # Step 2: Calculate volume metrics
        df = self.calculate_volume_metrics(df)
        
        # Step 3: Generate signals
        df = self.generate_signals(df, symbol)
        
        return df
    
    def run_strategy(self) -> pd.DataFrame:
        """
        Run the complete strategy on all symbols.
        
        Returns:
            Combined DataFrame with signals for all symbols
        """
        logger.info("Starting mean-reversion strategy...")
        
        # Load all data
        data = self.load_data()
        
        # Process each symbol
        all_results = []
        
        for symbol, df in data.items():
            try:
                result_df = self.process_symbol(symbol, df)
                
                # Select relevant columns for final output
                output_cols = [
                    'close', 'volume', 'return_1h', 
                    'volume_24h_avg', 'volume_ratio', 'signal', 'symbol'
                ]
                result_df = result_df[output_cols].copy()
                
                all_results.append(result_df)
                
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                continue
        
        if not all_results:
            raise ValueError("No symbols processed successfully")
        
        # Combine all results
        combined_df = pd.concat(all_results, axis=0).sort_index()
        
        return combined_df
    
    def print_summary_statistics(self, results_df: pd.DataFrame) -> None:
        """Print summary statistics of the strategy results."""
        logger.info("=" * 60)
        logger.info("STRATEGY SUMMARY STATISTICS")
        logger.info("=" * 60)
        
        # Overall statistics
        total_hours = len(results_df)
        total_signals = results_df['signal'].sum()
        signal_rate = total_signals / total_hours * 100
        
        logger.info(f"Total data points: {total_hours:,}")
        logger.info(f"Total signals: {total_signals}")
        logger.info(f"Signal rate: {signal_rate:.2f}%")
        
        # Date range
        start_date = results_df.index.min().strftime('%Y-%m-%d')
        end_date = results_df.index.max().strftime('%Y-%m-%d')
        logger.info(f"Date range: {start_date} to {end_date}")
        
        # Per-symbol breakdown
        logger.info("-" * 40)
        logger.info("PER-SYMBOL BREAKDOWN:")
        
        for symbol in results_df['symbol'].unique():
            symbol_data = results_df[results_df['symbol'] == symbol]
            symbol_signals = symbol_data['signal'].sum()
            symbol_rate = symbol_signals / len(symbol_data) * 100
            
            # Average metrics for signals
            signal_data = symbol_data[symbol_data['signal'] == 1]
            if len(signal_data) > 0:
                avg_return = signal_data['return_1h'].mean() * 100
                avg_volume_ratio = signal_data['volume_ratio'].mean()
                logger.info(f"{symbol:>8}: {symbol_signals:>3} signals ({symbol_rate:>5.2f}%) | "
                          f"Avg return: {avg_return:>6.2f}% | Avg vol ratio: {avg_volume_ratio:>4.1f}x")
            else:
                logger.info(f"{symbol:>8}: {symbol_signals:>3} signals ({symbol_rate:>5.2f}%)")
        
        # Signal timing distribution
        logger.info("-" * 40)
        logger.info("SIGNAL TIMING:")
        
        if total_signals > 0:
            signal_hours = results_df[results_df['signal'] == 1]
            
            # Hour of day distribution
            hour_counts = signal_hours.index.hour.value_counts().sort_index()
            top_hours = hour_counts.nlargest(3)
            
            logger.info("Most active hours (UTC):")
            for hour, count in top_hours.items():
                percentage = count / total_signals * 100
                logger.info(f"  {hour:02d}:00 - {count} signals ({percentage:.1f}%)")
        
        logger.info("=" * 60)
    
    def save_results(self, results_df: pd.DataFrame, filename: str = 'signals.csv') -> str:
        """Save results to CSV file."""
        filepath = os.path.join(self.results_dir, filename)
        
        # Reset index to save datetime as column
        save_df = results_df.reset_index()
        save_df.to_csv(filepath, index=False)
        
        logger.info(f"Results saved to {filepath}")
        return filepath


def main():
    """Main function to run the mean-reversion strategy."""
    try:
        # Initialize strategy
        strategy = MeanReversionStrategy()
        
        # Run the complete strategy
        results = strategy.run_strategy()
        
        # Print summary statistics
        strategy.print_summary_statistics(results)
        
        # Save results
        strategy.save_results(results)
        
        logger.info("✅ Strategy analysis complete!")
        
        # Quick validation
        total_signals = results['signal'].sum()
        if total_signals == 0:
            logger.warning("⚠️  No signals generated - consider adjusting thresholds")
        elif total_signals < 10:
            logger.warning("⚠️  Very few signals - may need parameter tuning")
        else:
            logger.info(f"🎯 Generated {total_signals} entry signals for backtesting")
            
    except Exception as e:
        logger.error(f"Strategy failed: {e}")
        raise


if __name__ == "__main__":
    main()