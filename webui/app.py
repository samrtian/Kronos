import os
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
import plotly.utils
from flask import Flask, render_template, request, jsonify, Response
from flask_cors import CORS
import sys
import warnings
import datetime
import time
import threading
import uuid
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

# Try to import akshare
try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
except ImportError:
    AKSHARE_AVAILABLE = False
    print("Warning: akshare not available, stock code query will be disabled")

# Add project root directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from model import Kronos, KronosTokenizer, KronosPredictor
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False
    print("Warning: Kronos model cannot be imported, will use simulated data for demonstration")

app = Flask(__name__)
CORS(app)

# Global variables to store models
tokenizer = None
model = None
predictor = None

# Progress tracking dictionary
progress_tracker = {}

def update_progress(task_id, progress, status, message):
    """Update progress for a task"""
    progress_tracker[task_id] = {
        'progress': progress,
        'status': status,
        'message': message,
        'timestamp': datetime.datetime.now().isoformat()
    }

def run_prediction_async(task_id, params):
    """Run prediction asynchronously with progress updates"""
    try:
        update_progress(task_id, 0, 'starting', 'Initializing prediction...')

        # Extract parameters
        file_path = params.get('file_path')
        stock_code = params.get('stock_code')
        lookback = params.get('lookback')
        pred_len = params.get('pred_len')
        temperature = params.get('temperature')
        top_p = params.get('top_p')
        sample_count = params.get('sample_count')
        apply_limits = params.get('apply_limits')
        limit_rate = params.get('limit_rate')
        start_date = params.get('start_date')

        update_progress(task_id, 10, 'loading', 'Loading data...')

        # Load data
        if stock_code:
            df, error = load_stock_data_akshare(stock_code, lookback=lookback+pred_len+50)
            if error:
                update_progress(task_id, 100, 'error', error)
                return
            data_source = f"Stock {stock_code} (akshare)"
        elif file_path:
            df, error = load_data_file(file_path)
            if error:
                update_progress(task_id, 100, 'error', error)
                return
            data_source = file_path
        else:
            update_progress(task_id, 100, 'error', 'No data source provided')
            return

        update_progress(task_id, 20, 'preparing', 'Preparing data for prediction...')

        if len(df) < lookback:
            update_progress(task_id, 100, 'error', f'Insufficient data length, need at least {lookback} rows')
            return

        # Perform prediction
        if MODEL_AVAILABLE and predictor is not None:
            try:
                required_cols = ['open', 'high', 'low', 'close']
                if 'volume' in df.columns:
                    required_cols.append('volume')

                update_progress(task_id, 30, 'processing', 'Preparing input data...')

                # Process time period selection (same logic as original predict function)
                if stock_code:
                    if len(df) < lookback:
                        update_progress(task_id, 100, 'error', f'Insufficient data, need at least {lookback} data points')
                        return

                    x_df = df.iloc[-lookback:][required_cols]
                    x_timestamp = df.iloc[-lookback:]['timestamps']

                    last_timestamp = df['timestamps'].iloc[-1]
                    time_diff = df['timestamps'].iloc[-1] - df['timestamps'].iloc[-2] if len(df) > 1 else pd.Timedelta(days=1)
                    y_timestamp = pd.date_range(
                        start=last_timestamp + time_diff,
                        periods=pred_len,
                        freq=time_diff
                    )
                    y_timestamp = pd.Series(y_timestamp, name='timestamps')

                    prediction_type = f"Kronos model prediction (Stock {stock_code}, latest {lookback} days)"

                elif start_date:
                    start_dt = pd.to_datetime(start_date)
                    mask = df['timestamps'] >= start_dt
                    time_range_df = df[mask]

                    if len(time_range_df) < lookback + pred_len:
                        update_progress(task_id, 100, 'error', f'Insufficient data from start time')
                        return

                    x_df = time_range_df.iloc[:lookback][required_cols]
                    x_timestamp = time_range_df.iloc[:lookback]['timestamps']
                    y_timestamp = time_range_df.iloc[lookback:lookback+pred_len]['timestamps']

                    start_timestamp = time_range_df['timestamps'].iloc[0]
                    end_timestamp = time_range_df['timestamps'].iloc[lookback+pred_len-1]
                    time_span = end_timestamp - start_timestamp

                    prediction_type = f"Kronos model prediction (within selected window)"
                else:
                    x_df = df.iloc[:lookback][required_cols]
                    x_timestamp = df.iloc[:lookback]['timestamps']
                    y_timestamp = df.iloc[lookback:lookback+pred_len]['timestamps']
                    prediction_type = "Kronos model prediction (latest data)"

                if isinstance(x_timestamp, pd.DatetimeIndex):
                    x_timestamp = pd.Series(x_timestamp, name='timestamps')
                if isinstance(y_timestamp, pd.DatetimeIndex):
                    y_timestamp = pd.Series(y_timestamp, name='timestamps')

                update_progress(task_id, 40, 'predicting', f'Running prediction (sample 1/{sample_count})...')

                # Run prediction with progress updates
                pred_df = predictor.predict(
                    df=x_df,
                    x_timestamp=x_timestamp,
                    y_timestamp=y_timestamp,
                    pred_len=pred_len,
                    T=temperature,
                    top_p=top_p,
                    sample_count=sample_count,
                    verbose=False
                )

                update_progress(task_id, 70, 'processing', 'Processing prediction results...')

                # Apply price limits if requested
                if apply_limits:
                    if stock_code:
                        last_close = float(df.iloc[-1]['close'])
                    elif start_date:
                        start_dt = pd.to_datetime(start_date)
                        mask = df['timestamps'] >= start_dt
                        time_range_df = df[mask]
                        last_close = float(time_range_df.iloc[lookback-1]['close'])
                    else:
                        last_close = float(df.iloc[lookback-1]['close'])

                    pred_df = apply_price_limits(pred_df, last_close, limit_rate)
                    prediction_type += f" (with ±{limit_rate*100:.0f}% price limits)"

            except Exception as e:
                update_progress(task_id, 100, 'error', f'Prediction failed: {str(e)}')
                return
        else:
            update_progress(task_id, 100, 'error', 'Model not loaded')
            return

        update_progress(task_id, 80, 'generating', 'Generating charts and results...')

        # Prepare actual data for comparison
        actual_data = []
        actual_df = None

        if stock_code:
            pass
        elif start_date:
            start_dt = pd.to_datetime(start_date)
            mask = df['timestamps'] >= start_dt
            time_range_df = df[mask]

            if len(time_range_df) >= lookback + pred_len:
                actual_df = time_range_df.iloc[lookback:lookback+pred_len]
                for i, (_, row) in enumerate(actual_df.iterrows()):
                    actual_data.append({
                        'timestamp': row['timestamps'].isoformat(),
                        'open': float(row['open']),
                        'high': float(row['high']),
                        'low': float(row['low']),
                        'close': float(row['close']),
                        'volume': float(row['volume']) if 'volume' in row else 0,
                        'amount': float(row['amount']) if 'amount' in row else 0
                    })
        else:
            if len(df) >= lookback + pred_len:
                actual_df = df.iloc[lookback:lookback+pred_len]
                for i, (_, row) in enumerate(actual_df.iterrows()):
                    actual_data.append({
                        'timestamp': row['timestamps'].isoformat(),
                        'open': float(row['open']),
                        'high': float(row['high']),
                        'low': float(row['low']),
                        'close': float(row['close']),
                        'volume': float(row['volume']) if 'volume' in row else 0,
                        'amount': float(row['amount']) if 'amount' in row else 0
                    })

        # Create chart
        if stock_code:
            historical_start_idx = max(0, len(df) - lookback)
        elif start_date:
            start_dt = pd.to_datetime(start_date)
            mask = df['timestamps'] >= start_dt
            historical_start_idx = df[mask].index[0] if len(df[mask]) > 0 else 0
        else:
            historical_start_idx = 0

        chart_json = create_prediction_chart(df, pred_df, lookback, pred_len, actual_df, historical_start_idx)

        # Prepare prediction results
        if 'timestamps' in df.columns:
            if stock_code:
                future_timestamps = y_timestamp
            elif start_date:
                start_dt = pd.to_datetime(start_date)
                mask = df['timestamps'] >= start_dt
                time_range_df = df[mask]

                if len(time_range_df) >= lookback:
                    last_timestamp = time_range_df['timestamps'].iloc[lookback-1]
                    time_diff = df['timestamps'].iloc[1] - df['timestamps'].iloc[0]
                    future_timestamps = pd.date_range(
                        start=last_timestamp + time_diff,
                        periods=pred_len,
                        freq=time_diff
                    )
                else:
                    future_timestamps = []
            else:
                last_timestamp = df['timestamps'].iloc[-1]
                time_diff = df['timestamps'].iloc[1] - df['timestamps'].iloc[0]
                future_timestamps = pd.date_range(
                    start=last_timestamp + time_diff,
                    periods=pred_len,
                    freq=time_diff
                )
        else:
            future_timestamps = range(len(df), len(df) + pred_len)

        prediction_results = []
        for i, (_, row) in enumerate(pred_df.iterrows()):
            prediction_results.append({
                'timestamp': future_timestamps[i].isoformat() if i < len(future_timestamps) else f"T{i}",
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume']) if 'volume' in row else 0,
                'amount': float(row['amount']) if 'amount' in row else 0
            })

        update_progress(task_id, 90, 'saving', 'Saving prediction results...')

        # Save prediction results
        try:
            save_prediction_results(
                file_path=data_source,
                prediction_type=prediction_type,
                prediction_results=prediction_results,
                actual_data=actual_data,
                input_data=x_df,
                prediction_params={
                    'lookback': lookback,
                    'pred_len': pred_len,
                    'temperature': temperature,
                    'top_p': top_p,
                    'sample_count': sample_count,
                    'start_date': start_date if start_date else 'latest',
                    'apply_price_limits': apply_limits,
                    'limit_rate': limit_rate,
                    'stock_code': stock_code if stock_code else None
                }
            )
        except Exception as e:
            print(f"Failed to save prediction results: {e}")

        # Store final results
        result = {
            'success': True,
            'prediction_type': prediction_type,
            'chart': chart_json,
            'prediction_results': prediction_results,
            'actual_data': actual_data,
            'has_comparison': len(actual_data) > 0,
            'message': f'Prediction completed, generated {pred_len} prediction points' + (f', including {len(actual_data)} actual data points for comparison' if len(actual_data) > 0 else '')
        }

        update_progress(task_id, 100, 'completed', 'Prediction completed successfully!')
        progress_tracker[task_id]['result'] = result

    except Exception as e:
        update_progress(task_id, 100, 'error', f'Prediction failed: {str(e)}')

# Available model configurations
AVAILABLE_MODELS = {
    'kronos-mini': {
        'name': 'Kronos-mini',
        'model_id': 'NeoQuasar/Kronos-mini',
        'tokenizer_id': 'NeoQuasar/Kronos-Tokenizer-2k',
        'context_length': 2048,
        'params': '4.1M',
        'description': 'Lightweight model, suitable for fast prediction'
    },
    'kronos-small': {
        'name': 'Kronos-small',
        'model_id': 'NeoQuasar/Kronos-small',
        'tokenizer_id': 'NeoQuasar/Kronos-Tokenizer-base',
        'context_length': 512,
        'params': '24.7M',
        'description': 'Small model, balanced performance and speed'
    },
    'kronos-base': {
        'name': 'Kronos-base',
        'model_id': 'NeoQuasar/Kronos-base',
        'tokenizer_id': 'NeoQuasar/Kronos-Tokenizer-base',
        'context_length': 512,
        'params': '102.3M',
        'description': 'Base model, provides better prediction quality'
    }
}

def load_data_files():
    """Scan data directory and return available data files"""
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    data_files = []
    
    if os.path.exists(data_dir):
        for file in os.listdir(data_dir):
            if file.endswith(('.csv', '.feather')):
                file_path = os.path.join(data_dir, file)
                file_size = os.path.getsize(file_path)
                data_files.append({
                    'name': file,
                    'path': file_path,
                    'size': f"{file_size / 1024:.1f} KB" if file_size < 1024*1024 else f"{file_size / (1024*1024):.1f} MB"
                })
    
    return data_files

def load_stock_data_akshare(symbol: str, lookback: int = 500) -> tuple:
    """
    Use akshare to download stock K-line data

    Args:
        symbol: Stock code
        lookback: Number of historical data days needed

    Returns:
        tuple: (DataFrame, error_message)
    """
    if not AKSHARE_AVAILABLE:
        return None, "akshare library not installed"

    print(f"Fetching stock data for {symbol}...")

    max_retries = 3
    df = None

    # Retry mechanism
    for attempt in range(1, max_retries + 1):
        try:
            # Use akshare to get A-share daily data
            df = ak.stock_zh_a_hist(symbol=symbol, period="daily", adjust="")

            if df is not None and not df.empty:
                break
        except Exception as e:
            print(f"Attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                time.sleep(1.5)

    # Check if data was successfully retrieved
    if df is None or df.empty:
        return None, f"Unable to fetch data for stock {symbol}. This may be due to network connection issues, server unavailability, or an incorrect stock code. Please check your network connection and try again later."

    # Rename columns to English
    df.rename(columns={
        "日期": "date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount"
    }, inplace=True)

    # Convert date format
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Convert numeric columns
    numeric_cols = ["open", "high", "low", "close", "volume", "amount"]
    for col in numeric_cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .replace({"--": None, "": None})
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fix invalid opening prices
    open_bad = (df["open"] == 0) | (df["open"].isna())
    if open_bad.any():
        df.loc[open_bad, "open"] = df["close"].shift(1)
        df["open"].fillna(df["close"], inplace=True)

    # Fix missing amount
    if df["amount"].isna().all() or (df["amount"] == 0).all():
        df["amount"] = df["close"] * df["volume"]

    # Rename date to timestamps for consistency
    df["timestamps"] = df["date"]

    # Only keep recent lookback+50 days of data (get some extra to ensure enough trading days)
    if len(df) > lookback + 50:
        df = df.iloc[-(lookback + 50):].reset_index(drop=True)

    print(f"Data fetched successfully: {len(df)} records, date range: {df['timestamps'].min()} ~ {df['timestamps'].max()}")

    return df, None


def apply_price_limits(pred_df: pd.DataFrame, last_close: float, limit_rate: float = 0.1) -> pd.DataFrame:
    """
    Apply price limit constraints (Chinese A-share ±10%)

    Args:
        pred_df: Prediction results DataFrame
        last_close: Last trading day's closing price
        limit_rate: Price limit rate (default 0.1 for 10%)

    Returns:
        pd.DataFrame: Prediction results after applying limits
    """
    print(f"Applying ±{limit_rate*100:.0f}% price limits...")

    pred_df = pred_df.reset_index(drop=True)
    cols = ["open", "high", "low", "close"]
    pred_df[cols] = pred_df[cols].astype("float64")

    for i in range(len(pred_df)):
        limit_up = last_close * (1 + limit_rate)
        limit_down = last_close * (1 - limit_rate)

        for col in cols:
            value = pred_df.at[i, col]
            if pd.notna(value):
                clipped = max(min(value, limit_up), limit_down)
                pred_df.at[i, col] = float(clipped)

        last_close = float(pred_df.at[i, "close"])

    return pred_df


def load_data_file(file_path):
    """Load data file"""
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.feather'):
            df = pd.read_feather(file_path)
        else:
            return None, "Unsupported file format"

        # Check required columns
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            return None, f"Missing required columns: {required_cols}"

        # Process timestamp column
        if 'timestamps' in df.columns:
            df['timestamps'] = pd.to_datetime(df['timestamps'])
        elif 'timestamp' in df.columns:
            df['timestamps'] = pd.to_datetime(df['timestamp'])
        elif 'date' in df.columns:
            # If column name is 'date', rename it to 'timestamps'
            df['timestamps'] = pd.to_datetime(df['date'])
        else:
            # If no timestamp column exists, create one
            df['timestamps'] = pd.date_range(start='2024-01-01', periods=len(df), freq='1H')

        # Ensure numeric columns are numeric type
        for col in ['open', 'high', 'low', 'close']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Process volume column (optional)
        if 'volume' in df.columns:
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce')

        # Process amount column (optional, but not used for prediction)
        if 'amount' in df.columns:
            df['amount'] = pd.to_numeric(df['amount'], errors='coerce')

        # Remove rows containing NaN values
        df = df.dropna()

        return df, None

    except Exception as e:
        return None, f"Failed to load file: {str(e)}"

def save_prediction_results(file_path, prediction_type, prediction_results, actual_data, input_data, prediction_params):
    """Save prediction results to file"""
    try:
        # Create prediction results directory
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prediction_results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'prediction_{timestamp}.json'
        filepath = os.path.join(results_dir, filename)
        
        # Prepare data for saving
        save_data = {
            'timestamp': datetime.datetime.now().isoformat(),
            'file_path': file_path,
            'prediction_type': prediction_type,
            'prediction_params': prediction_params,
            'input_data_summary': {
                'rows': len(input_data),
                'columns': list(input_data.columns),
                'price_range': {
                    'open': {'min': float(input_data['open'].min()), 'max': float(input_data['open'].max())},
                    'high': {'min': float(input_data['high'].min()), 'max': float(input_data['high'].max())},
                    'low': {'min': float(input_data['low'].min()), 'max': float(input_data['low'].max())},
                    'close': {'min': float(input_data['close'].min()), 'max': float(input_data['close'].max())}
                },
                'last_values': {
                    'open': float(input_data['open'].iloc[-1]),
                    'high': float(input_data['high'].iloc[-1]),
                    'low': float(input_data['low'].iloc[-1]),
                    'close': float(input_data['close'].iloc[-1])
                }
            },
            'prediction_results': prediction_results,
            'actual_data': actual_data,
            'analysis': {}
        }
        
        # If actual data exists, perform comparison analysis
        if actual_data and len(actual_data) > 0:
            # Calculate continuity analysis
            if len(prediction_results) > 0 and len(actual_data) > 0:
                last_pred = prediction_results[0]  # First prediction point
            first_actual = actual_data[0]      # First actual point
                
            save_data['analysis']['continuity'] = {
                    'last_prediction': {
                        'open': last_pred['open'],
                        'high': last_pred['high'],
                        'low': last_pred['low'],
                        'close': last_pred['close']
                    },
                    'first_actual': {
                        'open': first_actual['open'],
                        'high': first_actual['high'],
                        'low': first_actual['low'],
                        'close': first_actual['close']
                    },
                    'gaps': {
                        'open_gap': abs(last_pred['open'] - first_actual['open']),
                        'high_gap': abs(last_pred['high'] - first_actual['high']),
                        'low_gap': abs(last_pred['low'] - first_actual['low']),
                        'close_gap': abs(last_pred['close'] - first_actual['close'])
                    },
                    'gap_percentages': {
                        'open_gap_pct': (abs(last_pred['open'] - first_actual['open']) / first_actual['open']) * 100,
                        'high_gap_pct': (abs(last_pred['high'] - first_actual['high']) / first_actual['high']) * 100,
                        'low_gap_pct': (abs(last_pred['low'] - first_actual['low']) / first_actual['low']) * 100,
                        'close_gap_pct': (abs(last_pred['close'] - first_actual['close']) / first_actual['close']) * 100
                    }
                }
        
        # Save to file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        print(f"Prediction results saved to: {filepath}")
        return filepath
        
    except Exception as e:
        print(f"Failed to save prediction results: {e}")
        return None

def create_prediction_chart(df, pred_df, lookback, pred_len, actual_df=None, historical_start_idx=0):
    """Create prediction chart"""
    # Use specified historical data start position, not always from the beginning of df
    if historical_start_idx + lookback + pred_len <= len(df):
        # Display lookback historical points + pred_len prediction points starting from specified position
        historical_df = df.iloc[historical_start_idx:historical_start_idx+lookback]
        prediction_range = range(historical_start_idx+lookback, historical_start_idx+lookback+pred_len)
    else:
        # If data is insufficient, adjust to maximum available range
        available_lookback = min(lookback, len(df) - historical_start_idx)
        available_pred_len = min(pred_len, max(0, len(df) - historical_start_idx - available_lookback))
        historical_df = df.iloc[historical_start_idx:historical_start_idx+available_lookback]
        prediction_range = range(historical_start_idx+available_lookback, historical_start_idx+available_lookback+available_pred_len)
    
    # Create chart
    fig = go.Figure()
    
    # Add historical data (candlestick chart)
    fig.add_trace(go.Candlestick(
        x=historical_df['timestamps'] if 'timestamps' in historical_df.columns else historical_df.index,
        open=historical_df['open'],
        high=historical_df['high'],
        low=historical_df['low'],
        close=historical_df['close'],
        name='Historical Data (400 data points)',
        increasing_line_color='#26A69A',
        decreasing_line_color='#EF5350'
    ))
    
    # Add prediction data (candlestick chart)
    if pred_df is not None and len(pred_df) > 0:
        # Calculate prediction data timestamps - ensure continuity with historical data
        if 'timestamps' in df.columns and len(historical_df) > 0:
            # Start from the last timestamp of historical data, create prediction timestamps with the same time interval
            last_timestamp = historical_df['timestamps'].iloc[-1]
            time_diff = df['timestamps'].iloc[1] - df['timestamps'].iloc[0] if len(df) > 1 else pd.Timedelta(hours=1)
            
            pred_timestamps = pd.date_range(
                start=last_timestamp + time_diff,
                periods=len(pred_df),
                freq=time_diff
            )
        else:
            # If no timestamps, use index
            pred_timestamps = range(len(historical_df), len(historical_df) + len(pred_df))
        
        fig.add_trace(go.Candlestick(
            x=pred_timestamps,
            open=pred_df['open'],
            high=pred_df['high'],
            low=pred_df['low'],
            close=pred_df['close'],
            name='Prediction Data (120 data points)',
            increasing_line_color='#66BB6A',
            decreasing_line_color='#FF7043'
        ))
    
    # Add actual data for comparison (if exists)
    if actual_df is not None and len(actual_df) > 0:
        # Actual data should be in the same time period as prediction data
        if 'timestamps' in df.columns:
            # Actual data should use the same timestamps as prediction data to ensure time alignment
            if 'pred_timestamps' in locals():
                actual_timestamps = pred_timestamps
            else:
                # If no prediction timestamps, calculate from the last timestamp of historical data
                if len(historical_df) > 0:
                    last_timestamp = historical_df['timestamps'].iloc[-1]
                    time_diff = df['timestamps'].iloc[1] - df['timestamps'].iloc[0] if len(df) > 1 else pd.Timedelta(hours=1)
                    actual_timestamps = pd.date_range(
                        start=last_timestamp + time_diff,
                        periods=len(actual_df),
                        freq=time_diff
                    )
                else:
                    actual_timestamps = range(len(historical_df), len(historical_df) + len(actual_df))
        else:
            actual_timestamps = range(len(historical_df), len(historical_df) + len(actual_df))
        
        fig.add_trace(go.Candlestick(
            x=actual_timestamps,
            open=actual_df['open'],
            high=actual_df['high'],
            low=actual_df['low'],
            close=actual_df['close'],
            name='Actual Data (120 data points)',
            increasing_line_color='#FF9800',
            decreasing_line_color='#F44336'
        ))
    
    # Update layout
    fig.update_layout(
        title='Kronos Financial Prediction Results - 400 Historical Points + 120 Prediction Points vs 120 Actual Points',
        xaxis_title='Time',
        yaxis_title='Price',
        template='plotly_white',
        height=600,
        showlegend=True
    )
    
    # Ensure x-axis time continuity
    if 'timestamps' in historical_df.columns:
        # Get all timestamps and sort them
        all_timestamps = []
        if len(historical_df) > 0:
            all_timestamps.extend(historical_df['timestamps'])
        if 'pred_timestamps' in locals():
            all_timestamps.extend(pred_timestamps)
        if 'actual_timestamps' in locals():
            all_timestamps.extend(actual_timestamps)
        
        if all_timestamps:
            all_timestamps = sorted(all_timestamps)
            fig.update_xaxes(
                range=[all_timestamps[0], all_timestamps[-1]],
                rangeslider_visible=False,
                type='date'
            )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/api/data-files')
def get_data_files():
    """Get available data file list"""
    data_files = load_data_files()
    return jsonify(data_files)

@app.route('/api/load-stock', methods=['POST'])
def load_stock():
    """Load stock data by stock code using akshare"""
    try:
        if not AKSHARE_AVAILABLE:
            return jsonify({'error': 'akshare library not installed, please install it first'}), 400

        data = request.get_json()
        stock_code = data.get('stock_code')

        if not stock_code:
            return jsonify({'error': 'Stock code cannot be empty'}), 400

        # Fetch stock data
        df, error = load_stock_data_akshare(stock_code, lookback=500)
        if error:
            return jsonify({'error': error}), 400

        # Detect data time frequency
        def detect_timeframe(df):
            if len(df) < 2:
                return "Unknown"

            time_diffs = []
            for i in range(1, min(10, len(df))):
                diff = df['timestamps'].iloc[i] - df['timestamps'].iloc[i-1]
                time_diffs.append(diff)

            if not time_diffs:
                return "Unknown"

            avg_diff = sum(time_diffs, pd.Timedelta(0)) / len(time_diffs)

            if avg_diff < pd.Timedelta(minutes=1):
                return f"{avg_diff.total_seconds():.0f} seconds"
            elif avg_diff < pd.Timedelta(hours=1):
                return f"{avg_diff.total_seconds() / 60:.0f} minutes"
            elif avg_diff < pd.Timedelta(days=1):
                return f"{avg_diff.total_seconds() / 3600:.0f} hours"
            else:
                return f"{avg_diff.days} days"

        # Return data information
        data_info = {
            'rows': len(df),
            'columns': list(df.columns),
            'start_date': df['timestamps'].min().isoformat(),
            'end_date': df['timestamps'].max().isoformat(),
            'price_range': {
                'min': float(df[['open', 'high', 'low', 'close']].min().min()),
                'max': float(df[['open', 'high', 'low', 'close']].max().max())
            },
            'prediction_columns': ['open', 'high', 'low', 'close'] + (['volume'] if 'volume' in df.columns else []),
            'timeframe': detect_timeframe(df),
            'stock_code': stock_code,
            'data_source': 'akshare'
        }

        return jsonify({
            'success': True,
            'data_info': data_info,
            'message': f'Successfully loaded stock {stock_code}, total {len(df)} records'
        })

    except Exception as e:
        return jsonify({'error': f'Failed to load stock data: {str(e)}'}), 500


@app.route('/api/load-data', methods=['POST'])
def load_data():
    """Load data file"""
    try:
        data = request.get_json()
        file_path = data.get('file_path')

        if not file_path:
            return jsonify({'error': 'File path cannot be empty'}), 400

        df, error = load_data_file(file_path)
        if error:
            return jsonify({'error': error}), 400
        
        # Detect data time frequency
        def detect_timeframe(df):
            if len(df) < 2:
                return "Unknown"
            
            time_diffs = []
            for i in range(1, min(10, len(df))):  # Check first 10 time differences
                diff = df['timestamps'].iloc[i] - df['timestamps'].iloc[i-1]
                time_diffs.append(diff)
            
            if not time_diffs:
                return "Unknown"
            
            # Calculate average time difference
            avg_diff = sum(time_diffs, pd.Timedelta(0)) / len(time_diffs)
            
            # Convert to readable format
            if avg_diff < pd.Timedelta(minutes=1):
                return f"{avg_diff.total_seconds():.0f} seconds"
            elif avg_diff < pd.Timedelta(hours=1):
                return f"{avg_diff.total_seconds() / 60:.0f} minutes"
            elif avg_diff < pd.Timedelta(days=1):
                return f"{avg_diff.total_seconds() / 3600:.0f} hours"
            else:
                return f"{avg_diff.days} days"
        
        # Return data information
        data_info = {
            'rows': len(df),
            'columns': list(df.columns),
            'start_date': df['timestamps'].min().isoformat() if 'timestamps' in df.columns else 'N/A',
            'end_date': df['timestamps'].max().isoformat() if 'timestamps' in df.columns else 'N/A',
            'price_range': {
                'min': float(df[['open', 'high', 'low', 'close']].min().min()),
                'max': float(df[['open', 'high', 'low', 'close']].max().max())
            },
            'prediction_columns': ['open', 'high', 'low', 'close'] + (['volume'] if 'volume' in df.columns else []),
            'timeframe': detect_timeframe(df)
        }
        
        return jsonify({
            'success': True,
            'data_info': data_info,
            'message': f'Successfully loaded data, total {len(df)} rows'
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to load data: {str(e)}'}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    """Start prediction task"""
    try:
        data = request.get_json()

        # Generate unique task ID
        task_id = str(uuid.uuid4())

        # Extract all parameters
        params = {
            'file_path': data.get('file_path'),
            'stock_code': data.get('stock_code'),
            'lookback': int(data.get('lookback', 400)),
            'pred_len': int(data.get('pred_len', 120)),
            'temperature': float(data.get('temperature', 1.0)),
            'top_p': float(data.get('top_p', 0.9)),
            'sample_count': int(data.get('sample_count', 1)),
            'apply_limits': data.get('apply_price_limits', False),
            'limit_rate': float(data.get('limit_rate', 0.1)),
            'start_date': data.get('start_date')
        }

        # Start prediction in background thread
        thread = threading.Thread(target=run_prediction_async, args=(task_id, params))
        thread.daemon = True
        thread.start()

        return jsonify({
            'success': True,
            'task_id': task_id,
            'message': 'Prediction task started'
        })

    except Exception as e:
        return jsonify({'error': f'Failed to start prediction: {str(e)}'}), 500

@app.route('/api/predict-progress/<task_id>', methods=['GET'])
def get_prediction_progress(task_id):
    """Get prediction progress"""
    try:
        if task_id not in progress_tracker:
            return jsonify({'error': 'Task not found'}), 404

        progress_info = progress_tracker[task_id]
        response = {
            'progress': progress_info['progress'],
            'status': progress_info['status'],
            'message': progress_info['message'],
            'timestamp': progress_info['timestamp']
        }

        # If task is completed, include result
        if progress_info['status'] == 'completed' and 'result' in progress_info:
            response['result'] = progress_info['result']

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': f'Failed to get progress: {str(e)}'}), 500

@app.route('/api/predict-legacy', methods=['POST'])
def predict_legacy():
    """Perform prediction (legacy synchronous endpoint)"""
    try:
        data = request.get_json()
        file_path = data.get('file_path')
        stock_code = data.get('stock_code')  # New: support stock code
        lookback = int(data.get('lookback', 400))
        pred_len = int(data.get('pred_len', 120))

        # Get prediction quality parameters
        temperature = float(data.get('temperature', 1.0))
        top_p = float(data.get('top_p', 0.9))
        sample_count = int(data.get('sample_count', 1))

        # Get price limit parameter (new)
        apply_limits = data.get('apply_price_limits', False)
        limit_rate = float(data.get('limit_rate', 0.1))

        # Load data - support both file path and stock code
        if stock_code:
            # Use akshare to fetch stock data
            if not AKSHARE_AVAILABLE:
                return jsonify({'error': 'akshare library not installed'}), 400
            df, error = load_stock_data_akshare(stock_code, lookback=lookback+pred_len+50)
            if error:
                return jsonify({'error': error}), 400
            data_source = f"Stock {stock_code} (akshare)"
        elif file_path:
            # Load from file
            df, error = load_data_file(file_path)
            if error:
                return jsonify({'error': error}), 400
            data_source = file_path
        else:
            return jsonify({'error': 'Either file_path or stock_code must be provided'}), 400
        
        if len(df) < lookback:
            return jsonify({'error': f'Insufficient data length, need at least {lookback} rows'}), 400
        
        # Perform prediction
        if MODEL_AVAILABLE and predictor is not None:
            try:
                # Use real Kronos model
                # Only use necessary columns: OHLCV, excluding amount
                required_cols = ['open', 'high', 'low', 'close']
                if 'volume' in df.columns:
                    required_cols.append('volume')
                
                # Process time period selection
                start_date = data.get('start_date')

                # For stock code mode, always use latest data (ignore time window)
                if stock_code:
                    # Stock code mode: use latest available data
                    if len(df) < lookback:
                        return jsonify({'error': f'Insufficient data, need at least {lookback} data points, currently only {len(df)} available'}), 400

                    # Use the most recent lookback data points
                    x_df = df.iloc[-lookback:][required_cols]
                    x_timestamp = df.iloc[-lookback:]['timestamps']

                    # Generate future timestamps for prediction
                    last_timestamp = df['timestamps'].iloc[-1]
                    time_diff = df['timestamps'].iloc[-1] - df['timestamps'].iloc[-2] if len(df) > 1 else pd.Timedelta(days=1)
                    y_timestamp = pd.date_range(
                        start=last_timestamp + time_diff,
                        periods=pred_len,
                        freq=time_diff
                    )
                    y_timestamp = pd.Series(y_timestamp, name='timestamps')

                    prediction_type = f"Kronos model prediction (Stock {stock_code}, latest {lookback} days)"

                elif start_date:
                    # File mode with custom time period
                    start_dt = pd.to_datetime(start_date)

                    # Find data after start time
                    mask = df['timestamps'] >= start_dt
                    time_range_df = df[mask]

                    # Ensure sufficient data: lookback + pred_len
                    if len(time_range_df) < lookback + pred_len:
                        return jsonify({'error': f'Insufficient data from start time {start_dt.strftime("%Y-%m-%d %H:%M")}, need at least {lookback + pred_len} data points, currently only {len(time_range_df)} available'}), 400

                    # Use first lookback data points within selected window for prediction
                    x_df = time_range_df.iloc[:lookback][required_cols]
                    x_timestamp = time_range_df.iloc[:lookback]['timestamps']

                    # Use last pred_len data points within selected window as actual values
                    y_timestamp = time_range_df.iloc[lookback:lookback+pred_len]['timestamps']

                    # Calculate actual time period length
                    start_timestamp = time_range_df['timestamps'].iloc[0]
                    end_timestamp = time_range_df['timestamps'].iloc[lookback+pred_len-1]
                    time_span = end_timestamp - start_timestamp

                    prediction_type = f"Kronos model prediction (within selected window: first {lookback} data points for prediction, last {pred_len} data points for comparison, time span: {time_span})"
                else:
                    # File mode: use latest data
                    x_df = df.iloc[:lookback][required_cols]
                    x_timestamp = df.iloc[:lookback]['timestamps']
                    y_timestamp = df.iloc[lookback:lookback+pred_len]['timestamps']
                    prediction_type = "Kronos model prediction (latest data)"
                
                # Ensure timestamps are Series format, not DatetimeIndex, to avoid .dt attribute error in Kronos model
                if isinstance(x_timestamp, pd.DatetimeIndex):
                    x_timestamp = pd.Series(x_timestamp, name='timestamps')
                if isinstance(y_timestamp, pd.DatetimeIndex):
                    y_timestamp = pd.Series(y_timestamp, name='timestamps')
                
                # Set verbose=False to disable console progress bar
                pred_df = predictor.predict(
                    df=x_df,
                    x_timestamp=x_timestamp,
                    y_timestamp=y_timestamp,
                    pred_len=pred_len,
                    T=temperature,
                    top_p=top_p,
                    sample_count=sample_count,
                    verbose=False  # Disable console progress bar
                )

                # Apply price limits if requested
                if apply_limits:
                    if stock_code:
                        # Stock code mode: use last close from latest data
                        last_close = float(df.iloc[-1]['close'])
                    elif start_date:
                        # File mode with custom time period
                        start_dt = pd.to_datetime(start_date)
                        mask = df['timestamps'] >= start_dt
                        time_range_df = df[mask]
                        last_close = float(time_range_df.iloc[lookback-1]['close'])
                    else:
                        # File mode: latest data
                        last_close = float(df.iloc[lookback-1]['close'])

                    pred_df = apply_price_limits(pred_df, last_close, limit_rate)
                    prediction_type += f" (with ±{limit_rate*100:.0f}% price limits)"

            except Exception as e:
                return jsonify({'error': f'Kronos model prediction failed: {str(e)}'}), 500
        else:
            return jsonify({'error': 'Kronos model not loaded, please load model first'}), 400
        
        # Prepare actual data for comparison (if exists)
        actual_data = []
        actual_df = None

        # Stock code mode: no actual data (we're predicting future)
        if stock_code:
            # No actual data for future predictions
            pass

        elif start_date:  # File mode with custom time period
            # Fix logic: use data within selected window
            # Prediction uses first 400 data points within selected window
            # Actual data should be last 120 data points within selected window
            start_dt = pd.to_datetime(start_date)

            # Find data starting from start_date
            mask = df['timestamps'] >= start_dt
            time_range_df = df[mask]

            if len(time_range_df) >= lookback + pred_len:
                # Get last 120 data points within selected window as actual values
                actual_df = time_range_df.iloc[lookback:lookback+pred_len]

                for i, (_, row) in enumerate(actual_df.iterrows()):
                    actual_data.append({
                        'timestamp': row['timestamps'].isoformat(),
                        'open': float(row['open']),
                        'high': float(row['high']),
                        'low': float(row['low']),
                        'close': float(row['close']),
                        'volume': float(row['volume']) if 'volume' in row else 0,
                        'amount': float(row['amount']) if 'amount' in row else 0
                    })
        else:  # File mode: latest data
            # Prediction uses first 400 data points
            # Actual data should be 120 data points after first 400 data points
            if len(df) >= lookback + pred_len:
                actual_df = df.iloc[lookback:lookback+pred_len]
                for i, (_, row) in enumerate(actual_df.iterrows()):
                    actual_data.append({
                        'timestamp': row['timestamps'].isoformat(),
                        'open': float(row['open']),
                        'high': float(row['high']),
                        'low': float(row['low']),
                        'close': float(row['close']),
                        'volume': float(row['volume']) if 'volume' in row else 0,
                        'amount': float(row['amount']) if 'amount' in row else 0
                    })
        
        # Create chart - pass historical data start position
        if stock_code:
            # Stock code mode: use the last lookback points
            historical_start_idx = max(0, len(df) - lookback)
        elif start_date:
            # File mode with custom time period: find starting position of historical data in original df
            start_dt = pd.to_datetime(start_date)
            mask = df['timestamps'] >= start_dt
            historical_start_idx = df[mask].index[0] if len(df[mask]) > 0 else 0
        else:
            # File mode: latest data, start from beginning
            historical_start_idx = 0

        chart_json = create_prediction_chart(df, pred_df, lookback, pred_len, actual_df, historical_start_idx)
        
        # Prepare prediction result data - fix timestamp calculation logic
        if 'timestamps' in df.columns:
            if stock_code:
                # Stock code mode: y_timestamp already calculated above
                future_timestamps = y_timestamp
            elif start_date:
                # File mode with custom time period: use selected window data to calculate timestamps
                start_dt = pd.to_datetime(start_date)
                mask = df['timestamps'] >= start_dt
                time_range_df = df[mask]

                if len(time_range_df) >= lookback:
                    # Calculate prediction timestamps starting from last time point of selected window
                    last_timestamp = time_range_df['timestamps'].iloc[lookback-1]
                    time_diff = df['timestamps'].iloc[1] - df['timestamps'].iloc[0]
                    future_timestamps = pd.date_range(
                        start=last_timestamp + time_diff,
                        periods=pred_len,
                        freq=time_diff
                    )
                else:
                    future_timestamps = []
            else:
                # File mode: latest data, calculate from last time point of entire data file
                last_timestamp = df['timestamps'].iloc[-1]
                time_diff = df['timestamps'].iloc[1] - df['timestamps'].iloc[0]
                future_timestamps = pd.date_range(
                    start=last_timestamp + time_diff,
                    periods=pred_len,
                    freq=time_diff
                )
        else:
            future_timestamps = range(len(df), len(df) + pred_len)
        
        prediction_results = []
        for i, (_, row) in enumerate(pred_df.iterrows()):
            prediction_results.append({
                'timestamp': future_timestamps[i].isoformat() if i < len(future_timestamps) else f"T{i}",
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume']) if 'volume' in row else 0,
                'amount': float(row['amount']) if 'amount' in row else 0
            })
        
        # Save prediction results to file
        try:
            save_prediction_results(
                file_path=data_source,
                prediction_type=prediction_type,
                prediction_results=prediction_results,
                actual_data=actual_data,
                input_data=x_df,
                prediction_params={
                    'lookback': lookback,
                    'pred_len': pred_len,
                    'temperature': temperature,
                    'top_p': top_p,
                    'sample_count': sample_count,
                    'start_date': start_date if start_date else 'latest',
                    'apply_price_limits': apply_limits,
                    'limit_rate': limit_rate,
                    'stock_code': stock_code if stock_code else None
                }
            )
        except Exception as e:
            print(f"Failed to save prediction results: {e}")
        
        return jsonify({
            'success': True,
            'prediction_type': prediction_type,
            'chart': chart_json,
            'prediction_results': prediction_results,
            'actual_data': actual_data,
            'has_comparison': len(actual_data) > 0,
            'message': f'Prediction completed, generated {pred_len} prediction points' + (f', including {len(actual_data)} actual data points for comparison' if len(actual_data) > 0 else '')
        })
        
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/api/load-model', methods=['POST'])
def load_model():
    """Load Kronos model"""
    global tokenizer, model, predictor
    
    try:
        if not MODEL_AVAILABLE:
            return jsonify({'error': 'Kronos model library not available'}), 400
        
        data = request.get_json()
        model_key = data.get('model_key', 'kronos-small')
        device = data.get('device', 'cpu')
        
        if model_key not in AVAILABLE_MODELS:
            return jsonify({'error': f'Unsupported model: {model_key}'}), 400
        
        model_config = AVAILABLE_MODELS[model_key]
        
        # Load tokenizer and model
        tokenizer = KronosTokenizer.from_pretrained(model_config['tokenizer_id'])
        model = Kronos.from_pretrained(model_config['model_id'])
        
        # Create predictor
        predictor = KronosPredictor(model, tokenizer, device=device, max_context=model_config['context_length'])
        
        return jsonify({
            'success': True,
            'message': f'Model loaded successfully: {model_config["name"]} ({model_config["params"]}) on {device}',
            'model_info': {
                'name': model_config['name'],
                'params': model_config['params'],
                'context_length': model_config['context_length'],
                'description': model_config['description']
            }
        })
        
    except Exception as e:
        return jsonify({'error': f'Model loading failed: {str(e)}'}), 500

@app.route('/api/available-models')
def get_available_models():
    """Get available model list"""
    return jsonify({
        'models': AVAILABLE_MODELS,
        'model_available': MODEL_AVAILABLE
    })

@app.route('/api/model-status')
def get_model_status():
    """Get model status"""
    if MODEL_AVAILABLE:
        if predictor is not None:
            return jsonify({
                'available': True,
                'loaded': True,
                'message': 'Kronos model loaded and available',
                'current_model': {
                    'name': predictor.model.__class__.__name__,
                    'device': str(next(predictor.model.parameters()).device)
                }
            })
        else:
            return jsonify({
                'available': True,
                'loaded': False,
                'message': 'Kronos model available but not loaded'
            })
    else:
        return jsonify({
            'available': False,
            'loaded': False,
            'message': 'Kronos model library not available, please install related dependencies'
        })


@app.route('/api/akshare-status')
def get_akshare_status():
    """Get akshare availability status"""
    return jsonify({
        'available': AKSHARE_AVAILABLE,
        'message': 'akshare available, stock code query enabled' if AKSHARE_AVAILABLE else 'akshare not installed, please install it to enable stock code query'
    })

@app.route('/api/prediction-history/<stock_code>', methods=['GET'])
def get_prediction_history(stock_code):
    """Get prediction history for a specific stock"""
    try:
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prediction_results')
        prediction_files = []
        
        if os.path.exists(results_dir):
            for filename in os.listdir(results_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(results_dir, filename)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        # Check if this prediction is for the requested stock
                        if data.get('prediction_params', {}).get('stock_code') == stock_code:
                            # Extract timestamp from filename: prediction_20250826_163800.json
                            timestamp_str = filename.split('_')[1] + '_' + filename.split('_')[2].split('.')[0]
                            prediction_files.append({
                                'filename': filename,
                                'timestamp': timestamp_str,
                                'date': datetime.datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S').strftime('%Y-%m-%d %H:%M:%S'),
                                'file_path': filepath
                            })
                    except Exception as e:
                        print(f"Failed to read prediction file {filename}: {e}")
        
        # Sort by timestamp in descending order (newest first)
        prediction_files.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return jsonify({
            'success': True,
            'stock_code': stock_code,
            'predictions': prediction_files,
            'count': len(prediction_files)
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to get prediction history: {str(e)}'}), 500

@app.route('/api/prediction-detail/<filename>', methods=['GET'])
def get_prediction_detail(filename):
    """Get detailed prediction data for a specific prediction file"""
    try:
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prediction_results')
        filepath = os.path.join(results_dir, filename)
        
        if not os.path.exists(filepath):
            return jsonify({'error': 'Prediction file not found'}), 404
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return jsonify({
            'success': True,
            'prediction': data
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to get prediction detail: {str(e)}'}), 500

if __name__ == '__main__':
    print("Starting Kronos Web UI...")
    print(f"Model availability: {MODEL_AVAILABLE}")
    if MODEL_AVAILABLE:
        print("Tip: You can load Kronos model through /api/load-model endpoint")
    else:
        print("Tip: Will use simulated data for demonstration")

    app.run(debug=True, host='0.0.0.0', port=5000)
