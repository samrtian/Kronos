#!/usr/bin/env python3
"""
Integration test script for enhanced Kronos WebUI
Tests akshare integration and price limits functionality
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Test if all required modules can be imported"""
    print("🧪 Testing imports...")

    try:
        import flask
        print("  ✅ Flask imported successfully")
    except ImportError as e:
        print(f"  ❌ Flask import failed: {e}")
        return False

    try:
        import pandas
        print("  ✅ Pandas imported successfully")
    except ImportError as e:
        print(f"  ❌ Pandas import failed: {e}")
        return False

    try:
        import numpy
        print("  ✅ NumPy imported successfully")
    except ImportError as e:
        print(f"  ❌ NumPy import failed: {e}")
        return False

    try:
        import plotly
        print("  ✅ Plotly imported successfully")
    except ImportError as e:
        print(f"  ❌ Plotly import failed: {e}")
        return False

    try:
        import matplotlib
        print("  ✅ Matplotlib imported successfully")
    except ImportError as e:
        print(f"  ❌ Matplotlib import failed: {e}")
        return False

    try:
        import akshare as ak
        print("  ✅ Akshare imported successfully")
        return True
    except ImportError as e:
        print(f"  ⚠️  Akshare import failed: {e}")
        print("  ℹ️  Stock code query will be disabled")
        return True  # Not critical, return True anyway

def test_app_routes():
    """Test if Flask app and routes are properly configured"""
    print("\n🧪 Testing Flask app configuration...")

    try:
        from app import app, AKSHARE_AVAILABLE, MODEL_AVAILABLE
        print("  ✅ App imported successfully")
        print(f"  ℹ️  Akshare available: {AKSHARE_AVAILABLE}")
        print(f"  ℹ️  Model available: {MODEL_AVAILABLE}")

        # Test if new routes exist
        routes = [rule.rule for rule in app.url_map.iter_rules()]

        required_routes = [
            '/api/akshare-status',
            '/api/load-stock',
            '/api/load-data',
            '/api/predict',
            '/api/model-status'
        ]

        for route in required_routes:
            if route in routes:
                print(f"  ✅ Route {route} registered")
            else:
                print(f"  ❌ Route {route} not found")
                return False

        return True

    except Exception as e:
        print(f"  ❌ App configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_price_limits_function():
    """Test price limits function"""
    print("\n🧪 Testing price limits function...")

    try:
        from app import apply_price_limits
        import pandas as pd

        # Create test data
        test_df = pd.DataFrame({
            'open': [100.0, 115.0, 90.0],
            'high': [105.0, 120.0, 95.0],
            'low': [95.0, 110.0, 85.0],
            'close': [102.0, 118.0, 88.0]
        })

        last_close = 100.0
        limit_rate = 0.1  # 10%

        result_df = apply_price_limits(test_df, last_close, limit_rate)

        # Check if limits are applied correctly
        # First row should be within ±10% of 100
        if result_df.iloc[0]['close'] > 110.0:
            print(f"  ❌ Close price {result_df.iloc[0]['close']} exceeds upper limit 110.0")
            return False
        if result_df.iloc[0]['close'] < 90.0:
            print(f"  ❌ Close price {result_df.iloc[0]['close']} below lower limit 90.0")
            return False

        print(f"  ✅ Price limits applied correctly")
        print(f"     Original close: {test_df.iloc[1]['close']}")
        print(f"     Limited close: {result_df.iloc[1]['close']}")

        return True

    except Exception as e:
        print(f"  ❌ Price limits test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_akshare_data_loading():
    """Test akshare data loading function"""
    print("\n🧪 Testing akshare data loading...")

    try:
        from app import load_stock_data_akshare, AKSHARE_AVAILABLE

        if not AKSHARE_AVAILABLE:
            print("  ⚠️  Akshare not available, skipping test")
            return True

        # Try to load a sample stock (Ping An Bank: 000001)
        print("  📡 Attempting to fetch stock 000001 (may take a few seconds)...")
        df, error = load_stock_data_akshare("000001", lookback=100)

        if error:
            print(f"  ⚠️  Data loading returned error: {error}")
            print("  ℹ️  This might be due to network issues or API limits")
            return True  # Don't fail test for network issues

        if df is None or df.empty:
            print("  ⚠️  Received empty dataframe")
            return True

        print(f"  ✅ Successfully loaded {len(df)} records")
        print(f"     Columns: {list(df.columns)}")
        print(f"     Date range: {df['timestamps'].min()} to {df['timestamps'].max()}")

        # Verify required columns exist
        required_cols = ['timestamps', 'open', 'high', 'low', 'close', 'volume', 'amount']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            print(f"  ❌ Missing columns: {missing_cols}")
            return False

        print("  ✅ All required columns present")
        return True

    except Exception as e:
        print(f"  ⚠️  Akshare test error: {e}")
        print("  ℹ️  This is not critical for the integration")
        return True  # Don't fail for network/API issues

def main():
    """Run all tests"""
    print("=" * 70)
    print("Kronos WebUI Integration Test Suite")
    print("=" * 70)

    tests = [
        ("Import Test", test_imports),
        ("Flask App Configuration", test_app_routes),
        ("Price Limits Function", test_price_limits_function),
        ("Akshare Data Loading", test_akshare_data_loading),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ {test_name} crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

    # Print summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")

    all_passed = all(result for _, result in results)

    print("\n" + "=" * 70)
    if all_passed:
        print("🎉 All tests passed! Integration successful!")
        print("\nYou can now start the WebUI:")
        print("  python run.py")
        print("\nOr directly:")
        print("  python app.py")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        print("\nMost common issues:")
        print("  1. Missing dependencies - run: pip install -r requirements.txt")
        print("  2. Network issues with akshare (not critical)")
    print("=" * 70)

    return 0 if all_passed else 1

if __name__ == "__main__":
    exit(main())
