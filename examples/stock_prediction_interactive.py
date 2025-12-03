# -*- coding: utf-8 -*-
"""
stock_prediction_interactive.py

Description:
    交互式股票预测程序，支持用户输入股票代码或上传TXT文件。
    使用Kronos模型和akshare，基于最近100个交易日的K线数据预测未来1个月（约20个交易日）的走势。

Usage:
    python stock_prediction_interactive.py

Features:
    - 交互式用户界面
    - 支持单个股票代码输入
    - 支持从TXT文件批量读取股票代码
    - 自动下载最近100个交易日的K线数据
    - 预测未来20个交易日
    - 保存预测结果到CSV和可视化图表

Author: Kronos Stock Prediction System
Date: 2025
"""

import os
import sys
import time
import pandas as pd
import numpy as np
import akshare as ak
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import subprocess
import platform

sys.path.append("../")
from model import Kronos, KronosTokenizer, KronosPredictor

# ==================== 配置参数 ====================
TOKENIZER_PRETRAINED = "NeoQuasar/Kronos-Tokenizer-base"
MODEL_PRETRAINED = "NeoQuasar/Kronos-base"
DEVICE = "cpu"  # 使用CPU，如果有GPU可改为 "cuda:0"
MAX_CONTEXT = 512
T = 1.0
TOP_P = 0.9
SAMPLE_COUNT = 1

# 输出目录
SAVE_DIR = "./stock_predictions"
os.makedirs(SAVE_DIR, exist_ok=True)


# ==================== 工具函数 ====================

def print_banner():
    """打印程序标题"""
    print("\n" + "=" * 70)
    print("     Kronos 股票预测系统 - 交互式版本")
    print("     支持自定义历史数据长度和预测长度")
    print("=" * 70 + "\n")


def print_menu():
    """打印菜单选项"""
    print("\n请选择操作模式：")
    print("  [1] 输入单个股票代码")
    print("  [2] 从TXT文件读取多个股票代码")
    print("  [3] 退出程序")
    print("-" * 70)


def get_user_choice():
    """获取用户选择"""
    while True:
        choice = input("请输入选项 (1/2/3): ").strip()
        if choice in ['1', '2', '3']:
            return choice
        print("❌ 无效选项，请重新输入！")


def get_stock_code_from_input():
    """从用户输入获取股票代码"""
    print("\n请输入股票代码（例如：000001, 600000, 002594）：")
    code = input("股票代码: ").strip()
    if code:
        return [code]
    else:
        print("❌ 股票代码不能为空！")
        return None


def get_stock_codes_from_file():
    """从TXT文件读取股票代码"""
    print("\n请输入TXT文件路径（每行一个股票代码）：")
    file_path = input("文件路径: ").strip()

    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return None

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            codes = [line.strip() for line in f if line.strip()]

        if codes:
            print(f"✅ 成功读取 {len(codes)} 个股票代码: {', '.join(codes)}")
            return codes
        else:
            print("❌ 文件为空！")
            return None
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        return None


def get_prediction_parameters():
    """获取用户自定义的预测参数"""
    print("\n" + "-" * 70)
    print("请设置预测参数：")

    # 获取历史数据长度
    while True:
        try:
            lookback = input("历史数据长度（建议30-200个交易日，默认100）: ").strip()
            if lookback == "":
                lookback = 100
            else:
                lookback = int(lookback)

            if lookback < 10:
                print("⚠️  历史数据长度至少需要10个交易日")
                continue
            elif lookback > 500:
                print("⚠️  历史数据长度不建议超过500个交易日")
                continue
            break
        except ValueError:
            print("❌ 请输入有效的数字！")

    # 获取预测长度
    while True:
        try:
            pred_len = input("预测长度（建议5-60个交易日，默认20）: ").strip()
            if pred_len == "":
                pred_len = 20
            else:
                pred_len = int(pred_len)

            if pred_len < 1:
                print("⚠️  预测长度至少需要1个交易日")
                continue
            elif pred_len > 100:
                print("⚠️  预测长度不建议超过100个交易日")
                continue
            break
        except ValueError:
            print("❌ 请输入有效的数字！")

    print(f"\n✅ 参数设置完成: 历史数据={lookback}天, 预测长度={pred_len}天")
    print("-" * 70)

    return lookback, pred_len


def load_stock_data(symbol: str, lookback: int = 100) -> pd.DataFrame:
    """
    使用akshare下载股票K线数据

    Args:
        symbol: 股票代码
        lookback: 需要的历史数据天数

    Returns:
        pd.DataFrame: 包含K线数据的DataFrame
    """
    print(f"\n📥 正在获取股票 {symbol} 的K线数据...")

    max_retries = 3
    df = None

    # 重试机制
    for attempt in range(1, max_retries + 1):
        try:
            # 使用akshare获取A股日线数据
            df = ak.stock_zh_a_hist(symbol=symbol, period="daily", adjust="")

            if df is not None and not df.empty:
                break
        except Exception as e:
            print(f"⚠️  尝试 {attempt}/{max_retries} 失败: {e}")
            if attempt < max_retries:
                time.sleep(1.5)

    # 检查是否成功获取数据
    if df is None or df.empty:
        print(f"❌ 无法获取股票 {symbol} 的数据，请检查股票代码是否正确！")
        return None

    # 重命名列为英文
    df.rename(columns={
        "日期": "date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount"
    }, inplace=True)

    # 转换日期格式
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # 转换数值列
    numeric_cols = ["open", "high", "low", "close", "volume", "amount"]
    for col in numeric_cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .replace({"--": None, "": None})
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 修复无效的开盘价
    open_bad = (df["open"] == 0) | (df["open"].isna())
    if open_bad.any():
        df.loc[open_bad, "open"] = df["close"].shift(1)
        df["open"].fillna(df["close"], inplace=True)

    # 修复缺失的成交额
    if df["amount"].isna().all() or (df["amount"] == 0).all():
        df["amount"] = df["close"] * df["volume"]

    # 只保留最近lookback+50天的数据（多取一些以确保有足够的交易日）
    if len(df) > lookback + 50:
        df = df.iloc[-(lookback + 50):].reset_index(drop=True)

    print(f"✅ 数据获取成功: {len(df)} 条记录, 日期范围: {df['date'].min()} ~ {df['date'].max()}")

    return df


def prepare_prediction_inputs(df: pd.DataFrame, lookback: int, pred_len: int):
    """
    准备预测所需的输入数据

    Args:
        df: 完整的历史数据
        lookback: 使用的历史数据长度
        pred_len: 预测长度

    Returns:
        tuple: (x_df, x_timestamp, y_timestamp)
    """
    # 确保有足够的数据
    if len(df) < lookback:
        print(f"⚠️  警告: 数据不足{lookback}个交易日，实际只有{len(df)}个交易日")
        lookback = len(df)

    # 取最近lookback个交易日的数据
    x_df = df.iloc[-lookback:][["open", "high", "low", "close", "volume", "amount"]]
    x_timestamp = df.iloc[-lookback:]["date"]

    # 生成未来pred_len个交易日的时间戳（跳过周末）
    y_timestamp = pd.bdate_range(
        start=df["date"].iloc[-1] + pd.Timedelta(days=1),
        periods=pred_len
    )

    return x_df, pd.Series(x_timestamp.values), pd.Series(y_timestamp)


def apply_price_limits(pred_df: pd.DataFrame, last_close: float, limit_rate: float = 0.1):
    """
    应用涨跌幅限制（中国A股±10%）

    Args:
        pred_df: 预测结果DataFrame
        last_close: 最后一个交易日的收盘价
        limit_rate: 涨跌幅限制（默认0.1即10%）

    Returns:
        pd.DataFrame: 应用限制后的预测结果
    """
    print(f"🔒 应用 ±{limit_rate*100:.0f}% 涨跌幅限制...")

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


def open_file(file_path: str):
    """
    使用系统默认程序打开文件

    Args:
        file_path: 文件路径
    """
    try:
        system = platform.system()
        if system == "Windows":
            os.startfile(file_path)
        elif system == "Darwin":  # macOS
            subprocess.run(["open", file_path], check=True)
        else:  # Linux
            subprocess.run(["xdg-open", file_path], check=True)
        print(f"✅ 已自动打开文件: {file_path}")
    except Exception as e:
        print(f"⚠️  无法自动打开文件: {e}")
        print(f"   请手动打开: {file_path}")


def plot_prediction_result(df_hist: pd.DataFrame, df_pred: pd.DataFrame, symbol: str, lookback: int, pred_len: int):
    """
    绘制预测结果图表

    Args:
        df_hist: 历史数据
        df_pred: 预测数据
        symbol: 股票代码
        lookback: 历史数据长度
        pred_len: 预测长度
    """
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=False)

    # 绘制收盘价
    ax1.plot(df_hist["date"], df_hist["close"],
             label='历史数据', color='blue', linewidth=2, marker='o', markersize=3)
    ax1.plot(df_pred["date"], df_pred["close"],
             label='预测数据', color='red', linewidth=2, linestyle='--', marker='s', markersize=3)
    ax1.set_ylabel('收盘价 (元)', fontsize=12, fontweight='bold')
    ax1.set_title(f'股票 {symbol} - Kronos预测结果 (最近{lookback}日 → 未来{pred_len}日)',
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=11)
    ax1.grid(True, alpha=0.3)

    # 绘制成交量
    ax2.bar(df_hist["date"], df_hist["volume"],
            label='历史成交量', color='blue', alpha=0.6, width=0.8)
    ax2.bar(df_pred["date"], df_pred["volume"],
            label='预测成交量', color='red', alpha=0.6, width=0.8)
    ax2.set_ylabel('成交量', fontsize=12, fontweight='bold')
    ax2.set_xlabel('日期', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图表
    plot_path = os.path.join(SAVE_DIR, f"prediction_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"📊 预测图表已保存: {plot_path}")

    # 自动打开图表文件
    open_file(plot_path)

    return plot_path


def predict_stock(symbol: str, predictor: KronosPredictor, lookback: int, pred_len: int):
    """
    对单个股票进行预测

    Args:
        symbol: 股票代码
        predictor: Kronos预测器实例
        lookback: 历史数据长度
        pred_len: 预测长度

    Returns:
        bool: 预测是否成功
    """
    print(f"\n{'='*70}")
    print(f"  开始预测股票: {symbol}")
    print(f"{'='*70}")

    # 1. 获取数据
    df = load_stock_data(symbol, lookback)
    if df is None:
        return False

    # 2. 准备输入
    try:
        x_df, x_timestamp, y_timestamp = prepare_prediction_inputs(df, lookback, pred_len)
        print(f"📋 输入数据准备完成:")
        print(f"   - 历史数据: {len(x_df)} 个交易日")
        print(f"   - 预测长度: {pred_len} 个交易日")
    except Exception as e:
        print(f"❌ 数据准备失败: {e}")
        return False

    # 3. 执行预测
    print(f"\n🔮 正在使用Kronos模型进行预测...")
    try:
        pred_df = predictor.predict(
            df=x_df,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=pred_len,
            T=T,
            top_p=TOP_P,
            sample_count=SAMPLE_COUNT,
            verbose=True
        )

        pred_df["date"] = y_timestamp.values

        # 应用涨跌幅限制
        last_close = df["close"].iloc[-1]
        pred_df = apply_price_limits(pred_df, last_close, limit_rate=0.1)

        print(f"✅ 预测完成！")

    except Exception as e:
        print(f"❌ 预测失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 4. 保存结果
    try:
        # 合并历史数据和预测数据
        df_hist_display = df.iloc[-30:] if len(df) > 30 else df  # 只显示最近30天历史

        # 保存CSV
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_path = os.path.join(SAVE_DIR, f"prediction_{symbol}_{timestamp_str}.csv")

        df_output = pd.concat([
            df[["date", "open", "high", "low", "close", "volume", "amount"]],
            pred_df[["date", "open", "high", "low", "close", "volume", "amount"]]
        ]).reset_index(drop=True)

        df_output.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"💾 预测结果已保存: {csv_path}")

        # 自动打开CSV文件
        open_file(csv_path)

        # 绘制并保存图表
        plot_prediction_result(df_hist_display, pred_df, symbol, lookback, pred_len)

        # 打印预测摘要
        print(f"\n📈 预测摘要:")
        print(f"   当前收盘价: {last_close:.2f} 元")
        print(f"   预测第1天收盘价: {pred_df['close'].iloc[0]:.2f} 元 "
              f"({'↑' if pred_df['close'].iloc[0] > last_close else '↓'} "
              f"{abs(pred_df['close'].iloc[0] / last_close - 1) * 100:.2f}%)")
        print(f"   预测最后一天收盘价: {pred_df['close'].iloc[-1]:.2f} 元 "
              f"({'↑' if pred_df['close'].iloc[-1] > last_close else '↓'} "
              f"{abs(pred_df['close'].iloc[-1] / last_close - 1) * 100:.2f}%)")
        print(f"   预测期间最高价: {pred_df['high'].max():.2f} 元")
        print(f"   预测期间最低价: {pred_df['low'].min():.2f} 元")

        return True

    except Exception as e:
        print(f"❌ 保存结果失败: {e}")
        import traceback
        traceback.print_exc()
        return False


# ==================== 主程序 ====================

def main():
    """主程序入口"""
    print_banner()

    # 加载模型（只加载一次）
    print("🚀 正在加载Kronos模型...")
    print(f"   - Tokenizer: {TOKENIZER_PRETRAINED}")
    print(f"   - Model: {MODEL_PRETRAINED}")
    print(f"   - Device: {DEVICE}")

    try:
        tokenizer = KronosTokenizer.from_pretrained(TOKENIZER_PRETRAINED)
        model = Kronos.from_pretrained(MODEL_PRETRAINED)
        predictor = KronosPredictor(model, tokenizer, device=DEVICE, max_context=MAX_CONTEXT)
        print("✅ 模型加载完成！\n")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        print("请确保已下载模型文件到 NeoQuasar/ 目录")
        return

    # 主循环
    while True:
        print_menu()
        choice = get_user_choice()

        if choice == '1':
            # 单个股票代码
            codes = get_stock_code_from_input()
            if codes:
                # 获取预测参数
                lookback, pred_len = get_prediction_parameters()
                predict_stock(codes[0], predictor, lookback, pred_len)

        elif choice == '2':
            # 从文件读取
            codes = get_stock_codes_from_file()
            if codes:
                # 获取预测参数（批量处理使用相同参数）
                lookback, pred_len = get_prediction_parameters()

                success_count = 0
                for i, code in enumerate(codes, 1):
                    print(f"\n{'#'*70}")
                    print(f"  处理进度: {i}/{len(codes)}")
                    print(f"{'#'*70}")
                    if predict_stock(code, predictor, lookback, pred_len):
                        success_count += 1
                    time.sleep(2)  # 避免请求过快

                print(f"\n{'='*70}")
                print(f"  批量预测完成！")
                print(f"  成功: {success_count}/{len(codes)}")
                print(f"{'='*70}")

        elif choice == '3':
            print("\n👋 感谢使用Kronos股票预测系统，再见！\n")
            break

        # 询问是否继续
        if choice in ['1', '2']:
            cont = input("\n是否继续预测？(y/n): ").strip().lower()
            if cont != 'y':
                print("\n👋 感谢使用Kronos股票预测系统，再见！\n")
                break


if __name__ == "__main__":
    main()
