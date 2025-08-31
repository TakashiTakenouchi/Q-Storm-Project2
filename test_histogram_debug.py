#!/usr/bin/env python3
"""
ヒストグラム表示バグの診断スクリプト
Excelデータを読み込んで、ヒストグラム生成の各段階で問題を診断します。
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys

# ========================================
# 設定
# ========================================
# Excelファイルパス（実際のパスに変更してください）
FILE_PATH = "/mnt/data/fixed_extended_store_data_2024-FIX_kaizen_monthlyvol3.xlsx"
# または、サンプルデータを使用する場合は以下を使用
USE_SAMPLE_DATA = True

# フィルタ設定
FILTER_BY = 'shop'  # 'shop' or 'shop_code' or None
FILTER_VALUE = None  # 実際の店舗名を設定

# テスト対象のKPI
TEST_METRICS = [
    "Total_Sales", "gross_profit", "discount", 
    "purchasing", "rent", "personnel_expenses"
]

# ========================================
# データ読み込みと診断
# ========================================
def create_sample_data():
    """サンプルデータの作成"""
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=200, freq='D')
    shops = ['新宿店', '渋谷店', '池袋店']
    
    data = []
    for date in dates:
        for shop in shops:
            data.append({
                'Date': date,
                'shop': shop,
                'shop_code': f'S{shops.index(shop)+1:03d}',
                'Total_Sales': np.random.normal(100000, 20000),
                'gross_profit': np.random.normal(40000, 8000),
                'discount': np.random.normal(5000, 1000),
                'purchasing': np.random.normal(60000, 12000),
                'rent': 50000,
                'personnel_expenses': np.random.normal(30000, 5000),
                'Operating_profit': np.random.normal(10000, 3000)
            })
    
    df = pd.DataFrame(data)
    # いくつかのNaN値を追加
    df.loc[df.index[:10], 'discount'] = np.nan
    df.loc[df.index[50:60], 'Total_Sales'] = '文字列データ'  # 数値変換エラーのテスト
    
    return df

def diagnose_dataframe(df, title="データフレーム診断"):
    """データフレームの診断情報を出力"""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    
    print(f"形状: {df.shape}")
    print(f"\n列名（最初の10列）:")
    for i, col in enumerate(df.columns[:10]):
        print(f"  [{i:2d}] '{col}' (型: {df[col].dtype})")
    
    # 列名の空白チェック
    cols_with_spaces = [col for col in df.columns if col != col.strip()]
    if cols_with_spaces:
        print(f"\n⚠️ 前後に空白がある列: {cols_with_spaces}")
    
    # 重複列名チェック
    duplicated = df.columns[df.columns.duplicated()]
    if len(duplicated) > 0:
        print(f"\n⚠️ 重複している列名: {list(duplicated)}")
    
    return df

def test_column_conversion(df, col_name):
    """列の数値変換テスト"""
    print(f"\n--- '{col_name}' の変換テスト ---")
    
    if col_name not in df.columns:
        print(f"❌ 列 '{col_name}' が存在しません")
        similar = [c for c in df.columns if col_name.lower() in c.lower()]
        if similar:
            print(f"   類似列: {similar}")
        return None
    
    series = df[col_name]
    print(f"元のデータ型: {series.dtype}")
    print(f"非NaN値の数: {series.notna().sum()}/{len(series)}")
    
    # ユニークな値の確認（最初の10個）
    unique_vals = series.dropna().unique()[:10]
    print(f"ユニーク値のサンプル: {unique_vals}")
    
    # 数値変換
    numeric_series = pd.to_numeric(series, errors='coerce')
    numeric_count = numeric_series.notna().sum()
    print(f"数値変換後の非NaN値: {numeric_count}/{len(series)}")
    
    if numeric_count > 0:
        print(f"統計情報:")
        print(f"  平均: {numeric_series.mean():.2f}")
        print(f"  中央値: {numeric_series.median():.2f}")
        print(f"  最小: {numeric_series.min():.2f}")
        print(f"  最大: {numeric_series.max():.2f}")
    else:
        print("⚠️ すべての値が数値に変換できませんでした")
    
    return numeric_series

def test_filter(df, filter_by, filter_value):
    """フィルタテスト"""
    print(f"\n--- フィルタテスト ---")
    print(f"フィルタ: {filter_by} == '{filter_value}'")
    
    if filter_by and filter_value:
        if filter_by not in df.columns:
            print(f"❌ フィルタ列 '{filter_by}' が存在しません")
            return df
        
        unique_vals = df[filter_by].unique()
        print(f"{filter_by}のユニーク値: {unique_vals[:10]}")
        
        filtered = df[df[filter_by] == filter_value]
        print(f"フィルタ前: {len(df)} 行")
        print(f"フィルタ後: {len(filtered)} 行")
        
        if len(filtered) == 0:
            print(f"⚠️ フィルタ結果が空です。'{filter_value}' が存在しない可能性があります")
        
        return filtered
    else:
        print("フィルタなし")
        return df

def create_histogram_with_debug(df, metric, title="ヒストグラム"):
    """デバッグ情報付きでヒストグラムを作成"""
    print(f"\n--- ヒストグラム作成: {metric} ---")
    
    if metric not in df.columns:
        print(f"❌ 列 '{metric}' が存在しません")
        return None
    
    # 数値変換
    series = pd.to_numeric(df[metric], errors='coerce').dropna()
    
    if len(series) == 0:
        print(f"⚠️ '{metric}' に有効な数値データがありません")
        # 空のグラフに注記を追加
        fig = go.Figure()
        fig.add_annotation(
            text=f"データがありません<br>列: {metric}<br>すべての値が数値に変換できませんでした",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14)
        )
        fig.update_layout(
            title=f"{title}: {metric}（データなし）",
            xaxis_title=metric,
            yaxis_title="Count",
            height=400
        )
        return fig
    
    print(f"✅ 有効なデータ: {len(series)} 件")
    
    # ヒストグラム作成
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=series,
        nbinsx=30,
        name=metric,
        marker_color='lightblue',
        hovertemplate=(
            f'<b>{metric}</b><br>' +
            '範囲: %{x}<br>' +
            'カウント: %{y}<br>' +
            '<extra></extra>'
        )
    ))
    
    # 統計情報を注記として追加
    stats_text = (
        f"平均: {series.mean():.2f}<br>"
        f"中央値: {series.median():.2f}<br>"
        f"件数: {len(series)}"
    )
    fig.add_annotation(
        text=stats_text,
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        showarrow=False,
        font=dict(size=10),
        bgcolor="rgba(255,255,255,0.8)",
        align="left"
    )
    
    fig.update_layout(
        title=f"{title}: {metric}",
        xaxis_title=metric,
        yaxis_title="Count",
        bargap=0.05,
        height=400
    )
    
    return fig

# ========================================
# メイン処理
# ========================================
def main():
    print("Q-Storm ヒストグラム診断ツール")
    print("="*60)
    
    # データ読み込み
    if USE_SAMPLE_DATA:
        print("サンプルデータを使用します")
        df = create_sample_data()
    else:
        if not Path(FILE_PATH).exists():
            print(f"❌ ファイルが見つかりません: {FILE_PATH}")
            print("サンプルデータを使用します")
            df = create_sample_data()
        else:
            print(f"Excelファイルを読み込み中: {FILE_PATH}")
            df = pd.read_excel(FILE_PATH)
    
    # 列名の正規化
    print("\n列名を正規化中...")
    df.columns = [str(c).strip().replace('  ', ' ') for c in df.columns]
    
    # データフレーム診断
    df = diagnose_dataframe(df, "読み込み後のデータ診断")
    
    # フィルタ適用
    df_filtered = test_filter(df, FILTER_BY, FILTER_VALUE)
    
    # 各メトリクスのテスト
    print(f"\n{'='*60}")
    print("メトリクス別テスト")
    print(f"{'='*60}")
    
    for metric in TEST_METRICS:
        test_column_conversion(df_filtered, metric)
    
    # ヒストグラム作成テスト
    print(f"\n{'='*60}")
    print("ヒストグラム作成テスト")
    print(f"{'='*60}")
    
    for metric in TEST_METRICS[:3]:  # 最初の3つだけテスト
        fig = create_histogram_with_debug(df_filtered, metric)
        if fig:
            # 実際に表示したい場合はコメントアウトを外す
            # fig.show()
            print(f"✅ '{metric}' のヒストグラムを作成しました")
    
    print(f"\n{'='*60}")
    print("診断完了")
    print(f"{'='*60}")
    
    # 利用可能な店舗リスト
    if 'shop' in df.columns:
        shops = df['shop'].dropna().unique()
        print(f"\n利用可能な店舗: {list(shops[:10])}")
    
    return df

if __name__ == "__main__":
    df = main()