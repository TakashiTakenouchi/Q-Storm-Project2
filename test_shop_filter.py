#!/usr/bin/env python3
"""
店舗フィルタのデバッグスクリプト
特に「恵比寿」店のフィルタリング問題を診断します。
"""

import pandas as pd
import numpy as np
import json

def test_shop_filter():
    """店舗フィルタのテスト"""
    
    # サンプルデータの作成（実際のデータ構造を模倣）
    print("=" * 60)
    print("店舗フィルタテスト")
    print("=" * 60)
    
    # 実際によくある店舗名のパターン
    shops_patterns = [
        "恵比寿",           # 基本形
        "恵比寿店",         # 「店」付き
        " 恵比寿",          # 前にスペース
        "恵比寿 ",          # 後ろにスペース
        " 恵比寿 ",         # 前後にスペース
        "恵比寿　",         # 全角スペース
        "エビス",           # カタカナ
        "EBISU",           # 英語
        "恵比寿駅前店",     # 詳細な店舗名
        "恵比寿ガーデン",   # 別の恵比寿店
    ]
    
    # データフレーム作成
    data = []
    for shop in shops_patterns:
        for i in range(10):
            data.append({
                'shop': shop,
                'Total_Sales': np.random.randint(50000, 150000),
                'Date': pd.Timestamp('2024-01-01') + pd.Timedelta(days=i)
            })
    
    df = pd.DataFrame(data)
    
    print(f"\n元のデータ:")
    print(f"データ件数: {len(df)}")
    print(f"ユニークな店舗名: {df['shop'].unique().tolist()}")
    
    # フィルタテスト
    test_filters = [
        "恵比寿",
        "恵比寿店",
        " 恵比寿",
        "恵比寿 ",
        "えびす",
        "エビス"
    ]
    
    print("\n" + "=" * 60)
    print("フィルタテスト結果")
    print("=" * 60)
    
    for filter_text in test_filters:
        print(f"\nフィルタ: '{filter_text}'")
        print(f"  (repr: {repr(filter_text)})")
        
        # 方法1: 完全一致
        df_copy = df.copy()
        exact_match = df_copy[df_copy['shop'] == filter_text]
        print(f"  完全一致: {len(exact_match)} 件")
        
        # 方法2: strip後の完全一致
        df_copy = df.copy()
        df_copy['shop'] = df_copy['shop'].str.strip()
        strip_match = df_copy[df_copy['shop'] == filter_text.strip()]
        print(f"  strip後完全一致: {len(strip_match)} 件")
        
        # 方法3: 部分一致（contains）
        df_copy = df.copy()
        partial_match = df_copy[df_copy['shop'].str.contains(filter_text, na=False, case=False)]
        print(f"  部分一致: {len(partial_match)} 件")
        if len(partial_match) > 0:
            matched_shops = partial_match['shop'].unique().tolist()
            print(f"    マッチした店舗: {matched_shops}")
        
        # 方法4: strip + 部分一致（推奨方法）
        df_copy = df.copy()
        df_copy['shop'] = df_copy['shop'].str.strip()
        filter_clean = filter_text.strip()
        
        # まず完全一致
        mask = df_copy['shop'] == filter_clean
        if mask.sum() == 0:
            # 完全一致がなければ部分一致
            mask = df_copy['shop'].str.contains(filter_clean, na=False, case=False)
        
        recommended_match = df_copy[mask]
        print(f"  推奨方法（strip+部分一致）: {len(recommended_match)} 件")
        if len(recommended_match) > 0:
            matched_shops = recommended_match['shop'].unique().tolist()
            print(f"    マッチした店舗: {matched_shops[:5]}")  # 最初の5件

def test_real_data_structure():
    """実際のデータ構造をシミュレート"""
    print("\n" + "=" * 60)
    print("実データ構造のシミュレーション")
    print("=" * 60)
    
    # 実際のExcelファイルでよくあるパターン
    df = pd.DataFrame({
        'shop': ['新宿店', '渋谷店', '恵比寿', '池袋店', '品川店'] * 20,
        'shop_code': ['S001', 'S002', 'S003', 'S004', 'S005'] * 20,
        'Total_Sales': np.random.randint(50000, 200000, 100),
        'Operating_profit': np.random.randint(5000, 50000, 100)
    })
    
    print(f"データ形状: {df.shape}")
    print(f"店舗リスト: {df['shop'].unique().tolist()}")
    
    # 「恵比寿」でフィルタ
    filter_shop = "恵比寿"
    print(f"\n'{filter_shop}'でフィルタリング:")
    
    # ステップ1: 利用可能な店舗を確認
    available_shops = df['shop'].dropna().unique().tolist()
    print(f"利用可能な店舗: {available_shops}")
    
    # ステップ2: フィルタ適用
    df_filtered = df[df['shop'] == filter_shop]
    print(f"フィルタ結果: {len(df_filtered)} 件")
    
    if len(df_filtered) > 0:
        print(f"成功: データが見つかりました")
        print(f"サンプルデータ:")
        print(df_filtered.head())
    else:
        print(f"エラー: データが見つかりませんでした")
        # 類似する店舗名を探す
        similar = [s for s in available_shops if filter_shop in s or s in filter_shop]
        print(f"類似する店舗名: {similar}")

def generate_test_recommendations():
    """推奨されるデバッグ手順"""
    print("\n" + "=" * 60)
    print("推奨デバッグ手順")
    print("=" * 60)
    
    recommendations = [
        "1. データアップロード後、ブラウザの開発者ツール（F12）でコンソールログを確認",
        "2. /api/get_available_shops APIの応答を確認して実際の店舗名を確認",
        "3. ヒストグラム作成時のネットワークタブで/api/generate_graphリクエストのペイロードを確認",
        "4. サーバーログで「Available shops in data:」を確認",
        "5. 店舗名に特殊文字（スペース、全角文字など）が含まれていないか確認",
        "6. 店舗フィルタなしでヒストグラムが表示されることを確認",
        "7. 他の店舗名でフィルタが機能するか確認"
    ]
    
    for rec in recommendations:
        print(f"  {rec}")
    
    print("\n一般的な問題と解決策:")
    issues = {
        "店舗名の不一致": "データの店舗名と入力した店舗名が完全に一致していない（スペースなど）",
        "文字コード": "全角・半角の違い、見えない文字（改行、タブなど）",
        "データ型": "shop列が文字列型でない可能性",
        "空白文字": "前後の空白、連続する空白、全角スペース",
        "NULL値": "shop列にNULL値が多く含まれている"
    }
    
    for issue, solution in issues.items():
        print(f"  問題: {issue}")
        print(f"    → 解決策: {solution}")

if __name__ == "__main__":
    # テスト実行
    test_shop_filter()
    test_real_data_structure()
    generate_test_recommendations()
    
    print("\n" + "=" * 60)
    print("テスト完了")
    print("=" * 60)