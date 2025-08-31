#!/usr/bin/env python3
"""
Data Analysis Agent - ステップ1&2の実装
既存機能を維持しながらエージェント化
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, Optional, List
import logging
import json
import traceback

# 既存のモジュールをインポート
from advanced_feature_engineering import (
    AdvancedDataQualityAnalyzer,
    LangChainMissingValueHandler,
    ComprehensiveFeatureEngineering
)

logger = logging.getLogger(__name__)

class DataAnalysisAgent:
    """
    QCストーリー ステップ1（テーマ選定）とステップ2（現状把握）を担当
    既存のapp.pyの機能をエージェント化
    """
    
    def __init__(self):
        self.name = "DataAnalysisAgent"
        self.field_name_mapping = self._get_field_mapping()
        self.reverse_mapping = {v: k for k, v in self.field_name_mapping.items()}
        
    def _get_field_mapping(self) -> Dict[str, str]:
        """フィールド名マッピングを返す（app.pyから移植）"""
        return {
            'shop': '店舗名', 'shop_code': '店舗コード', 'Date': '営業日付', 
            'Total_Sales': '店舗別売上高', 'gross_profit': '売上総利益（粗利）', 
            'discount': '値引・割引（月額）', 'purchasing': '仕入高',
            'rent': '家賃', 'personnel_expenses': '人件費', 
            'depreciation': '減価償却費', 'sales_promotion': '販売促進費', 
            'head_office_expenses': '本部費用配賦', 'operating_cost': '営業経費',
            'Operating_profit': '営業利益', 
            'Mens_JACKETS&OUTER2': 'メンズ ジャケット・アウター 売上高',
            'Mens_KNIT': 'メンズ ニット 売上高', 
            'Mens_PANTS': 'メンズ パンツ 売上高',
            'WOMEN\'S_JACKETS2': 'レディース ジャケット 売上高', 
            'WOMEN\'S_TOPS': 'レディース トップス 売上高',
            'WOMEN\'S_ONEPIECE': 'レディース ワンピース 売上高', 
            'WOMEN\'S_bottoms': 'レディース ボトムス 売上高',
            'WOMEN\'S_SCARF & STOLES': 'レディース スカーフ・ストール 売上高',
            'Inventory': '在庫金額', 'Months_of_inventory': '在庫月数', 
            'BEP': '損益分岐点（BEP）', 'Average_Temperature': '平均気温',
            'Number_of_guests': '来客数', 'Price_per_customer': '客単価',
            'judge': '判定（評価）'
        }
    
    def analyze_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        ステップ1: 特徴量分析（テーマ選定）
        app.pyのget_feature_analysis()を移植
        """
        try:
            analyzer = AdvancedDataQualityAnalyzer(df)
            field_info_df, _ = analyzer.display_field_analysis()
            
            results = []
            numeric_cols = []
            categorical_cols = []
            date_cols = []
            
            for record in field_info_df.to_dict('records'):
                col_name = record['フィールド名']
                jp_name = self.field_name_mapping.get(col_name, col_name)
                record['日本語名'] = jp_name
                
                # データ型カテゴリを判定
                dtype_category = self._get_data_type_category(df[col_name])
                record['データ型カテゴリ'] = dtype_category
                
                # 列を分類
                if col_name in ['shop', 'shop_code', 'judge']:
                    categorical_cols.append(jp_name)
                elif dtype_category == '日付/時間':
                    date_cols.append(jp_name)
                else:
                    numeric_cols.append(jp_name)
                
                results.append(record)
            
            logger.info(f"Feature analysis completed: {len(numeric_cols)} numeric, {len(categorical_cols)} categorical")
            
            return {
                'status': 'success',
                'feature_analysis': results,
                'columns_info': {
                    'numeric': numeric_cols,
                    'categorical': categorical_cols,
                    'date': date_cols
                }
            }
        except Exception as e:
            logger.error(f"Feature analysis error: {e}\n{traceback.format_exc()}")
            return {'status': 'error', 'message': str(e)}
    
    def propose_improvement(self, df: pd.DataFrame, field_name: str) -> Dict[str, Any]:
        """
        欠損値処理の提案
        app.pyのpropose_improvement()を移植
        """
        try:
            handler = LangChainMissingValueHandler()
            if df[field_name].isnull().sum() > 0:
                proposals = handler.interactive_missing_value_handler(df, field_name)
                return {
                    'status': 'success',
                    'field_name': field_name,
                    'suggestion': proposals['suggestion'],
                    'options': proposals['options']
                }
            else:
                return {
                    'status': 'info',
                    'message': 'このフィールドに欠損値の問題は見つかりませんでした。'
                }
        except Exception as e:
            logger.error(f"Proposal error: {e}\n{traceback.format_exc()}")
            return {'status': 'error', 'message': str(e)}
    
    def apply_improvement(self, df: pd.DataFrame, field_name: str, method: str) -> Dict[str, Any]:
        """
        改善策の適用
        app.pyのapply_improvement()を移植
        """
        try:
            analyzer = ComprehensiveFeatureEngineering(df)
            treated_series = analyzer.apply_missing_value_treatment(field_name, method)
            df[field_name] = treated_series
            
            return {
                'status': 'success',
                'message': f'フィールド「{self.field_name_mapping.get(field_name, field_name)}」に「{method}」を適用しました。',
                'updated_df': df
            }
        except Exception as e:
            logger.error(f"Apply improvement error: {e}\n{traceback.format_exc()}")
            return {'status': 'error', 'message': str(e)}
    
    def generate_histogram(self, df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        ステップ2: ヒストグラム生成（現状把握）
        app.pyのgenerate_graph()のhistogram部分を移植
        """
        try:
            target_col_jp = params.get('target_col')
            group_by_col_jp = params.get('group_by_col')
            filter_shop = params.get('filter_shop')
            
            # 日本語名から英語フィールド名に変換
            target_col = self.reverse_mapping.get(target_col_jp, target_col_jp)
            
            logger.info(f"Generating histogram: target={target_col}, filter={filter_shop}")
            
            # データフレームのコピー
            df_copy = df.copy()
            
            # 店舗フィルタ適用
            if filter_shop and str(filter_shop).strip() != '':
                df_copy = self._apply_shop_filter(df_copy, filter_shop)
            
            if len(df_copy) == 0:
                return self._create_empty_histogram(target_col_jp, filter_shop, df)
            
            # target_colの存在確認
            if target_col not in df_copy.columns:
                return {
                    'status': 'error',
                    'message': f'列 「{target_col}」 が見つかりません。'
                }
            
            # 数値データに変換
            df_copy[target_col] = pd.to_numeric(df_copy[target_col], errors='coerce')
            df_plot = df_copy.dropna(subset=[target_col])
            
            if len(df_plot) == 0:
                return self._create_empty_histogram(target_col_jp, filter_shop, df)
            
            # 統計情報を計算
            stats = {
                'mean': df_plot[target_col].mean(),
                'median': df_plot[target_col].median(),
                'std': df_plot[target_col].std(),
                'min': df_plot[target_col].min(),
                'max': df_plot[target_col].max(),
                'count': len(df_plot)
            }
            
            title = f'{target_col_jp}の分布'
            if filter_shop:
                title += f'（{filter_shop}）'
            
            # グラフ作成
            if group_by_col_jp and group_by_col_jp != '':
                group_by_col = self.reverse_mapping.get(group_by_col_jp, group_by_col_jp)
                if group_by_col in df_plot.columns:
                    fig = px.histogram(
                        df_plot, x=target_col, color=group_by_col,
                        title=f'{title}（{group_by_col_jp}別）',
                        labels={target_col: target_col_jp, group_by_col: group_by_col_jp},
                        nbins=30
                    )
                else:
                    fig = self._create_basic_histogram(df_plot, target_col, target_col_jp, title, stats)
            else:
                fig = self._create_basic_histogram(df_plot, target_col, target_col_jp, title, stats)
            
            fig.update_layout(
                template="plotly_dark",
                font=dict(family="sans-serif", size=12, color="white"),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0.1)',
                height=500,
                bargap=0.05
            )
            
            return {
                'status': 'success',
                'graph_json': json.loads(fig.to_json())
            }
            
        except Exception as e:
            logger.error(f"Histogram generation error: {e}\n{traceback.format_exc()}")
            return {'status': 'error', 'message': str(e)}
    
    def generate_line_chart(self, df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        ステップ2: 折れ線グラフ生成（現状把握）
        app.pyのgenerate_graph()のline部分を移植
        """
        try:
            date_col_jp = params.get('date_col')
            value_col_jp = params.get('value_col')
            filter_shop = params.get('filter_shop')
            date_start = params.get('date_start')
            date_end = params.get('date_end')
            
            # 日本語名から英語フィールド名に変換
            date_col = self.reverse_mapping.get(date_col_jp, date_col_jp)
            value_col = self.reverse_mapping.get(value_col_jp, value_col_jp)
            
            logger.info(f"Line chart: date={date_col}, value={value_col}, shop={filter_shop}")
            
            # データフレームのコピー
            df_copy = df.copy()
            
            # 店舗フィルタ適用
            if filter_shop and str(filter_shop).strip() != '':
                df_copy = self._apply_shop_filter(df_copy, filter_shop)
            
            # 列の存在確認
            if date_col not in df_copy.columns:
                return {'status': 'error', 'message': f'日付列 {date_col} が見つかりません'}
            if value_col not in df_copy.columns:
                return {'status': 'error', 'message': f'値列 {value_col} が見つかりません'}
            
            # データ型の変換
            df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')
            df_copy[value_col] = pd.to_numeric(df_copy[value_col], errors='coerce')
            df_copy = df_copy.dropna(subset=[date_col, value_col])
            
            # 日付範囲フィルタ
            if date_start:
                df_copy = df_copy[df_copy[date_col] >= pd.to_datetime(date_start)]
            if date_end:
                df_copy = df_copy[df_copy[date_col] <= pd.to_datetime(date_end)]
            
            if len(df_copy) == 0:
                return {'status': 'error', 'message': '指定された条件に該当するデータがありません'}
            
            # 日付でソート
            df_copy = df_copy.sort_values(by=date_col)
            
            # タイトル作成
            title = f'{value_col_jp}の時系列推移'
            if filter_shop:
                title += f'（{filter_shop}）'
            
            # 店舗別グラフの作成
            if 'shop' in df_copy.columns and df_copy['shop'].nunique() > 1:
                fig = px.line(
                    df_copy, x=date_col, y=value_col, color='shop',
                    title=title,
                    labels={date_col: date_col_jp, value_col: value_col_jp, 'shop': '店舗名'},
                    markers=True
                )
            else:
                fig = px.line(
                    df_copy, x=date_col, y=value_col,
                    title=title,
                    labels={date_col: date_col_jp, value_col: value_col_jp},
                    markers=True
                )
            
            fig.update_xaxes(rangeslider_visible=True)
            fig.update_layout(
                template="plotly_dark",
                font=dict(family="sans-serif", size=12, color="white"),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0.1)',
                hovermode='x unified',
                height=500
            )
            
            return {
                'status': 'success',
                'graph_json': json.loads(fig.to_json())
            }
            
        except Exception as e:
            logger.error(f"Line chart generation error: {e}\n{traceback.format_exc()}")
            return {'status': 'error', 'message': str(e)}
    
    def _get_data_type_category(self, series: pd.Series) -> str:
        """データ型カテゴリの判定"""
        dtype = str(series.dtype)
        col_name = series.name
        
        categorical_fields = ['shop', 'shop_code', 'judge']
        if col_name in categorical_fields:
            return 'カテゴリ'
        
        if 'datetime' in dtype:
            return '日付/時間'
        elif 'int' in dtype or 'float' in dtype:
            return '数値'
        elif 'object' in dtype or 'category' in dtype:
            try:
                pd.to_datetime(series, errors='raise')
                return '日付/時間'
            except:
                if series.nunique() < 10:
                    return 'カテゴリ'
                else:
                    return 'テキスト'
        return '不明'
    
    def _apply_shop_filter(self, df: pd.DataFrame, filter_shop: str) -> pd.DataFrame:
        """店舗フィルタの適用"""
        if 'shop' not in df.columns:
            return df
        
        filter_shop_clean = str(filter_shop).strip()
        df['shop'] = df['shop'].astype(str).str.strip()
        
        # まず完全一致を試す
        mask = df['shop'] == filter_shop_clean
        if mask.sum() > 0:
            return df[mask]
        
        # 部分一致を試す
        mask = df['shop'].str.contains(filter_shop_clean, na=False, case=False)
        if mask.sum() > 0:
            return df[mask]
        
        # マッチしない場合は空のDataFrame
        return df.iloc[0:0]
    
    def _create_basic_histogram(self, df: pd.DataFrame, col: str, col_jp: str, 
                               title: str, stats: Dict) -> go.Figure:
        """基本的なヒストグラムを作成"""
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=df[col],
            nbinsx=30,
            name=col_jp,
            hovertemplate=(
                f'<b>{col_jp}</b><br>' +
                '範囲: %{x}<br>' +
                'カウント: %{y}<br>' +
                f'<br><b>統計情報:</b><br>' +
                f'平均: {stats["mean"]:.2f}<br>' +
                f'中央値: {stats["median"]:.2f}<br>' +
                f'標準偏差: {stats["std"]:.2f}<br>' +
                f'最小: {stats["min"]:.2f}<br>' +
                f'最大: {stats["max"]:.2f}<br>' +
                f'件数: {stats["count"]}<br>' +
                '<extra></extra>'
            )
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title=col_jp,
            yaxis_title="Count"
        )
        return fig
    
    def _create_empty_histogram(self, target_col_jp: str, filter_shop: str, 
                               df: pd.DataFrame) -> Dict[str, Any]:
        """空のヒストグラムを作成（データなし時）"""
        available_shops = df['shop'].dropna().unique().tolist() if 'shop' in df.columns else []
        
        shops_to_display = available_shops[:20]
        if len(available_shops) > 20:
            shops_display_text = ', '.join(shops_to_display) + f'... (他{len(available_shops)-20}店舗)'
        else:
            shops_display_text = ', '.join(shops_to_display) if shops_to_display else '店舗データなし'
        
        error_message = (f"データが見つかりませんでした<br><br>"
                        f"<b>入力された店舗名:</b> '{filter_shop}'<br><br>"
                        f"<b>利用可能な店舗名({len(available_shops)}店舗):</b><br>"
                        f"<small>{shops_display_text}</small>")
        
        fig = go.Figure()
        fig.add_annotation(
            text=error_message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=12),
            align="center"
        )
        fig.update_layout(
            title=f"{target_col_jp}の分布（データなし）",
            xaxis_title=target_col_jp,
            yaxis_title="Count",
            height=500
        )
        
        return {
            'status': 'success',
            'graph_json': json.loads(fig.to_json())
        }