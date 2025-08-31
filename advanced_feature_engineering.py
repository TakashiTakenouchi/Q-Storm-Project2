#!/usr/bin/env python3
"""
Q-Storm Platform v3.0.0 - Advanced Feature Engineering
LangChain統合 + AutoViz自動EDA + インタラクティブ処理
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
from typing import Dict, List, Tuple, Any, Optional
import logging
from datetime import datetime

# LangChain imports (オプショナル)
try:
    from langchain.llms import OpenAI
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("LangChain not available. Using local NLP engine.")

# AutoViz imports (オプショナル)
try:
    from autoviz.AutoViz_Class import AutoViz_Class
    AUTOVIZ_AVAILABLE = True
except ImportError:
    AUTOVIZ_AVAILABLE = False
    print("AutoViz not available. Using built-in visualization.")

logger = logging.getLogger(__name__)

class AdvancedDataQualityAnalyzer:
    """データ品質分析クラス"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.field_analysis = None
        self.type_inconsistencies = None
        self.outlier_results = None
        
    def display_field_analysis(self) -> Tuple[pd.DataFrame, List[Dict]]:
        """
        Tabular形式でフィールド詳細情報を表示
        """
        field_info_data = []
        
        for col in self.df.columns:
            # 実際の型を安全に取得
            try:
                if not self.df[col].dropna().empty:
                    actual_type = type(self.df[col].dropna().iloc[0]).__name__
                    sample_value = str(self.df[col].dropna().iloc[0])[:50]  # 最大50文字
                else:
                    actual_type = 'N/A'
                    sample_value = 'N/A'
            except Exception:
                actual_type = 'Unknown'
                sample_value = 'N/A'
            
            field_info_data.append({
                'フィールド名': col,
                'データ型': str(self.df[col].dtype),
                '実際の型': actual_type,
                '非null件数': int(self.df[col].count()),
                '欠損値件数': int(self.df[col].isnull().sum()),
                '欠損率(%)': round(self.df[col].isnull().sum() / len(self.df) * 100, 2),
                'ユニーク値数': int(self.df[col].nunique()),
                'サンプル値': sample_value
            })
        
        field_info = pd.DataFrame(field_info_data)
        
        # データ型不整合チェック
        type_mismatch = []
        for col in self.df.columns:
            try:
                expected_type = self.df[col].dtype
                # 各値の型をチェック
                type_counts = {}
                for val in self.df[col].dropna():
                    val_type = type(val).__name__
                    type_counts[val_type] = type_counts.get(val_type, 0) + 1
                
                if len(type_counts) > 1:
                    type_mismatch.append({
                        'フィールド': col,
                        '期待型': str(expected_type),
                        '実際の型分布': type_counts
                    })
            except Exception as e:
                logger.warning(f"Type check error for column {col}: {e}")
        
        self.field_analysis = field_info
        self.type_inconsistencies = type_mismatch
        
        return field_info, type_mismatch
    
    def identify_type_inconsistencies(self) -> pd.DataFrame:
        """
        フィールドのデータ型と実データの不整合を特定
        """
        inconsistencies = []
        
        for col in self.df.columns:
            expected_type = self.df[col].dtype
            
            for idx, value in enumerate(self.df[col]):
                if pd.notna(value):
                    # オブジェクト型なのに数値
                    if expected_type == 'object' and isinstance(value, (int, float)):
                        inconsistencies.append({
                            'フィールド': col,
                            '行番号': idx,
                            '値': str(value)[:50],
                            '期待型': 'text/string',
                            '実際型': type(value).__name__
                        })
                    # 数値型なのに文字列
                    elif expected_type in ['int64', 'float64'] and isinstance(value, str):
                        try:
                            # 数値に変換可能かチェック
                            float(value)
                        except ValueError:
                            inconsistencies.append({
                                'フィールド': col,
                                '行番号': idx,
                                '値': str(value)[:50],
                                '期待型': 'numeric',
                                '実際型': 'string'
                            })
                
                # 最初の100行のみチェック（パフォーマンス対策）
                if idx >= 100:
                    break
        
        return pd.DataFrame(inconsistencies)
    
    def detect_outliers_3sigma(self, exclude_fields: List[str] = None) -> Tuple[Dict, List]:
        """
        3σルールによる外れ値検出とヒストグラム作成
        """
        if exclude_fields is None:
            exclude_fields = ['shop_code', 'id', 'index', 'code']
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        # 除外フィールドを適用
        numeric_cols = [col for col in numeric_cols 
                       if col.lower() not in [field.lower() for field in exclude_fields]]
        
        outlier_results = {}
        plots = []
        
        for col in numeric_cols:
            if col in self.df.columns:
                # 基本統計量
                mean = self.df[col].mean()
                std = self.df[col].std()
                
                # 3σ閾値
                upper_bound = mean + 3 * std
                lower_bound = mean - 3 * std
                
                # 外れ値特定
                outliers_mask = (self.df[col] > upper_bound) | (self.df[col] < lower_bound)
                outliers = self.df[outliers_mask]
                
                # ヒストグラム作成（3σライン付き）
                fig = go.Figure()
                
                # ヒストグラム
                fig.add_trace(go.Histogram(
                    x=self.df[col].dropna(),
                    name=f'{col} 分布',
                    opacity=0.7,
                    marker_color='blue'
                ))
                
                # 3σライン
                fig.add_vline(x=upper_bound, line_dash="dash", line_color="red", 
                             annotation_text=f"上限3σ ({upper_bound:.2f})")
                fig.add_vline(x=lower_bound, line_dash="dash", line_color="red", 
                             annotation_text=f"下限3σ ({lower_bound:.2f})")
                
                # 平均線
                fig.add_vline(x=mean, line_dash="solid", line_color="green", 
                             annotation_text=f"平均 ({mean:.2f})")
                
                fig.update_layout(
                    title=f'{col} の分布と外れ値検出（3σルール）',
                    xaxis_title=col,
                    yaxis_title='頻度',
                    height=400,
                    showlegend=True
                )
                
                outlier_results[col] = {
                    'mean': float(mean),
                    'std': float(std),
                    'upper_bound': float(upper_bound),
                    'lower_bound': float(lower_bound),
                    'outlier_count': int(len(outliers)),
                    'outlier_percentage': round(len(outliers) / len(self.df) * 100, 2),
                    'outlier_indices': outliers.index.tolist()[:10],  # 最初の10個
                    'outlier_values': outliers[col].tolist()[:10]  # 最初の10個
                }
                
                plots.append({
                    'column': col,
                    'plot': json.loads(fig.to_json())
                })
        
        self.outlier_results = outlier_results
        return outlier_results, plots

class LangChainMissingValueHandler:
    """LangChain統合欠損値処理クラス"""
    
    def __init__(self):
        self.llm = None
        self.setup_langchain()
    
    def setup_langchain(self):
        """LangChain初期化（OpenAI APIキー不要の代替実装も含む）"""
        if LANGCHAIN_AVAILABLE:
            try:
                # OpenAI APIキーがある場合
                self.llm = OpenAI(temperature=0.7)
            except Exception:
                # 代替：ローカル自然言語生成
                self.llm = self.create_local_nlp_engine()
        else:
            self.llm = self.create_local_nlp_engine()
    
    def create_local_nlp_engine(self):
        """ローカル自然言語生成エンジン（LangChain不要）"""
        class LocalNLPEngine:
            def generate(self, prompt: str) -> str:
                return self.generate_response(prompt)
            
            def generate_response(self, prompt: str) -> str:
                """テンプレートベースの応答生成"""
                if "欠損値" in prompt:
                    return "欠損値の処理方法を選択してください。"
                return "データ分析を実行します。"
        
        return LocalNLPEngine()
    
    def generate_missing_value_suggestions(self, field_name: str, missing_count: int, 
                                          total_count: int, field_type: str) -> str:
        """欠損値に対する自然言語提案生成"""
        missing_rate = round((missing_count / total_count) * 100, 2)
        
        suggestion = f"""
        📊 データ分析レポート
        
        フィールド「{field_name}」の欠損値分析:
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        📈 統計情報:
        • 欠損値数: {missing_count}件
        • 全データ数: {total_count}件
        • 欠損率: {missing_rate}%
        • データ型: {field_type}
        
        💡 推奨される対処法:
        
        1️⃣ 代表値による補完
           • 平均値: 数値データの一般的な補完方法
           • 中央値: 外れ値の影響を受けにくい補完
           • 最頻値: カテゴリカルデータに適した補完
        
        2️⃣ 予測による補完
           • 線形補間: 前後の値から推定
           • 回帰予測: 他の変数から予測
        
        3️⃣ 欠損値の削除
           • 行削除: 欠損値を含む行を削除
           • 列削除: 欠損率が高い場合の選択肢
        
        ⚠️ 注意事項:
        """
        
        if missing_rate < 5:
            suggestion += "• 欠損率が低いため、単純な補完で問題ありません"
        elif missing_rate < 20:
            suggestion += "• 中程度の欠損率です。補完方法の選択が重要です"
        else:
            suggestion += "• 欠損率が高いため、慎重な処理が必要です"
        
        return suggestion
    
    def interactive_missing_value_handler(self, df: pd.DataFrame, field_name: str) -> Dict:
        """段階的ユーザー選択による欠損値処理"""
        missing_count = df[field_name].isnull().sum()
        total_count = len(df)
        field_type = str(df[field_name].dtype)
        
        # 自然言語提案生成
        suggestion = self.generate_missing_value_suggestions(
            field_name, missing_count, total_count, field_type
        )
        
        # 処理オプション
        processing_options = {
            'fill_mean': {
                'label': '平均値で補完',
                'applicable': df[field_name].dtype in ['int64', 'float64'],
                'function': lambda: df[field_name].fillna(df[field_name].mean())
            },
            'fill_median': {
                'label': '中央値で補完',
                'applicable': df[field_name].dtype in ['int64', 'float64'],
                'function': lambda: df[field_name].fillna(df[field_name].median())
            },
            'fill_mode': {
                'label': '最頻値で補完',
                'applicable': True,
                'function': lambda: df[field_name].fillna(df[field_name].mode()[0] if not df[field_name].mode().empty else np.nan)
            },
            'interpolate': {
                'label': '線形補間',
                'applicable': df[field_name].dtype in ['int64', 'float64'],
                'function': lambda: df[field_name].interpolate(method='linear')
            },
            'forward_fill': {
                'label': '前方補完',
                'applicable': True,
                'function': lambda: df[field_name].fillna(method='ffill')
            },
            'backward_fill': {
                'label': '後方補完',
                'applicable': True,
                'function': lambda: df[field_name].fillna(method='bfill')
            },
            'drop_rows': {
                'label': '欠損値を含む行を削除',
                'applicable': True,
                'function': lambda: df.dropna(subset=[field_name])
            }
        }
        
        # 適用可能なオプションのみフィルタリング
        available_options = {k: v for k, v in processing_options.items() if v['applicable']}
        
        return {
            'field_name': field_name,
            'missing_count': missing_count,
            'total_count': total_count,
            'missing_rate': round((missing_count / total_count) * 100, 2),
            'field_type': field_type,
            'suggestion': suggestion,
            'options': available_options
        }

class AutoVizIntegration:
    """AutoViz統合クラス"""
    
    def __init__(self):
        self.autoviz_available = AUTOVIZ_AVAILABLE
        
    def implement_autoviz_eda(self, df: pd.DataFrame, filename: str = 'uploaded_data.csv') -> Optional[Dict]:
        """AutoVizによる自動EDA実装"""
        if not self.autoviz_available:
            return self.create_fallback_eda(df)
        
        # 一時ファイル保存
        temp_file = f'temp_{filename}'
        df.to_csv(temp_file, index=False)
        
        try:
            # AutoViz実行
            AV = AutoViz_Class()
            autoviz_plots = AV.AutoViz(
                temp_file,
                sep=',',
                depVar='',  # 目的変数指定（空の場合は自動推定）
                dfte=df,
                header=0,
                verbose=1,
                lowess=False,
                chart_format='html',
                max_rows_analyzed=min(len(df), 10000),  # 最大10000行
                max_cols_analyzed=min(len(df.columns), 30)  # 最大30列
            )
            
            return {'status': 'success', 'plots': autoviz_plots}
            
        except Exception as e:
            logger.error(f"AutoViz実行エラー: {e}")
            return self.create_fallback_eda(df)
        finally:
            # 一時ファイル削除
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    def create_fallback_eda(self, df: pd.DataFrame) -> Dict:
        """AutoVizが使用できない場合のフォールバックEDA"""
        plots = []
        
        # 数値列の分布
        numeric_cols = df.select_dtypes(include=[np.number]).columns[:10]
        for col in numeric_cols:
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=df[col].dropna(), name=col))
            fig.update_layout(title=f'Distribution of {col}', xaxis_title=col, yaxis_title='Count')
            plots.append({'type': 'histogram', 'column': col, 'plot': json.loads(fig.to_json())})
        
        # 相関行列
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns.tolist(),
                y=corr_matrix.columns.tolist(),
                colorscale='RdBu',
                zmid=0
            ))
            fig.update_layout(title='Correlation Matrix')
            plots.append({'type': 'correlation', 'plot': json.loads(fig.to_json())})
        
        return {'status': 'fallback', 'plots': plots}

class ComprehensiveFeatureEngineering:
    """統合特徴量エンジニアリングクラス"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.quality_analyzer = AdvancedDataQualityAnalyzer(df)
        self.missing_handler = LangChainMissingValueHandler()
        self.autoviz = AutoVizIntegration()
        self.results = {}
        
    def create_comprehensive_dashboard(self) -> Dict:
        """統合ダッシュボード作成"""
        
        # 1. フィールド分析
        field_info, type_mismatches = self.quality_analyzer.display_field_analysis()
        type_inconsistencies = self.quality_analyzer.identify_type_inconsistencies()
        
        # 2. 外れ値検出
        outlier_results, outlier_plots = self.quality_analyzer.detect_outliers_3sigma()
        
        # 3. 欠損値分析
        missing_value_analysis = {}
        for col in self.df.columns:
            if self.df[col].isnull().sum() > 0:
                missing_value_analysis[col] = self.missing_handler.interactive_missing_value_handler(
                    self.df, col
                )
        
        # 4. AutoViz EDA（オプショナル）
        autoviz_results = self.autoviz.implement_autoviz_eda(self.df)
        
        # 5. 欠損値ヒートマップ
        missing_heatmap = self.create_missing_value_heatmap()
        
        # 統合結果
        comprehensive_results = {
            'field_analysis': {
                'summary': field_info.to_dict('records'),
                'type_mismatches': type_mismatches,
                'type_inconsistencies': type_inconsistencies.to_dict('records') if not type_inconsistencies.empty else []
            },
            'outlier_analysis': {
                'results': outlier_results,
                'plots': outlier_plots
            },
            'missing_value_analysis': missing_value_analysis,
            'missing_heatmap': missing_heatmap,
            'autoviz_results': autoviz_results,
            'summary_stats': self.create_summary_statistics()
        }
        
        return comprehensive_results
    
    def create_missing_value_heatmap(self) -> Dict:
        """欠損値のヒートマップ作成"""
        # 欠損値マトリックス作成
        missing_matrix = self.df.isnull().astype(int)
        
        fig = go.Figure(data=go.Heatmap(
            z=missing_matrix.T.values,
            x=list(range(len(self.df))),
            y=missing_matrix.columns.tolist(),
            colorscale='RdYlBu_r',
            showscale=True,
            hovertemplate='Row: %{x}<br>Column: %{y}<br>Missing: %{z}<extra></extra>'
        ))
        
        fig.update_layout(
            title='欠損値ヒートマップ（赤: 欠損、青: 存在）',
            xaxis_title='行インデックス',
            yaxis_title='カラム名',
            height=max(400, len(missing_matrix.columns) * 20),
            width=800
        )
        
        return json.loads(fig.to_json())
    
    def create_summary_statistics(self) -> Dict:
        """サマリー統計作成"""
        return {
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns),
            'numeric_columns': len(self.df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': len(self.df.select_dtypes(include=['object', 'category']).columns),
            'total_missing_values': int(self.df.isnull().sum().sum()),
            'missing_percentage': round(self.df.isnull().sum().sum() / (len(self.df) * len(self.df.columns)) * 100, 2),
            'memory_usage': f"{self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
        }
    
    def apply_missing_value_treatment(self, field_name: str, method: str) -> pd.Series:
        """欠損値処理の適用"""
        handler_info = self.missing_handler.interactive_missing_value_handler(self.df, field_name)
        
        if method in handler_info['options']:
            return handler_info['options'][method]['function']()
        else:
            raise ValueError(f"Method {method} not available for field {field_name}")
    
    def generate_feature_engineering_report(self) -> str:
        """特徴量エンジニアリングレポート生成"""
        dashboard = self.create_comprehensive_dashboard()
        
        report = f"""
        ========================================
        特徴量エンジニアリング総合レポート
        ========================================
        
        作成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        1. データ概要
        ----------------------------------------
        • 総行数: {dashboard['summary_stats']['total_rows']}
        • 総列数: {dashboard['summary_stats']['total_columns']}
        • 数値列: {dashboard['summary_stats']['numeric_columns']}
        • カテゴリ列: {dashboard['summary_stats']['categorical_columns']}
        • メモリ使用量: {dashboard['summary_stats']['memory_usage']}
        
        2. データ品質
        ----------------------------------------
        • 総欠損値数: {dashboard['summary_stats']['total_missing_values']}
        • 欠損率: {dashboard['summary_stats']['missing_percentage']}%
        • 型不整合フィールド数: {len(dashboard['field_analysis']['type_mismatches'])}
        
        3. 外れ値分析
        ----------------------------------------
        """
        
        for col, outlier_info in dashboard['outlier_analysis']['results'].items():
            report += f"""
        • {col}:
          - 外れ値数: {outlier_info['outlier_count']}
          - 外れ値率: {outlier_info['outlier_percentage']}%
          - 3σ範囲: [{outlier_info['lower_bound']:.2f}, {outlier_info['upper_bound']:.2f}]
            """
        
        report += """
        
        4. 推奨アクション
        ----------------------------------------
        """
        
        # 欠損値処理の推奨
        for col, missing_info in dashboard['missing_value_analysis'].items():
            if missing_info['missing_rate'] > 50:
                report += f"\n• {col}: 欠損率が高いため、列削除を検討"
            elif missing_info['missing_rate'] > 20:
                report += f"\n• {col}: 予測モデルによる補完を推奨"
            elif missing_info['missing_rate'] > 0:
                report += f"\n• {col}: 代表値による補完を推奨"
        
        return report

def create_feature_engineering_interface():
    """ユーザーインターフェース設計"""
    interface_sections = {
        '1. データ品質チェック': {
            'フィールド情報表': {
                'type': 'table',
                'description': 'Tabular形式でフィールド詳細情報を表示',
                'columns': ['フィールド名', 'データ型', '実際の型', '非null件数', 
                          '欠損値件数', '欠損率(%)', 'ユニーク値数', 'サンプル値']
            },
            'データ型不整合': {
                'type': 'list',
                'description': '型不整合の詳細リスト'
            },
            '欠損値マップ': {
                'type': 'heatmap',
                'description': '欠損値の分布をヒートマップで可視化'
            }
        },
        
        '2. 欠損値処理': {
            'LangChain提案': {
                'type': 'text',
                'description': '自然言語での状況説明と推奨処理'
            },
            '処理方法選択': {
                'type': 'radio',
                'options': ['平均値補完', '中央値補完', '最頻値補完', 
                          '線形補間', '前方補完', '後方補完', '行削除']
            },
            '詳細オプション': {
                'type': 'collapsible',
                'description': '高度な処理オプション'
            }
        },
        
        '3. 外れ値分析': {
            '3σ検出結果': {
                'type': 'summary',
                'description': '統計サマリーと外れ値数'
            },
            'ヒストグラム': {
                'type': 'plot',
                'description': '3σ閾値ライン付きグラフ'
            },
            '外れ値リスト': {
                'type': 'table',
                'description': '該当行の詳細表示'
            }
        },
        
        '4. AutoViz EDA': {
            '自動可視化': {
                'type': 'auto',
                'description': 'AutoVizによる自動EDA実行'
            },
            '統合ダッシュボード': {
                'type': 'dashboard',
                'description': '全分析結果の統合表示'
            }
        }
    }
    
    return interface_sections