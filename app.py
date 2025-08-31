#!/usr/bin/env python3
"""
Q-Storm Platform v4.2.0 - QC Story Navigator Edition
- ヒストグラム作成時のデータ型バグを修正
- ヒストグラムにカテゴリ別のグループ化機能を追加
"""

import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from flask import Flask, render_template, request, jsonify, session, send_from_directory, redirect, url_for
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import logging
from datetime import datetime, timedelta
import uuid
import traceback
import openpyxl
import secrets

# 既存の高度な特徴量エンジニアリングモジュールをインポート
from advanced_feature_engineering import (
    AdvancedDataQualityAnalyzer,
    LangChainMissingValueHandler,
    ComprehensiveFeatureEngineering
)

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flaskアプリケーション初期化
app = Flask(__name__)
app.config['SECRET_KEY'] = 'q-storm-platform-v4-secret-key-2024'
app.config['SESSION_TYPE'] = 'filesystem'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=2)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB
app.config['UPLOAD_FOLDER'] = 'uploads'

# アップロード設定
ALLOWED_EXTENSIONS = {'csv', 'xls', 'xlsx', 'xlsm'}

# セッションデータストア
session_data_store = {}

# ユーザーデータストア（本番環境ではデータベースを使用すべき）
users_store = {}

# アクティブセッション管理
active_sessions = {}

# ディレクトリ作成
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# デフォルトユーザーの作成
users_store['admin'] = {
    'password_hash': generate_password_hash('admin123'),
    'role': 'admin',
    'created_at': datetime.now().isoformat()
}
users_store['user'] = {
    'password_hash': generate_password_hash('user123'),
    'role': 'user',
    'created_at': datetime.now().isoformat()
}

# --- データセットの日本語・英語フィールド名対応 ---
FIELD_NAME_MAPPING = {
    'shop': '店舗名', 'shop_code': '店舗コード', 'Date': '営業日付', 'Total_Sales': '店舗別売上高',
    'gross_profit': '売上総利益（粗利）', 'discount': '値引・割引（月額）', 'purchasing': '仕入高',
    'rent': '家賃', 'personnel_expenses': '人件費', 'depreciation': '減価償却費',
    'sales_promotion': '販売促進費', 'head_office_expenses': '本部費用配賦', 'operating_cost': '営業経費',
    'Operating_profit': '営業利益', 'Mens_JACKETS&OUTER2': 'メンズ ジャケット・アウター 売上高',
    'Mens_KNIT': 'メンズ ニット 売上高', 'Mens_PANTS': 'メンズ パンツ 売上高',
    'WOMEN\'S_JACKETS2': 'レディース ジャケット 売上高', 'WOMEN\'S_TOPS': 'レディース トップス 売上高',
    'WOMEN\'S_ONEPIECE': 'レディース ワンピース 売上高', 'WOMEN\'S_bottoms': 'レディース ボトムス 売上高',
    'WOMEN\'S_SCARF & STOLES': 'レディース スカーフ・ストール 売上高', 'Inventory': '在庫金額',
    'Months_of_inventory': '在庫月数', 'BEP': '損益分岐点（BEP）', 'Average_Temperature': '平均気温',
    'Number_of_guests': '来客数', 'Price_per_customer': '客単価',
    'Mens_JACKETS&OUTER2R': 'メンズ ジャケット・アウター 売上構成比', 'Mens_KNITR': 'メンズ ニット 売上構成比',
    'Mens_PANTSR': 'メンズ パンツ 売上構成比', 'WOMEN\'S_JACKETSR': 'レディース ジャケット 売上構成比',
    'WOMEN\'S_TOPSR': 'レディース トップス 売上構成比', 'WOMEN\'S_ONEPIECER': 'レディース ワンピース 売上構成比',
    'WOMEN\'S_bottomsR': 'レディース ボトムス 売上構成比',
    'WOMEN\'S_SCARF & STOLESR': 'レディース スカーフ・ストール 売上構成比', 'judge': '判定（評価）'
}
REVERSE_FIELD_NAME_MAPPING = {v: k for k, v in FIELD_NAME_MAPPING.items()}

# 定義済みメトリクスリスト（参考コードから移植）
HIST_METRICS = [
    "Total_Sales", "gross_profit", "discount", "purchasing", "rent",
    "personnel_expenses", "depreciation", "sales_promotion",
    "head_office_expenses", "operating_cost", "Operating_profit"
]

LINE_METRICS = [
    "Mens_JACKETS&OUTER2", "Mens_KNIT", "Mens_PANTS",
    "WOMEN'S_JACKETS2", "WOMEN'S_TOPS", "WOMEN'S_ONEPIECE",
    "WOMEN'S_bottoms", "WOMEN'S_SCARF & STOLES",
    "Inventory", "Months_of_inventory", "BEP",
    "Average_Temperature", "Number_of_guests", "Price_per_customer",
    "Mens_JACKETS&OUTER2R", "Mens_KNITR", "Mens_PANTSR",
    "WOMEN'S_JACKETSR", "WOMEN'S_TOPSR", "WOMEN'S_ONEPIECER",
    "WOMEN'S_bottomsR"
]

# --- ヘルパー関数 ---
def normalize_column_name(col_name):
    """列名を正規化（空白削除、全角→半角変換など）"""
    import unicodedata
    # 前後の空白削除
    col_name = str(col_name).strip()
    # 連続する空白を単一に
    col_name = ' '.join(col_name.split())
    # 全角英数字を半角に変換
    col_name = unicodedata.normalize('NFKC', col_name)
    return col_name

def find_similar_columns(df, target_col):
    """類似する列名を探す"""
    similar = []
    target_lower = target_col.lower().replace(' ', '').replace('_', '')
    for col in df.columns:
        col_lower = col.lower().replace(' ', '').replace('_', '')
        if target_lower in col_lower or col_lower in target_lower:
            similar.append(col)
    return similar

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_session_id():
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return session['session_id']

def get_session_data():
    session_id = get_session_id()
    return session_data_store.get(session_id)

def update_session_data(data):
    session_id = get_session_id()
    if session_id in session_data_store:
        session_data_store[session_id].update(data)
    else:
        session_data_store[session_id] = data

def safe_json_convert(obj):
    if isinstance(obj, np.integer): return int(obj)
    elif isinstance(obj, np.floating): return float(obj) if np.isfinite(obj) else None
    elif isinstance(obj, np.ndarray): return [safe_json_convert(i) for i in obj.tolist()]
    elif isinstance(obj, pd.Timestamp): return obj.isoformat()
    elif isinstance(obj, (datetime, timedelta)): return obj.isoformat()
    elif isinstance(obj, dict): return {k: safe_json_convert(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)): return [safe_json_convert(i) for i in obj]
    return obj

# --- 認証デコレータ ---
def login_required(f):
    """ログイン必須デコレータ"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({'status': 'error', 'message': 'ログインが必要です'}), 401
        # アクティブセッションの最終活動時刻を更新
        session_id = session.get('session_id')
        if session_id in active_sessions:
            active_sessions[session_id]['last_activity'] = datetime.now()
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    """管理者権限必須デコレータ"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({'status': 'error', 'message': 'ログインが必要です'}), 401
        if session.get('role') != 'admin':
            return jsonify({'status': 'error', 'message': '管理者権限が必要です'}), 403
        return f(*args, **kwargs)
    return decorated_function

# --- 認証関連ルート ---
@app.route('/login', methods=['POST'])
def login():
    """ユーザーログイン"""
    try:
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        
        if not username or not password:
            return jsonify({'status': 'error', 'message': 'ユーザー名とパスワードが必要です'}), 400
        
        # ユーザー認証
        user = users_store.get(username)
        if not user or not check_password_hash(user['password_hash'], password):
            logger.warning(f"Failed login attempt for user: {username}")
            return jsonify({'status': 'error', 'message': '認証に失敗しました'}), 401
        
        # セッション作成
        session_id = secrets.token_urlsafe(32)
        session['user_id'] = username
        session['role'] = user['role']
        session['session_id'] = session_id
        session['login_time'] = datetime.now().isoformat()
        session.permanent = True
        
        # アクティブセッションに追加
        active_sessions[session_id] = {
            'user_id': username,
            'login_time': datetime.now(),
            'last_activity': datetime.now()
        }
        
        logger.info(f"User logged in: {username}")
        
        return jsonify({
            'status': 'success',
            'message': 'ログイン成功',
            'user': {
                'username': username,
                'role': user['role']
            }
        })
        
    except Exception as e:
        logger.error(f"Login error: {e}\n{traceback.format_exc()}")
        return jsonify({'status': 'error', 'message': 'ログインエラー'}), 500

@app.route('/logout', methods=['POST'])
@login_required
def logout():
    """ユーザーログアウト"""
    try:
        session_id = session.get('session_id')
        user_id = session.get('user_id')
        
        # セッションから削除
        if session_id in active_sessions:
            del active_sessions[session_id]
        
        # セッションクリア
        session.clear()
        
        logger.info(f"User logged out: {user_id}")
        
        return jsonify({
            'status': 'success',
            'message': 'ログアウトしました'
        })
        
    except Exception as e:
        logger.error(f"Logout error: {e}")
        return jsonify({'status': 'error', 'message': 'ログアウトエラー'}), 500

@app.route('/register', methods=['POST'])
@admin_required
def register():
    """新規ユーザー登録（管理者のみ）"""
    try:
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        role = data.get('role', 'user')
        
        if not username or not password:
            return jsonify({'status': 'error', 'message': 'ユーザー名とパスワードが必要です'}), 400
        
        if username in users_store:
            return jsonify({'status': 'error', 'message': 'ユーザー名は既に使用されています'}), 409
        
        # ユーザー登録
        users_store[username] = {
            'password_hash': generate_password_hash(password),
            'role': role,
            'created_at': datetime.now().isoformat(),
            'created_by': session.get('user_id')
        }
        
        logger.info(f"New user registered: {username} by {session.get('user_id')}")
        
        return jsonify({
            'status': 'success',
            'message': 'ユーザー登録完了',
            'user': {
                'username': username,
                'role': role
            }
        })
        
    except Exception as e:
        logger.error(f"Registration error: {e}\n{traceback.format_exc()}")
        return jsonify({'status': 'error', 'message': '登録エラー'}), 500

@app.route('/check_auth', methods=['GET'])
def check_auth():
    """認証状態チェック"""
    if 'user_id' in session:
        return jsonify({
            'status': 'authenticated',
            'user': {
                'username': session['user_id'],
                'role': session.get('role', 'user')
            }
        })
    return jsonify({'status': 'not_authenticated'})

# --- ルート定義 ---
@app.route('/')
@login_required
def index():
    """メインページ（ログイン必須）"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
@login_required
def upload_file():
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'ファイルが選択されていません'}), 400
    
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'status': 'error', 'message': '無効なファイル形式です'}), 400

    try:
        filename = secure_filename(file.filename)
        df = pd.read_excel(file) if filename.endswith(('.xls', '.xlsx')) else pd.read_csv(file, encoding='utf-8-sig')
        
        # 列名の正規化：前後の空白削除、連続空白を単一に
        df.columns = [str(c).strip().replace('  ', ' ') for c in df.columns]
        
        # デバッグ用：列名とデータ型を出力
        logger.info(f"Uploaded file columns: {list(df.columns)}")
        logger.info(f"Data shape: {df.shape}")
        
        # 特定の列名の修正（WOMEN'S_SCARF & STOLESの重複問題対応）
        if 'WOMEN\'S_SCARF & STOLES' in df.columns:
            # 構成比の列名を明確に区別
            columns_list = list(df.columns)
            scarf_count = columns_list.count('WOMEN\'S_SCARF & STOLES')
            if scarf_count > 1:
                # 2つ目を構成比として扱う
                first_idx = columns_list.index('WOMEN\'S_SCARF & STOLES')
                columns_list[first_idx] = 'WOMEN\'S_SCARF & STOLES'
                for i in range(first_idx + 1, len(columns_list)):
                    if columns_list[i] == 'WOMEN\'S_SCARF & STOLES':
                        columns_list[i] = 'WOMEN\'S_SCARF & STOLESR'
                        break
                df.columns = columns_list

        session_id = get_session_id()
        session_data_store[session_id] = {
            'df': df, 'filename': filename, 'qc_step': 1, 'history': []
        }
        return jsonify({'status': 'success', 'message': 'ファイルアップロード成功。ステップ1: テーマ選定を開始します。'})
    except Exception as e:
        logger.error(f"Upload error: {e}\n{traceback.format_exc()}")
        return jsonify({'status': 'error', 'message': f'ファイル処理エラー: {e}'}), 500

@app.route('/api/get_available_shops', methods=['GET'])
@login_required
def get_available_shops():
    """利用可能な店舗リストを返す"""
    session_data = get_session_data()
    if not session_data or 'df' not in session_data:
        return jsonify({'status': 'error', 'message': 'データがありません'}), 404
    
    try:
        df = session_data['df']
        shops = []
        if 'shop' in df.columns:
            # データベースの生の店舗名を返す（日本語変換を削除）
            shops = df['shop'].dropna().unique().tolist()
        return jsonify({'status': 'success', 'shops': shops})
    except Exception as e:
        logger.error(f"Get shops error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/get_feature_analysis', methods=['GET'])
@login_required
def get_feature_analysis():
    session_data = get_session_data()
    if not session_data or 'df' not in session_data:
        return jsonify({'status': 'error', 'message': 'データがありません'}), 404

    try:
        df = session_data['df']
        analyzer = AdvancedDataQualityAnalyzer(df)
        field_info_df, _ = analyzer.display_field_analysis()
        
        results = []
        numeric_cols = []
        categorical_cols = []
        date_cols = []
        
        for record in field_info_df.to_dict('records'):
            col_name = record['フィールド名']
            jp_name = FIELD_NAME_MAPPING.get(col_name, col_name)
            record['日本語名'] = jp_name
            
            # データ型カテゴリを判定
            dtype_category = get_data_type_category(df[col_name])
            record['データ型カテゴリ'] = dtype_category
            
            # 列を分類
            if col_name in ['shop', 'shop_code', 'judge']:
                # 特定のカテゴリカル項目
                categorical_cols.append(jp_name)
            elif dtype_category == '日付/時間':
                date_cols.append(jp_name)
            else:
                # その他はすべて数値として扱う
                numeric_cols.append(jp_name)
            
            results.append(record)
        
        # ログ出力で確認
        logger.info(f"Numeric columns: {len(numeric_cols)} items")
        logger.info(f"Categorical columns: {len(categorical_cols)} items")
        logger.info(f"Date columns: {len(date_cols)} items")
        
        update_session_data({
            'columns_info': {
                'numeric': numeric_cols, 
                'categorical': categorical_cols, 
                'date': date_cols
            }
        })

        return jsonify({
            'status': 'success',
            'feature_analysis': safe_json_convert(results),
            'columns_info': {
                'numeric': numeric_cols, 
                'categorical': categorical_cols, 
                'date': date_cols
            }
        })
    except Exception as e:
        logger.error(f"Feature analysis error: {e}\n{traceback.format_exc()}")
        return jsonify({'status': 'error', 'message': f'特徴量分析エラー: {e}'}), 500

def get_data_type_category(series):
    dtype = str(series.dtype)
    col_name = series.name
    
    # 特定のカテゴリカルフィールドを判定
    categorical_fields = ['shop', 'shop_code', 'judge']
    if col_name in categorical_fields:
        return 'カテゴリ'
    
    if 'datetime' in dtype: 
        return '日付/時間'
    elif 'int' in dtype or 'float' in dtype:
        # 数値型として分類
        return '数値'
    elif 'object' in dtype or 'category' in dtype:
        try:
            pd.to_datetime(series, errors='raise')
            return '日付/時間'
        except:
            # ユニーク値が少ない場合はカテゴリ
            if series.nunique() < 10:
                return 'カテゴリ'
            else:
                return 'テキスト'
    return '不明'

@app.route('/api/propose_improvement', methods=['POST'])
@login_required
def propose_improvement():
    session_data = get_session_data()
    if not session_data or 'df' not in session_data:
        return jsonify({'status': 'error', 'message': 'データがありません'}), 404
    try:
        field_name = request.json.get('field_name')
        df = session_data['df']
        handler = LangChainMissingValueHandler()
        if df[field_name].isnull().sum() > 0:
            proposals = handler.interactive_missing_value_handler(df, field_name)
            return jsonify({'status': 'success', 'field_name': field_name, 'suggestion': proposals['suggestion'], 'options': proposals['options']})
        else:
            return jsonify({'status': 'info', 'message': 'このフィールドに明確な欠損値の問題は見つかりませんでした。'})
    except Exception as e:
        logger.error(f"Proposal error: {e}\n{traceback.format_exc()}")
        return jsonify({'status': 'error', 'message': f'改善提案エラー: {e}'}), 500

@app.route('/api/apply_improvement', methods=['POST'])
@login_required
def apply_improvement():
    session_data = get_session_data()
    if not session_data or 'df' not in session_data:
        return jsonify({'status': 'error', 'message': 'データがありません'}), 404
    try:
        field_name = request.json.get('field_name')
        method = request.json.get('method')
        df = session_data['df']
        analyzer = ComprehensiveFeatureEngineering(df)
        treated_series = analyzer.apply_missing_value_treatment(field_name, method)
        df[field_name] = treated_series
        update_session_data({'df': df})
        return jsonify({'status': 'success', 'message': f'フィールド「{FIELD_NAME_MAPPING.get(field_name, field_name)}」に「{method}」を適用しました。'})
    except Exception as e:
        logger.error(f"Apply improvement error: {e}\n{traceback.format_exc()}")
        return jsonify({'status': 'error', 'message': f'改善適用エラー: {e}'}), 500

@app.route('/api/generate_graph', methods=['POST'])
@login_required
def generate_graph():
    session_data = get_session_data()
    if not session_data or 'df' not in session_data:
        return jsonify({'status': 'error', 'message': 'データがありません'}), 404

    try:
        params = request.json
        graph_type = params.get('graph_type')
        df = session_data['df']
        fig = None
        
        logger.info(f"Graph generation request: {params}")

        if graph_type == 'histogram':
            target_col_jp = params.get('target_col')
            group_by_col_jp = params.get('group_by_col')
            filter_shop = params.get('filter_shop')  # 店舗フィルタ（直接処理）
            
            # 日本語名から英語フィールド名に変換（target_colのみ）
            target_col = REVERSE_FIELD_NAME_MAPPING.get(target_col_jp, target_col_jp)
            
            # デバッグ情報
            logger.info(f"=== Histogram Debug Info ===")
            logger.info(f"Target column: {target_col_jp} -> {target_col}")
            logger.info(f"Filter shop: '{filter_shop}'")
            logger.info(f"Group by: {group_by_col_jp}")
            
            # データフレームのコピーを作成
            df_copy = df.copy()
            initial_rows = len(df_copy)
            logger.info(f"Initial data rows: {initial_rows}")
            
            # 利用可能な店舗名を最初に確認
            if 'shop' in df_copy.columns:
                available_shops = df_copy['shop'].dropna().unique().tolist()
                logger.info(f"Available shops in data: {available_shops[:10]}")  # 最初の10店舗を表示
                
                # 店舗名のデータ型と例を確認
                shop_sample = df_copy['shop'].dropna().head(5).tolist()
                logger.info(f"Shop column sample: {shop_sample}")
            
            # 店舗フィルタ適用（改善版）
            if filter_shop and str(filter_shop).strip() != '':
                filter_shop_clean = str(filter_shop).strip()
                logger.info(f"Applying shop filter: '{filter_shop_clean}'")
                
                if 'shop' in df_copy.columns:
                    # 店舗列の前後の空白を削除
                    df_copy['shop'] = df_copy['shop'].astype(str).str.strip()
                    
                    # まず完全一致を試す
                    mask_exact = df_copy['shop'] == filter_shop_clean
                    exact_match_count = mask_exact.sum()
                    logger.info(f"Exact match for '{filter_shop_clean}': {exact_match_count} rows")
                    
                    if exact_match_count > 0:
                        df_copy = df_copy[mask_exact]
                        logger.info(f"Using exact match results: {len(df_copy)} rows")
                    else:
                        # 完全一致がない場合は部分一致を試す
                        logger.info(f"No exact match. Trying partial match...")
                        mask_partial = df_copy['shop'].str.contains(filter_shop_clean, na=False, case=False)
                        partial_match_count = mask_partial.sum()
                        logger.info(f"Partial match for '{filter_shop_clean}': {partial_match_count} rows")
                        
                        if partial_match_count > 0:
                            df_copy = df_copy[mask_partial]
                            matched_shops = df_copy['shop'].unique().tolist()
                            logger.info(f"Matched shops: {matched_shops}")
                        else:
                            # 類似する店舗名を探す
                            similar_shops = [s for s in available_shops if filter_shop_clean.lower() in s.lower() or s.lower() in filter_shop_clean.lower()]
                            logger.info(f"No matches found. Similar shops: {similar_shops}")
                            df_copy = df_copy.iloc[0:0]  # 空のDataFrameにする
                else:
                    logger.warning("'shop' column not found in dataframe")
            
            logger.info(f"After filtering: {len(df_copy)} rows")
            
            # フィルタ後のデータが空の場合（利用可能な店舗名を表示）
            if len(df_copy) == 0:
                available_shops = df['shop'].dropna().unique().tolist() if 'shop' in df.columns else []
                logger.warning(f"No data after shop filter: '{filter_shop}'")
                
                # 利用可能な店舗を最大20個まで表示
                shops_to_display = available_shops[:20]
                if len(available_shops) > 20:
                    shops_display_text = ', '.join(shops_to_display) + f'... (他{len(available_shops)-20}店舗)'
                else:
                    shops_display_text = ', '.join(shops_to_display) if shops_to_display else '店舗データなし'
                
                error_message = (f"データが見つかりませんでした<br><br>"
                                f"<b>入力された店舗名:</b> '{filter_shop}'<br><br>"
                                f"<b>利用可能な店舗名({len(available_shops)}店舗):</b><br>"
                                f"<small>{shops_display_text}</small>")
                
                # 空のヒストグラムに詳細な注記を追加
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
            else:
                # target_colが存在するか確認（類似列名も探す）
                if target_col not in df_copy.columns:
                    # 類似列名を探す
                    similar_cols = [col for col in df_copy.columns if target_col.lower() in col.lower()]
                    logger.error(f"Column '{target_col}' not found. Similar columns: {similar_cols}")
                    
                    # エラーメッセージに候補を含める
                    error_msg = f'列 「{target_col}」 が見つかりません。'
                    if similar_cols:
                        error_msg += f' 類似列: {similar_cols}'
                    return jsonify({'status': 'error', 'message': error_msg}), 400
                
                # 数値データに変換
                original_count = df_copy[target_col].notna().sum()
                df_copy[target_col] = pd.to_numeric(df_copy[target_col], errors='coerce')
                numeric_count = df_copy[target_col].notna().sum()
                
                logger.info(f"Column '{target_col}': {original_count} non-null -> {numeric_count} numeric values")
                
                # NaNを除去
                df_plot = df_copy.dropna(subset=[target_col])
                
                if len(df_plot) == 0:
                    logger.warning(f"All values are NaN after numeric conversion for '{target_col}'")
                    # データなしの注記を表示
                    fig = go.Figure()
                    fig.add_annotation(
                        text=f"数値データがありません<br>列: {target_col_jp}<br>すべての値が数値に変換できませんでした",
                        xref="paper", yref="paper",
                        x=0.5, y=0.5,
                        showarrow=False,
                        font=dict(size=14)
                    )
                    fig.update_layout(
                        title=f"{target_col_jp}の分布（数値データなし）",
                        xaxis_title=target_col_jp,
                        yaxis_title="Count",
                        height=500
                    )
                else:
                    # 統計情報を計算
                    stats = {
                        'mean': df_plot[target_col].mean(),
                        'median': df_plot[target_col].median(),
                        'std': df_plot[target_col].std(),
                        'min': df_plot[target_col].min(),
                        'max': df_plot[target_col].max(),
                        'count': len(df_plot)
                    }
                    logger.info(f"Stats for '{target_col}': mean={stats['mean']:.2f}, count={stats['count']}")
                    
                    title = f'{target_col_jp}の分布'
                    if filter_shop:
                        title += f'（{filter_shop}）'
                    
                    # グループ化処理
                    if group_by_col_jp and group_by_col_jp != '':
                        group_by_col = REVERSE_FIELD_NAME_MAPPING.get(group_by_col_jp, group_by_col_jp)
                        if group_by_col in df_plot.columns:
                            fig = px.histogram(
                                df_plot, 
                                x=target_col, 
                                color=group_by_col,
                                title=f'{title}（{group_by_col_jp}別）',
                                labels={target_col: target_col_jp, group_by_col: group_by_col_jp},
                                nbins=30
                            )
                        else:
                            fig = px.histogram(
                                df_plot, 
                                x=target_col,
                                title=title,
                                labels={target_col: target_col_jp},
                                nbins=30
                            )
                    else:
                        # goを使用してより詳細な制御
                        fig = go.Figure()
                        fig.add_trace(go.Histogram(
                            x=df_plot[target_col],
                            nbinsx=30,
                            name=target_col_jp,
                            hovertemplate=(
                                f'<b>{target_col_jp}</b><br>' +
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
                            xaxis_title=target_col_jp,
                            yaxis_title="Count"
                        )
                    
                    fig.update_layout(bargap=0.05)

        elif graph_type == 'line':
            date_col_jp = params.get('date_col')
            value_col_jp = params.get('value_col')
            filter_shop_jp = params.get('filter_shop')  # 店舗フィルタ
            date_start = params.get('date_start')  # 開始日
            date_end = params.get('date_end')  # 終了日
            
            # 日本語名から英語フィールド名に変換
            date_col = REVERSE_FIELD_NAME_MAPPING.get(date_col_jp, date_col_jp)
            value_col = REVERSE_FIELD_NAME_MAPPING.get(value_col_jp, value_col_jp)
            
            logger.info(f"Line graph: date={date_col}, value={value_col}, shop={filter_shop_jp}, range={date_start}~{date_end}")
            
            # データフレームのコピーを作成
            df_copy = df.copy()
            
            # 店舗フィルタ適用（改善版）
            if filter_shop and str(filter_shop).strip() != '':
                filter_shop_clean = str(filter_shop).strip()
                if 'shop' in df_copy.columns:
                    df_copy['shop'] = df_copy['shop'].astype(str).str.strip()
                    # まず完全一致、次に部分一致
                    mask = df_copy['shop'] == filter_shop_clean
                    if mask.sum() == 0:
                        mask = df_copy['shop'].str.contains(filter_shop_clean, na=False, case=False)
                    df_copy = df_copy[mask]
                    logger.info(f"Line graph shop filter '{filter_shop_clean}': {len(df_copy)} rows")
            
            # 列の存在確認
            if date_col not in df_copy.columns:
                return jsonify({'status': 'error', 'message': f'日付列 {date_col} が見つかりません'}), 400
            if value_col not in df_copy.columns:
                return jsonify({'status': 'error', 'message': f'値列 {value_col} が見つかりません'}), 400
            
            # データ型の変換
            df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')
            df_copy[value_col] = pd.to_numeric(df_copy[value_col], errors='coerce')
            
            # NaNを除去
            df_copy = df_copy.dropna(subset=[date_col, value_col])
            
            # 日付範囲フィルタ
            if date_start:
                df_copy = df_copy[df_copy[date_col] >= pd.to_datetime(date_start)]
            if date_end:
                df_copy = df_copy[df_copy[date_col] <= pd.to_datetime(date_end)]
            
            if len(df_copy) == 0:
                return jsonify({'status': 'error', 'message': '指定された条件に該当するデータがありません'}), 400
            
            # 日付でソート
            df_copy = df_copy.sort_values(by=date_col)
            
            # タイトル作成
            title = f'{value_col_jp}の時系列推移'
            if filter_shop_jp:
                title += f'（{filter_shop_jp}）'
            elif 'shop' in df_copy.columns and df_copy['shop'].nunique() > 1:
                title += '（店舗別）'
            
            # 店舗別にグループ化するかチェック
            if 'shop' in df_copy.columns and df_copy['shop'].nunique() > 1:
                fig = px.line(
                    df_copy, 
                    x=date_col, 
                    y=value_col, 
                    color='shop',
                    title=title,
                    labels={date_col: date_col_jp, value_col: value_col_jp, 'shop': '店舗名'},
                    markers=True
                )
            else:
                fig = px.line(
                    df_copy, 
                    x=date_col, 
                    y=value_col,
                    title=title,
                    labels={date_col: date_col_jp, value_col: value_col_jp},
                    markers=True
                )
            
            # レンジスライダーを追加
            fig.update_xaxes(rangeslider_visible=True)

        else:
            return jsonify({'status': 'error', 'message': '未対応のグラフタイプです'}), 400

        # 共通のレイアウト設定
        fig.update_layout(
            template="plotly_dark",
            font=dict(family="sans-serif", size=12, color="white"),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0.1)',
            hovermode='x unified',
            height=500,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        graph_json = json.loads(fig.to_json())
        return jsonify({'status': 'success', 'graph_json': graph_json})

    except Exception as e:
        logger.error(f"Graph generation error: {e}\n{traceback.format_exc()}")
        return jsonify({'status': 'error', 'message': f'グラフ生成エラー: {str(e)}'}), 500

@app.route('/export_data', methods=['POST'])
@login_required
def export_data():
    """
    データエクスポート機能
    処理済みデータをCSVまたはExcel形式でエクスポート
    """
    try:
        session_id = request.json.get('session_id')
        export_format = request.json.get('format', 'csv')
        include_processed = request.json.get('include_processed', True)
        
        if not session_id or session_id not in session_data_store:
            return jsonify({'status': 'error', 'message': 'セッションが見つかりません'}), 404
        
        df = session_data_store[session_id]['data']
        
        # 処理済みデータを含めるかどうか
        if include_processed and 'processed_data' in session_data_store[session_id]:
            processed_df = session_data_store[session_id]['processed_data']
            if processed_df is not None:
                df = processed_df
        
        # ファイル名の生成
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if export_format == 'excel':
            filename = f'export_data_{timestamp}.xlsx'
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Excelファイルとして保存（複数シート対応）
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='データ', index=False)
                
                # データ統計情報を別シートに追加
                stats_df = df.describe(include='all')
                stats_df.to_excel(writer, sheet_name='統計情報')
                
                # データ型情報を別シートに追加
                dtype_df = pd.DataFrame({
                    'カラム名': df.columns,
                    'データ型': df.dtypes.astype(str),
                    '非NULL件数': df.count(),
                    'NULL件数': df.isnull().sum(),
                    'ユニーク値数': df.nunique()
                })
                dtype_df.to_excel(writer, sheet_name='データ型情報', index=False)
            
            logger.info(f"Excel file exported: {filename}")
            
        else:  # CSV形式
            filename = f'export_data_{timestamp}.csv'
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            df.to_csv(filepath, index=False, encoding='utf-8-sig')
            logger.info(f"CSV file exported: {filename}")
        
        # ダウンロード用のURLを生成
        download_url = f'/download/{filename}'
        
        return jsonify({
            'status': 'success',
            'message': f'{export_format.upper()}形式でエクスポート完了',
            'filename': filename,
            'download_url': download_url,
            'rows': len(df),
            'columns': len(df.columns)
        })
        
    except Exception as e:
        logger.error(f"Export error: {e}\n{traceback.format_exc()}")
        return jsonify({'status': 'error', 'message': f'エクスポートエラー: {str(e)}'}), 500

@app.route('/download/<filename>')
@login_required
def download_file(filename):
    """
    エクスポートしたファイルのダウンロード
    """
    try:
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)
    except Exception as e:
        logger.error(f"Download error: {e}")
        return jsonify({'status': 'error', 'message': 'ファイルが見つかりません'}), 404

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
