#!/usr/bin/env python3
"""
Q-Storm Platform v4.2.1 - Agent Integration Version
既存のAPIを維持しながらエージェントアーキテクチャに移行
"""

import os
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify, session
from werkzeug.utils import secure_filename
import logging
from datetime import timedelta
import uuid
import traceback

# エージェントをインポート
from agents import DataAnalysisAgent

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

# エージェントインスタンス（シングルトン）
data_agent = DataAnalysisAgent()

# ディレクトリ作成
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- ヘルパー関数 ---
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
    """JSONシリアライズ可能な形式に変換"""
    if isinstance(obj, np.integer): 
        return int(obj)
    elif isinstance(obj, np.floating): 
        return float(obj) if np.isfinite(obj) else None
    elif isinstance(obj, np.ndarray): 
        return [safe_json_convert(i) for i in obj.tolist()]
    elif isinstance(obj, pd.Timestamp): 
        return obj.isoformat()
    elif isinstance(obj, dict): 
        return {k: safe_json_convert(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)): 
        return [safe_json_convert(i) for i in obj]
    return obj

# --- ルート定義 ---
@app.route('/')
def index():
    """メインページ"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """ファイルアップロード処理"""
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'ファイルが選択されていません'}), 400
    
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'status': 'error', 'message': '無効なファイル形式です'}), 400

    try:
        filename = secure_filename(file.filename)
        df = pd.read_excel(file) if filename.endswith(('.xls', '.xlsx')) else pd.read_csv(file, encoding='utf-8-sig')
        
        # 列名の正規化
        df.columns = [str(c).strip().replace('  ', ' ') for c in df.columns]
        
        logger.info(f"Uploaded file columns: {list(df.columns)}")
        logger.info(f"Data shape: {df.shape}")
        
        # WOMEN'S_SCARF & STOLESの重複問題対応
        if 'WOMEN\'S_SCARF & STOLES' in df.columns:
            columns_list = list(df.columns)
            scarf_count = columns_list.count('WOMEN\'S_SCARF & STOLES')
            if scarf_count > 1:
                first_idx = columns_list.index('WOMEN\'S_SCARF & STOLES')
                columns_list[first_idx] = 'WOMEN\'S_SCARF & STOLES'
                for i in range(first_idx + 1, len(columns_list)):
                    if columns_list[i] == 'WOMEN\'S_SCARF & STOLES':
                        columns_list[i] = 'WOMEN\'S_SCARF & STOLESR'
                        break
                df.columns = columns_list

        session_id = get_session_id()
        session_data_store[session_id] = {
            'df': df, 
            'filename': filename, 
            'qc_step': 1, 
            'history': []
        }
        return jsonify({'status': 'success', 'message': 'ファイルアップロード成功。ステップ1: テーマ選定を開始します。'})
    except Exception as e:
        logger.error(f"Upload error: {e}\n{traceback.format_exc()}")
        return jsonify({'status': 'error', 'message': f'ファイル処理エラー: {e}'}), 500

@app.route('/api/get_available_shops', methods=['GET'])
def get_available_shops():
    """利用可能な店舗リストを返す"""
    session_data = get_session_data()
    if not session_data or 'df' not in session_data:
        return jsonify({'status': 'error', 'message': 'データがありません'}), 404
    
    try:
        df = session_data['df']
        shops = []
        if 'shop' in df.columns:
            shops = df['shop'].dropna().unique().tolist()
        return jsonify({'status': 'success', 'shops': shops})
    except Exception as e:
        logger.error(f"Get shops error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/get_feature_analysis', methods=['GET'])
def get_feature_analysis():
    """
    特徴量分析の取得（エージェント使用）
    """
    session_data = get_session_data()
    if not session_data or 'df' not in session_data:
        return jsonify({'status': 'error', 'message': 'データがありません'}), 404

    try:
        df = session_data['df']
        
        # エージェントを使用して分析
        result = data_agent.analyze_features(df)
        
        if result['status'] == 'success':
            # セッションデータを更新
            update_session_data({
                'columns_info': result['columns_info']
            })
            
            return jsonify({
                'status': 'success',
                'feature_analysis': safe_json_convert(result['feature_analysis']),
                'columns_info': result['columns_info']
            })
        else:
            return jsonify(result), 500
            
    except Exception as e:
        logger.error(f"Feature analysis error: {e}\n{traceback.format_exc()}")
        return jsonify({'status': 'error', 'message': f'特徴量分析エラー: {e}'}), 500

@app.route('/api/propose_improvement', methods=['POST'])
def propose_improvement():
    """
    改善提案の取得（エージェント使用）
    """
    session_data = get_session_data()
    if not session_data or 'df' not in session_data:
        return jsonify({'status': 'error', 'message': 'データがありません'}), 404
    
    try:
        field_name = request.json.get('field_name')
        df = session_data['df']
        
        # エージェントを使用して提案を取得
        result = data_agent.propose_improvement(df, field_name)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Proposal error: {e}\n{traceback.format_exc()}")
        return jsonify({'status': 'error', 'message': f'改善提案エラー: {e}'}), 500

@app.route('/api/apply_improvement', methods=['POST'])
def apply_improvement():
    """
    改善の適用（エージェント使用）
    """
    session_data = get_session_data()
    if not session_data or 'df' not in session_data:
        return jsonify({'status': 'error', 'message': 'データがありません'}), 404
    
    try:
        field_name = request.json.get('field_name')
        method = request.json.get('method')
        df = session_data['df']
        
        # エージェントを使用して改善を適用
        result = data_agent.apply_improvement(df, field_name, method)
        
        if result['status'] == 'success' and 'updated_df' in result:
            # データフレームを更新
            update_session_data({'df': result['updated_df']})
            return jsonify({
                'status': result['status'],
                'message': result['message']
            })
        else:
            return jsonify(result)
            
    except Exception as e:
        logger.error(f"Apply improvement error: {e}\n{traceback.format_exc()}")
        return jsonify({'status': 'error', 'message': f'改善適用エラー: {e}'}), 500

@app.route('/api/generate_graph', methods=['POST'])
def generate_graph():
    """
    グラフ生成（エージェント使用）
    """
    session_data = get_session_data()
    if not session_data or 'df' not in session_data:
        return jsonify({'status': 'error', 'message': 'データがありません'}), 404

    try:
        params = request.json
        graph_type = params.get('graph_type')
        df = session_data['df']
        
        logger.info(f"Graph generation request: {params}")
        
        # エージェントを使用してグラフを生成
        if graph_type == 'histogram':
            result = data_agent.generate_histogram(df, params)
        elif graph_type == 'line':
            result = data_agent.generate_line_chart(df, params)
        else:
            return jsonify({'status': 'error', 'message': '未対応のグラフタイプです'}), 400
        
        if result['status'] == 'success':
            return jsonify(result)
        else:
            return jsonify(result), 400 if result['status'] == 'error' else 200

    except Exception as e:
        logger.error(f"Graph generation error: {e}\n{traceback.format_exc()}")
        return jsonify({'status': 'error', 'message': f'グラフ生成エラー: {str(e)}'}), 500

@app.route('/api/agent_status', methods=['GET'])
def get_agent_status():
    """
    エージェントの状態を取得（新規API）
    """
    return jsonify({
        'status': 'success',
        'agent_info': {
            'name': data_agent.name,
            'version': '1.0.0',
            'capabilities': ['feature_analysis', 'improvement_proposal', 'histogram', 'line_chart']
        }
    })

if __name__ == '__main__':
    logger.info("Starting Q-Storm Platform with Agent Integration...")
    logger.info("Agent system initialized successfully")
    app.run(debug=True, host='0.0.0.0', port=5001)