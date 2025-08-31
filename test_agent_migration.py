#!/usr/bin/env python3
"""
エージェント移行テストスクリプト
既存機能が正常に動作することを確認
"""

import requests
import json
import pandas as pd
import numpy as np
from io import BytesIO
import time

class TestAgentMigration:
    """エージェント移行のテスト"""
    
    def __init__(self, base_url="http://127.0.0.1:5001"):
        self.base_url = base_url
        self.session = requests.Session()
        self.test_data = None
        
    def create_test_data(self):
        """テスト用データを作成"""
        np.random.seed(42)
        
        # サンプルデータ作成
        dates = pd.date_range('2024-01-01', periods=100)
        shops = ['店舗A', '店舗B', '店舗C']
        
        data = {
            'Date': np.repeat(dates, 3),
            'shop': shops * 100,
            'shop_code': ['A001', 'B001', 'C001'] * 100,
            'Total_Sales': np.random.randint(10000, 100000, 300),
            'Operating_profit': np.random.randint(1000, 20000, 300),
            'Number_of_guests': np.random.randint(50, 500, 300),
            'Price_per_customer': np.random.randint(500, 3000, 300),
            'judge': np.random.choice(['良好', '普通', '要改善'], 300)
        }
        
        # 欠損値を意図的に追加
        df = pd.DataFrame(data)
        df.loc[10:20, 'Operating_profit'] = np.nan
        df.loc[50:55, 'Number_of_guests'] = np.nan
        
        # Excelファイルとして保存
        buffer = BytesIO()
        df.to_excel(buffer, index=False)
        buffer.seek(0)
        
        return buffer, df
    
    def test_upload(self):
        """ファイルアップロードのテスト"""
        print("\n1. Testing file upload...")
        
        buffer, self.test_data = self.create_test_data()
        files = {'file': ('test_data.xlsx', buffer, 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')}
        
        response = self.session.post(f"{self.base_url}/upload", files=files)
        assert response.status_code == 200, f"Upload failed: {response.text}"
        
        result = response.json()
        assert result['status'] == 'success', f"Upload status not success: {result}"
        print("✓ File upload successful")
        
    def test_feature_analysis(self):
        """特徴量分析のテスト"""
        print("\n2. Testing feature analysis...")
        
        response = self.session.get(f"{self.base_url}/api/get_feature_analysis")
        assert response.status_code == 200, f"Feature analysis failed: {response.text}"
        
        result = response.json()
        assert result['status'] == 'success', f"Analysis status not success: {result}"
        assert 'columns_info' in result, "Missing columns_info"
        assert 'numeric' in result['columns_info'], "Missing numeric columns"
        assert 'categorical' in result['columns_info'], "Missing categorical columns"
        assert 'date' in result['columns_info'], "Missing date columns"
        
        # 列の分類が正しいか確認
        numeric_cols = result['columns_info']['numeric']
        assert any('売上高' in col for col in numeric_cols), "Sales column not in numeric"
        
        categorical_cols = result['columns_info']['categorical']
        assert any('店舗' in col for col in categorical_cols), "Shop column not in categorical"
        
        print(f"✓ Feature analysis successful")
        print(f"  - Numeric columns: {len(numeric_cols)}")
        print(f"  - Categorical columns: {len(categorical_cols)}")
        print(f"  - Date columns: {len(result['columns_info']['date'])}")
        
    def test_improvement_proposal(self):
        """改善提案のテスト"""
        print("\n3. Testing improvement proposal...")
        
        # Operating_profitには欠損値があるはず
        response = self.session.post(
            f"{self.base_url}/api/propose_improvement",
            json={'field_name': 'Operating_profit'}
        )
        assert response.status_code == 200, f"Proposal failed: {response.text}"
        
        result = response.json()
        assert result['status'] == 'success', f"Proposal status not success: {result}"
        assert 'suggestion' in result, "Missing suggestion"
        assert 'options' in result, "Missing options"
        
        print("✓ Improvement proposal successful")
        print(f"  - Options available: {list(result['options'].keys())}")
        
    def test_histogram_generation(self):
        """ヒストグラム生成のテスト"""
        print("\n4. Testing histogram generation...")
        
        # 全店舗のヒストグラム
        response = self.session.post(
            f"{self.base_url}/api/generate_graph",
            json={
                'graph_type': 'histogram',
                'target_col': '店舗別売上高',
                'filter_shop': '',
                'group_by_col': ''
            }
        )
        assert response.status_code == 200, f"Histogram failed: {response.text}"
        
        result = response.json()
        assert result['status'] == 'success', f"Histogram status not success: {result}"
        assert 'graph_json' in result, "Missing graph_json"
        
        print("✓ Histogram (all shops) successful")
        
        # 特定店舗のヒストグラム
        response = self.session.post(
            f"{self.base_url}/api/generate_graph",
            json={
                'graph_type': 'histogram',
                'target_col': '営業利益',
                'filter_shop': '店舗A',
                'group_by_col': ''
            }
        )
        assert response.status_code == 200, f"Filtered histogram failed: {response.text}"
        
        result = response.json()
        assert result['status'] == 'success', f"Filtered histogram status not success"
        print("✓ Histogram (filtered) successful")
        
    def test_line_chart_generation(self):
        """折れ線グラフ生成のテスト"""
        print("\n5. Testing line chart generation...")
        
        response = self.session.post(
            f"{self.base_url}/api/generate_graph",
            json={
                'graph_type': 'line',
                'date_col': '営業日付',
                'value_col': '店舗別売上高',
                'filter_shop': '',
                'date_start': '2024-01-01',
                'date_end': '2024-03-31'
            }
        )
        assert response.status_code == 200, f"Line chart failed: {response.text}"
        
        result = response.json()
        assert result['status'] == 'success', f"Line chart status not success: {result}"
        assert 'graph_json' in result, "Missing graph_json"
        
        print("✓ Line chart successful")
        
    def test_shop_list(self):
        """店舗リスト取得のテスト"""
        print("\n6. Testing shop list...")
        
        response = self.session.get(f"{self.base_url}/api/get_available_shops")
        assert response.status_code == 200, f"Shop list failed: {response.text}"
        
        result = response.json()
        assert result['status'] == 'success', f"Shop list status not success: {result}"
        assert 'shops' in result, "Missing shops"
        assert len(result['shops']) == 3, f"Expected 3 shops, got {len(result['shops'])}"
        assert '店舗A' in result['shops'], "Missing 店舗A"
        
        print("✓ Shop list successful")
        print(f"  - Shops: {result['shops']}")
        
    def test_agent_status(self):
        """エージェントステータスのテスト（新規API）"""
        print("\n7. Testing agent status (new API)...")
        
        response = self.session.get(f"{self.base_url}/api/agent_status")
        assert response.status_code == 200, f"Agent status failed: {response.text}"
        
        result = response.json()
        assert result['status'] == 'success', f"Agent status not success: {result}"
        assert 'agent_info' in result, "Missing agent_info"
        assert result['agent_info']['name'] == 'DataAnalysisAgent', "Wrong agent name"
        
        print("✓ Agent status successful")
        print(f"  - Agent: {result['agent_info']['name']}")
        print(f"  - Capabilities: {result['agent_info']['capabilities']}")
        
    def run_all_tests(self):
        """全テストを実行"""
        print("=" * 60)
        print("Q-Storm Agent Migration Test Suite")
        print("=" * 60)
        
        try:
            # サーバーが起動しているか確認
            response = requests.get(f"{self.base_url}/")
            assert response.status_code == 200, "Server not responding"
            print("✓ Server is running")
            
            # 各テストを実行
            self.test_upload()
            self.test_feature_analysis()
            self.test_improvement_proposal()
            self.test_histogram_generation()
            self.test_line_chart_generation()
            self.test_shop_list()
            self.test_agent_status()
            
            print("\n" + "=" * 60)
            print("✅ All tests passed successfully!")
            print("The agent migration maintains backward compatibility.")
            print("=" * 60)
            
        except AssertionError as e:
            print(f"\n❌ Test failed: {e}")
            return False
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            return False
        
        return True

def main():
    """メイン実行"""
    print("\nStarting agent migration test...")
    print("Make sure app_with_agent.py is running on port 5001")
    print("You can start it with: python app_with_agent.py\n")
    
    input("Press Enter when server is ready...")
    
    tester = TestAgentMigration()
    success = tester.run_all_tests()
    
    if success:
        print("\n🎉 Migration test completed successfully!")
        print("You can now safely use app_with_agent.py instead of app.py")
    else:
        print("\n⚠️ Migration test failed. Please check the errors above.")

if __name__ == "__main__":
    main()