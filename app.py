"""
Flaskアプリケーション - リーチング分析・統計アプリ
ピーク分析と統計検定機能を提供
"""
from flask import Flask, request, jsonify, render_template
from typing import Dict, Any, Tuple

from utils.data_processing import parse_text_data
from utils.peak_analysis import analyze_peak_data
from utils.plotting import plot_angular_velocity_with_peaks
from utils.statistical_tests import (
    perform_mannwhitney_test,
    perform_wilcoxon_test,
    perform_ttest
)
import config

app = Flask(__name__)


def handle_error(error: Exception, error_type: str = "エラー") -> Tuple[Dict[str, str], int]:
    """
    エラーハンドリングの統一関数
    
    Args:
        error: 発生した例外
        error_type: エラーの種類（ログ用）
        
    Returns:
        (エラーレスポンス辞書, HTTPステータスコード) のタプル
    """
    error_message = str(error)
    print(f"分析エラー({error_type}): {error_message}")
    return jsonify({"error": f"分析中にエラーが発生しました: {error_message}"}), 500


def extract_peak_params(request_form) -> Dict[str, Any]:
    """
    リクエストからピーク検出パラメータを抽出
    
    Args:
        request_form: Flask request.form オブジェクト
        
    Returns:
        ピーク検出パラメータの辞書
    """
    return {
        'min_peak_height': float(
            request_form.get('min_peak_height', config.DEFAULT_PEAK_PARAMS['min_peak_height'])
        ),
        'max_peak_height': float(
            request_form.get('max_peak_height', config.DEFAULT_PEAK_PARAMS['max_peak_height'])
        ),
        'peak_prominence': float(
            request_form.get('peak_prominence', config.DEFAULT_PEAK_PARAMS['peak_prominence'])
        ),
        'peak_distance': int(
            request_form.get('peak_distance', config.DEFAULT_PEAK_PARAMS['peak_distance'])
        ),
    }


@app.route('/')
def index():
    """メインページを表示"""
    return render_template('index.html')


@app.route('/analyze-peak', methods=['POST'])
def analyze_endpoint_peak():
    """
    ピーク分析エンドポイント
    CSVファイルから角速度のピークを検出し、分析結果を返す
    """
    # ファイルの検証
    if 'file' not in request.files:
        return jsonify({"error": "ファイルがありません"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "ファイルが選択されていません"}), 400
    
    try:
        # パラメータの抽出
        params = extract_peak_params(request.form)
        
        # ピーク分析の実行
        analysis_results = analyze_peak_data(file.stream, params)
        
        # プロットの生成
        img_b64 = plot_angular_velocity_with_peaks(
            analysis_results["df"],
            config.TIME_COL,
            'angular_velocity_filtered',
            analysis_results["peak_indices"]
        )
        
        # レスポンスの構築
        return jsonify({
            "image": img_b64,
            "peaks": analysis_results["peaks"],
            "peak_count": analysis_results["peak_count"],
            "peak_averages": analysis_results["peak_averages"]
        })
        
    except Exception as e:
        return handle_error(e, "Peak")


@app.route('/analyze-mann-whitney', methods=['POST'])
def analyze_endpoint_mw():
    """
    マン・ホイットニーU検定エンドポイント
    対応のない2群間のノンパラメトリック検定を実行
    """
    try:
        # データの取得とパース
        data1_str = request.form.get('data1')
        data2_str = request.form.get('data2')
        alternative = request.form.get('alternative', config.DEFAULT_ALTERNATIVE)
        
        group1 = parse_text_data(data1_str)
        group2 = parse_text_data(data2_str)
        
        # データの検証
        if not group1 or not group2:
            raise ValueError("両方のグループにデータを入力してください。")
        
        # 検定の実行
        results = perform_mannwhitney_test(group1, group2, alternative)
        
        return jsonify(results)
        
    except Exception as e:
        return handle_error(e, "Mann-Whitney")


@app.route('/analyze-wilcoxon', methods=['POST'])
def analyze_endpoint_wilcoxon():
    """
    ウィルコクソン符号順位検定エンドポイント
    対応のある2群間のノンパラメトリック検定を実行
    """
    try:
        # データの取得とパース
        data1_str = request.form.get('data1')
        data2_str = request.form.get('data2')
        alternative = request.form.get('alternative', config.DEFAULT_ALTERNATIVE)
        
        group1 = parse_text_data(data1_str)
        group2 = parse_text_data(data2_str)
        
        # データの検証
        if not group1 or not group2:
            raise ValueError("両方のグループにデータを入力してください。")
        
        # 検定の実行
        results = perform_wilcoxon_test(group1, group2, alternative)
        
        return jsonify(results)
        
    except Exception as e:
        return handle_error(e, "Wilcoxon")


@app.route('/analyze-ttest', methods=['POST'])
def analyze_endpoint_ttest():
    """
    t検定エンドポイント
    対応のない2群間のt検定を実行（正規性・等分散性の検定も含む）
    """
    try:
        # データの取得とパース
        data1_str = request.form.get('data1')
        data2_str = request.form.get('data2')
        alternative = request.form.get('alternative', config.DEFAULT_ALTERNATIVE)
        
        group1 = parse_text_data(data1_str)
        group2 = parse_text_data(data2_str)
        
        # データの検証
        if not group1 or not group2:
            raise ValueError("両方のグループにデータを入力してください。")
        
        # 検定の実行
        results = perform_ttest(group1, group2, alternative)
        
        return jsonify(results)
        
    except Exception as e:
        return handle_error(e, "t-test")


if __name__ == '__main__':
    app.run(debug=True)
