"""
Flaskアプリケーション - リーチング分析・統計アプリ
ピーク分析と統計検定機能を提供
"""
from flask import Flask, request, jsonify, render_template
from typing import Dict, Any, Tuple

from utils.data_processing import parse_text_data
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
    params = {
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
    
    # 対象関節（指定がある場合のみ追加）
    target_joint = request_form.get('target_joint')
    if target_joint and target_joint.strip():
        params['target_joint'] = target_joint.strip()
    
    return params


def _parse_input_dataset(raw_data: Any, field_name: str) -> list:
    """
    JSONペイロードから受け取ったデータを数値リストに変換

    Args:
        raw_data: 文字列（カンマ区切り）または数値リスト
        field_name: フィールド名（エラーメッセージ用）

    Returns:
        数値リスト

    Raises:
        ValueError: データ変換に失敗した場合
    """
    if raw_data is None:
        raise ValueError(f"'{field_name}' にデータがありません。")

    if isinstance(raw_data, (list, tuple)):
        converted = []
        for idx, item in enumerate(raw_data):
            try:
                converted.append(float(item))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"'{field_name}' の値を数値に変換できません (index {idx}): {item}"
                ) from exc
        if not converted:
            raise ValueError(f"'{field_name}' が空です。")
        return converted

    if isinstance(raw_data, str):
        values = parse_text_data(raw_data)
        if not values:
            raise ValueError(f"'{field_name}' が空です。")
    return values

    raise ValueError(f"'{field_name}' の形式が不正です。文字列または数値リストを指定してください。")


def _parse_groups_payload(raw_groups: Any) -> list:
    """
    JSONペイロードから受け取った複数グループデータを数値リストのリストに変換

    Args:
        raw_groups: グループデータのリスト

    Returns:
        数値リストのリスト

    Raises:
        ValueError: データ変換に失敗した場合
    """
    if not isinstance(raw_groups, list) or len(raw_groups) < 2:
        raise ValueError("2つ以上のグループデータをリストで指定してください。")

    parsed_groups = []
    for idx, raw_group in enumerate(raw_groups):
        group_values = _parse_input_dataset(raw_group, f'groups[{idx}]')
        if len(group_values) < 2:
            raise ValueError(f"グループ {idx + 1} のデータ数が不足しています (2以上必要)。")
        parsed_groups.append(group_values)

    return parsed_groups


def _parse_alpha_value(raw_alpha: Any) -> float:
    """
    フロントエンドから渡された有意水準を安全に数値へ変換

    Args:
        raw_alpha: 変換対象の値

    Returns:
        0 < alpha <= 1 の範囲に正規化された有意水準
    """
    try:
        alpha = float(raw_alpha)
    except (TypeError, ValueError):
        return config.SIGNIFICANCE_LEVEL

    if alpha <= 0 or alpha > 1:
        return config.SIGNIFICANCE_LEVEL

    return alpha


@app.route('/')
def index():
    """メインページを表示"""
    return render_template('index.html')


@app.route('/get-available-joints', methods=['POST'])
def get_available_joints():
    """
    CSVファイルから利用可能な関節リストを取得するエンドポイント
    VRMログフォーマットの場合、モード13のrotation_y列を抽出して返す
    """
    if 'file' not in request.files:
        return jsonify({"error": "ファイルがありません"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "ファイルが選択されていません"}), 400
    
    try:
        from utils.data_processing import detect_csv_format, get_available_joints_from_vrm_log
        
        # フォーマットを判別
        csv_format = detect_csv_format(file.stream)
        
        if csv_format == 'vrm_log':
            # VRMログフォーマットの場合、利用可能な関節リストを取得
            joints = get_available_joints_from_vrm_log(file.stream)
            return jsonify({
                "format": "vrm_log",
                "joints": joints,
                "default_joint": config.ANGLE_COL
            })
        else:
            # シンプルフォーマットの場合、固定の関節を返す
            return jsonify({
                "format": "simple",
                "joints": [config.ANGLE_COL],
                "default_joint": config.ANGLE_COL
            })
    except Exception as e:
        return handle_error(e, "GetJoints")


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
        from utils.peak_analysis import analyze_peak_data
        from utils.plotting import plot_angular_velocity_with_peaks

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
        from utils.statistical_tests import perform_mannwhitney_test
        # データの取得とパース
        data1_str = request.form.get('data1')
        data2_str = request.form.get('data2')
        alternative = request.form.get('alternative', config.DEFAULT_ALTERNATIVE)
        alpha = _parse_alpha_value(request.form.get('alpha', config.SIGNIFICANCE_LEVEL))
        
        group1 = parse_text_data(data1_str)
        group2 = parse_text_data(data2_str)
        
        # データの検証
        if not group1 or not group2:
             raise ValueError("両方のグループにデータを入力してください。")
        
        # 検定の実行
        results = perform_mannwhitney_test(group1, group2, alternative)
        results['alpha'] = alpha
        results['significant'] = bool(results['p_value'] <= alpha)
        
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
        from utils.statistical_tests import perform_wilcoxon_test
        # データの取得とパース
        data1_str = request.form.get('data1')
        data2_str = request.form.get('data2')
        alternative = request.form.get('alternative', config.DEFAULT_ALTERNATIVE)
        alpha = _parse_alpha_value(request.form.get('alpha', config.SIGNIFICANCE_LEVEL))

        group1 = parse_text_data(data1_str)
        group2 = parse_text_data(data2_str)

        # データの検証
        if not group1 or not group2:
             raise ValueError("両方のグループにデータを入力してください。")
        
        # 検定の実行
        results = perform_wilcoxon_test(group1, group2, alternative)
        results['alpha'] = alpha
        results['significant'] = bool(results['p_value'] <= alpha)
        
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
        from utils.statistical_tests import perform_ttest
        # データの取得とパース
        data1_str = request.form.get('data1')
        data2_str = request.form.get('data2')
        alternative = request.form.get('alternative', config.DEFAULT_ALTERNATIVE)
        alpha = _parse_alpha_value(request.form.get('alpha', config.SIGNIFICANCE_LEVEL))

        group1 = parse_text_data(data1_str)
        group2 = parse_text_data(data2_str)

        # データの検証
        if not group1 or not group2:
             raise ValueError("両方のグループにデータを入力してください。")
        
        # 検定の実行
        results = perform_ttest(group1, group2, alternative)
        results['alpha'] = alpha
        results['significant'] = bool(results['p_value'] <= alpha)
        
        return jsonify(results)
        
    except Exception as e:
        return handle_error(e, "t-test")


@app.route('/analyze-anova', methods=['POST'])
def analyze_endpoint_anova():
    """
    分散分析 (ANOVA / Kruskal-Wallis) エンドポイント
    """
    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({"error": "JSONボディが必要です。"}), 400

    try:
        from utils.statistical_tests import perform_oneway_anova_or_kruskal
        groups_raw = payload.get('groups')
        groups = _parse_groups_payload(groups_raw)
        alpha = _parse_alpha_value(payload.get('alpha', config.SIGNIFICANCE_LEVEL))

        results = perform_oneway_anova_or_kruskal(groups, alpha=alpha)
        return jsonify(results)

    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return handle_error(exc, "ANOVA/Kruskal")


@app.route('/analyze-multiple-tests', methods=['POST'])
def analyze_multiple_tests():
    """
    複数の検定をまとめて実行し、Holm補正を適用するエンドポイント

    期待するJSON形式:
    {
        "tests": [
            {
                "type": "mann_whitney" | "wilcoxon" | "ttest",
                "data1": "...",  # 文字列(カンマ区切り)または数値リスト
                "data2": "...",  # 同上
                "alternative": "two-sided" | "less" | "greater"
            },
            ...
        ],
        "alpha": 0.05
    }
    """
    payload = request.get_json(silent=True)
    if not payload or 'tests' not in payload:
        return jsonify({"error": "JSONボディに 'tests' フィールドが必要です。"}), 400

    from utils.statistical_tests import (
        perform_mannwhitney_test,
        perform_wilcoxon_test,
        perform_ttest,
        apply_holm_correction,
    )

    tests_config = payload.get('tests', [])
    if not isinstance(tests_config, list) or not tests_config:
        return jsonify({"error": "'tests' は1件以上の要素を持つリストで指定してください。"}), 400

    alpha = _parse_alpha_value(payload.get('alpha', config.SIGNIFICANCE_LEVEL))

    dispatch_map = {
        'mann_whitney': perform_mannwhitney_test,
        'wilcoxon': perform_wilcoxon_test,
        'ttest': perform_ttest,
    }

    aggregated_results = []
    successful_indices = []
    p_values = []

    for idx, test_cfg in enumerate(tests_config):
        if not isinstance(test_cfg, dict):
            aggregated_results.append({
                "index": idx,
                "success": False,
                "error": "各テスト設定はオブジェクトで指定してください。"
            })
            continue

        test_type = test_cfg.get('type')
        if test_type not in dispatch_map:
            aggregated_results.append({
                "index": idx,
                "success": False,
                "error": f"未知の検定タイプです: {test_type}"
            })
            continue

        try:
            group1 = _parse_input_dataset(test_cfg.get('data1'), 'data1')
            group2 = _parse_input_dataset(test_cfg.get('data2'), 'data2')
            alternative = test_cfg.get('alternative', config.DEFAULT_ALTERNATIVE)

            result = dispatch_map[test_type](group1, group2, alternative)
            result['alpha'] = alpha
            result['significant'] = bool(result['p_value'] <= alpha)

            aggregated_results.append({
                "index": idx,
                "type": test_type,
                "success": True,
                "result": result
            })
            successful_indices.append(len(aggregated_results) - 1)
            p_values.append(result['p_value'])

        except Exception as exc:
            aggregated_results.append({
                "index": idx,
                "type": test_type,
                "success": False,
                "error": str(exc)
            })

    correction_summary = None
    if p_values:
        correction_summary = apply_holm_correction(p_values, alpha)
        adjusted_p_values = correction_summary["adjusted_p_values"]
        significance_flags = correction_summary["significant"]

        for result_index, adj_p, is_sig in zip(successful_indices, adjusted_p_values, significance_flags):
            aggregated_results[result_index]["result"]["adjusted_p_value"] = adj_p
            aggregated_results[result_index]["result"]["significant_after_correction"] = bool(is_sig)

    response_payload = {
        "alpha": alpha,
        "correction": correction_summary,
        "tests": aggregated_results
    }

    return jsonify(response_payload)


if __name__ == '__main__':
    app.run(debug=True)
