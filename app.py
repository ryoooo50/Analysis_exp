# app.py (テキスト入力版)

from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import io
import base64
import os

from scipy.signal import butter, filtfilt, find_peaks
from scipy.stats import mannwhitneyu, wilcoxon, ttest_ind,shapiro,levene # 統計検定ライブラリ
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)

# ===================================================================
# セクション 1: ピーク分析 (変更なし)
# ===================================================================
# (このセクションの関数は前回のコードと同一です)

def calculate_angular_velocity(df, time_col, angle_rad_col):
    time_diff = df[time_col].diff()
    angle_diff = df[angle_rad_col].diff()
    angular_velocity = np.where(time_diff > 0, angle_diff / time_diff, 0)
    df['angular_velocity_raw'] = np.abs(angular_velocity)
    return df

def apply_lowpass_filter(data, cutoff, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

def find_velocity_peaks(df, velocity_col, height, prominence, distance):
    find_peaks_result = find_peaks(
        df[velocity_col], height=height, prominence=prominence, distance=distance
    )
    if find_peaks_result and len(find_peaks_result) >= 2:
        return find_peaks_result[0], find_peaks_result[1]
    else:
        return np.array([]), {}

def analyze_data_from_stream(file_stream, params):
    try:
        df = pd.read_csv(file_stream, encoding='utf-8')
    except UnicodeDecodeError:
        file_stream.seek(0)
        df = pd.read_csv(file_stream, encoding='shift_jis')

    df = df.rename(columns=lambda x: x.strip())
    
    TIME_COL = 'time'
    ANGLE_COL = 'Lower_Arm_R/rotation_y'
    CUTOFF_FREQ = 5.0

    if TIME_COL not in df.columns or ANGLE_COL not in df.columns:
        raise ValueError(f"CSVに '{TIME_COL}' または '{ANGLE_COL}' の列が見つかりません。")

    df['angle_rad'] = np.deg2rad(df[ANGLE_COL])
    df = calculate_angular_velocity(df, TIME_COL, 'angle_rad')
    sampling_freq = 1 / df[TIME_COL].diff().mean()
    df['angular_velocity_filtered'] = apply_lowpass_filter(
        df['angular_velocity_raw'].fillna(0), cutoff=CUTOFF_FREQ, fs=sampling_freq
    )
    peak_height_range = (params['min_peak_height'], params['max_peak_height'])
    peaks, properties = find_velocity_peaks(
        df, 'angular_velocity_filtered',
        height=peak_height_range,
        prominence=params['peak_prominence'],
        distance=params['peak_distance']
    )
    peak_info = pd.DataFrame({
        'peak_time_s': df.loc[peaks, TIME_COL].to_numpy(),
        'peak_velocity_rad_s': properties.get('peak_heights', np.array([]))
    })
    peak_averages = []
    if not peak_info.empty:
        num_peaks = len(peak_info)
        group_indices = np.arange(num_peaks) // 10
        avg_data = peak_info.groupby(group_indices)['peak_velocity_rad_s'].mean()
        for i_raw, avg_val in avg_data.items():
            i = int(i_raw)
            start_peak = i * 10 + 1
            end_peak = min((i + 1) * 10, num_peaks)
            peak_averages.append({
                "interval": f"Peaks {start_peak}-{end_peak}",
                "average_velocity": avg_val
            })

    fig, ax1 = plt.subplots(figsize=(12, 6))
    sns.set_style("whitegrid")
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Angular Velocity (rad/s)', color='tab:blue')
    ax1.plot(df[TIME_COL], df['angular_velocity_filtered'], color='tab:blue', label='Angular Velocity')
    ax1.plot(df.loc[peaks, TIME_COL], df.loc[peaks, 'angular_velocity_filtered'], "x", color='red', markersize=10, label='Peak')
    ax1.legend()
    plt.title('Angular Velocity and Detected Peaks')
    plt.grid(True)
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    plt.close(fig)

    return {
        "image": img_b64, 
        "peaks": peak_info.to_dict(orient='records'),
        "peak_count": len(peaks),
        "peak_averages": peak_averages
    }

# ===================================================================
# セクション 2: 統計検定 (テキスト入力版に変更)
# ===================================================================

def parse_text_data(data_string):
    """カンマ区切りのテキストを数値のリストに変換する"""
    if not data_string:
        return []
    
    # 全角カンマを半角に置換、空白を削除
    data_string = data_string.replace('，', ',').strip()
    
    values = []
    for item in data_string.split(','):
        item = item.strip() # 前後の空白を削除
        if item: # 空の文字列は無視
            try:
                values.append(float(item))
            except ValueError:
                raise ValueError(f"データ '{item}' を数値に変換できません。カンマ区切りで数値を入力してください。")
    return values

def run_shapiro(data):
    """シャピロ・ウィルク検定を実行し、結果を辞書で返す"""
    if len(data) < 3: return {"stat": 0, "p": 0} # 3サンプル未満は検定不可
    stat, p = shapiro(data)
    return {"stat": stat, "p": p}

# ===================================================================
# セクション 3: Flaskルーティング (APIエンドポイント)
# ===================================================================

@app.route('/')
def index():
    return render_template('index.html')

# --- 1. ピーク分析用エンドポイント (変更なし) ---
@app.route('/analyze-peak', methods=['POST'])
def analyze_endpoint_peak():
    if 'file' not in request.files: return jsonify({"error": "ファイルがありません"}), 400
    file = request.files['file']
    if file.filename == '': return jsonify({"error": "ファイルが選択されていません"}), 400
    
    if file:
        try:
            params = {
                'min_peak_height': float(request.form.get('min_peak_height', 1.0)),
                'max_peak_height': float(request.form.get('max_peak_height', 100.0)),
                'peak_prominence': float(request.form.get('peak_prominence', 1.0)),
                'peak_distance': int(request.form.get('peak_distance', 50)),
            }
            results = analyze_data_from_stream(file.stream, params)
            return jsonify(results)
        except Exception as e:
            print(f"分析エラー(Peak): {e}")
            return jsonify({"error": f"分析中にエラーが発生しました: {e}"}), 500

# --- 2. マン・ホイットニーU検定用エンドポイント (テキスト入力版に変更) ---
@app.route('/analyze-mann-whitney', methods=['POST'])
def analyze_endpoint_mw():
    try:
        data1_str = request.form.get('data1')
        data2_str = request.form.get('data2')
        alternative = request.form.get('alternative', 'two-sided')
        
        group1 = parse_text_data(data1_str)
        group2 = parse_text_data(data2_str)
        
        if not group1 or not group2:
             raise ValueError("両方のグループにデータを入力してください。")
        
        u_stat, p_value = mannwhitneyu(group1, group2, alternative=alternative)
        
        return jsonify({"u_stat": u_stat, "p_value": p_value})
        
    except Exception as e:
        print(f"分析エラー(MW): {e}")
        return jsonify({"error": f"分析中にエラーが発生しました: {e}"}), 500

# --- 3. ウィルコクソン検定用エンドポイント (テキスト入力版に変更) ---
@app.route('/analyze-wilcoxon', methods=['POST'])
def analyze_endpoint_wilcoxon():
    try:
        data1_str = request.form.get('data1')
        data2_str = request.form.get('data2')
        alternative = request.form.get('alternative', 'two-sided')

        group1 = parse_text_data(data1_str)
        group2 = parse_text_data(data2_str)

        if not group1 or not group2:
             raise ValueError("両方のグループにデータを入力してください。")
        
        if len(group1) != len(group2):
            raise ValueError(f"ウィルコクソン検定は対応のあるデータ（同サイズ）が必要です。Group 1: {len(group1)}件, Group 2: {len(group2)}件")

        w_stat, p_value = wilcoxon(group1, group2, alternative=alternative)
        
        return jsonify({"w_stat": w_stat, "p_value": p_value})
        
    except Exception as e:
        print(f"分析エラー(Wilcoxon): {e}")
        return jsonify({"error": f"分析中にエラーが発生しました: {e}"}), 500
    

# --- 4. t検定用エンドポイント [新規追加] ---
@app.route('/analyze-ttest', methods=['POST'])
def analyze_endpoint_ttest():
    try:
        data1_str = request.form.get('data1')
        data2_str = request.form.get('data2')
        alternative = request.form.get('alternative', 'two-sided')

        group1 = parse_text_data(data1_str)
        group2 = parse_text_data(data2_str)

        if not group1 or not group2:
             raise ValueError("両方のグループにデータを入力してください。")
        
        # 正規性の検定
        shapiro1 = run_shapiro(group1)
        shapiro2 = run_shapiro(group2)
        is_normal = (shapiro1['p'] > 0.05 and shapiro2['p'] > 0.05)
        
        # t検定の前提条件が満たされない場合はメッセージを付与
        message = ""
        if not is_normal:
            message = "警告: データが正規分布に従っていないため、t検定の結果は信頼性が低い可能性があります。マン・ホイットニーU検定を推奨します。"

        # 等分散性の検定
        levene_stat, levene_p = levene(group1, group2)
        equal_var = (levene_p > 0.05)
        
        # t検定の実施
        stat, p_value = ttest_ind(group1, group2, equal_var=equal_var, alternative=alternative)
        
        return jsonify({
            "test_name": "対応のないt検定","stat_name": "t値",
            "stat": stat, "p_value": p_value,
            "shapiro1": shapiro1, "shapiro2": shapiro2,
            "normality": bool(is_normal),
            "levene": {"stat": levene_stat, "p": levene_p},
            "equal_var": bool(equal_var),
            "message": message
        })
        
    except Exception as e:
        print(f"分析エラー(ttest): {e}")
        return jsonify({"error": f"分析中にエラーが発生しました: {e}"}), 500

if __name__ == '__main__':
    app.run(debug=True)