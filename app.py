# app.py (平均値計算機能付き 完全版)

from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import io
import base64

# --- データ分析に必要なライブラリをインポート ---
from scipy.signal import butter, filtfilt, find_peaks
import matplotlib
# バックエンドでmatplotlibを正しく動作させるためのおまじない
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)

# ===================================================================
# STEP 1: Jupyter Notebookから分析関数を移植
# ===================================================================

# Matplotlibの日本語設定
try:
    plt.rcParams['font.family'] = 'Hiragino Sans'
    plt.rcParams['font.sans-serif'] = ['Hiragino Sans']
except:
    try:
        plt.rcParams['font.family'] = 'Yu Gothic'
        plt.rcParams['font.sans-serif'] = ['Yu Gothic']
    except:
        print("警告: 日本語フォントが見つかりません。グラフのラベルが文字化けする可能性があります。")

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
    peaks, properties = find_peaks(
        df[velocity_col],
        height=height,
        prominence=prominence,
        distance=distance
    )
    return peaks, properties

# ===================================================================
# STEP 2: メインの分析処理とWeb APIを定義
# ===================================================================

def analyze_data_from_stream(file_stream, params):
    """アップロードされたファイルストリームとパラメータを受け取り、分析を実行して結果を返す関数"""
    
    # --- 1. データの読み込みと前処理 ---
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

    # --- 2. 角速度の計算、フィルタリング、ピーク検出 ---
    df = calculate_angular_velocity(df, TIME_COL, 'angle_rad')
    sampling_freq = 1 / df[TIME_COL].diff().mean()
    df['angular_velocity_filtered'] = apply_lowpass_filter(
        df['angular_velocity_raw'].fillna(0), 
        cutoff=CUTOFF_FREQ, 
        fs=sampling_freq
    )
    peaks, properties = find_velocity_peaks(
        df, 'angular_velocity_filtered',
        height=params['peak_height'],
        prominence=params['peak_prominence'],
        distance=params['peak_distance']
    )
    
    # --- 3. 結果をフロントエンドに返す形式に変換 ---
    peak_info = pd.DataFrame({
        'peak_time_s': df.loc[peaks, TIME_COL].values,
        'peak_velocity_rad_s': properties['peak_heights'],
    })
    
    # --- 3-EX. 10回ごとの平均値を計算 (ここが新しい部分) ---
    peak_averages = []
    if not peak_info.empty:
        # 10個ずつのグループに分けるためのインデックスを作成
        num_peaks = len(peak_info)
        group_indices = np.arange(num_peaks) // 10
        
        # グループごとに平均値を計算
        avg_data = peak_info.groupby(group_indices)['peak_velocity_rad_s'].mean()
        
        # 結果を整形
        for i, avg_val in avg_data.items():
            start_peak = i * 10 + 1
            end_peak = min((i + 1) * 10, num_peaks)
            peak_averages.append({
                "interval": f"ピーク {start_peak}-{end_peak}",
                "average_velocity": avg_val
            })

    # (3-2) グラフ画像の作成
    fig, ax1 = plt.subplots(figsize=(12, 6))
    sns.set_style("whitegrid")
    ax1.set_xlabel('時間 (s)')
    ax1.set_ylabel('角速度 (rad/s)', color='tab:blue')
    ax1.plot(df[TIME_COL], df['angular_velocity_filtered'], color='tab:blue', label='角速度')
    ax1.plot(df.loc[peaks, TIME_COL], df.loc[peaks, 'angular_velocity_filtered'], "x", color='red', markersize=10, label='ピーク')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.legend()
    plt.title('角速度と検出されたピーク')
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
        "peak_averages": peak_averages  # 新しく追加したデータ
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_endpoint():
    if 'file' not in request.files:
        return jsonify({"error": "ファイルがありません"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "ファイルが選択されていません"}), 400
    
    if file:
        try:
            params = {
                'peak_height': float(request.form.get('peak_height', 1.0)),
                'peak_prominence': float(request.form.get('peak_prominence', 1.0)),
                'peak_distance': int(request.form.get('peak_distance', 50)),
            }
            results = analyze_data_from_stream(file.stream, params)
            return jsonify(results)
        except Exception as e:
            print(f"分析エラー: {e}")
            return jsonify({"error": f"分析中にエラーが発生しました: {e}"}), 500

if __name__ == '__main__':
    app.run(debug=True)