# app.py (英語表記・フォント問題解決版)

from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import io
import base64
import os

from scipy.signal import butter, filtfilt, find_peaks
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)

# --- 分析関数 (変更なし) ---
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
        df[velocity_col],
        height=height,
        prominence=prominence,
        distance=distance
    )
    if find_peaks_result and len(find_peaks_result) >= 2:
        peaks = find_peaks_result[0]
        properties = find_peaks_result[1]
        return peaks, properties
    else:
        return np.array([]), {}

# --- メイン分析処理 (グラフ部分を英語に変更) ---
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
        df['angular_velocity_raw'].fillna(0), 
        cutoff=CUTOFF_FREQ, 
        fs=sampling_freq
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
            # 【修正】区間名を英語に変更
            peak_averages.append({
                "interval": f"Peaks {start_peak}-{end_peak}",
                "average_velocity": avg_val
            })

    # 【修正】グラフのテキストをすべて英語に変更
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

# --- Flaskルーティング (変更なし) ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_endpoint():
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
            print(f"分析エラー: {e}")
            return jsonify({"error": f"分析中にエラーが発生しました: {e}"}), 500

if __name__ == '__main__':
    app.run(debug=True)