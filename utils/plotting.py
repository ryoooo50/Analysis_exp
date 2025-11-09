"""
プロット関連の関数
"""
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def plot_angular_velocity_with_peaks(
    df: pd.DataFrame,
    time_col: str,
    velocity_col: str,
    peaks: np.ndarray
) -> str:
    """
    角速度と検出されたピークをプロットし、base64エンコードされた画像を返す
    
    Args:
        df: データフレーム
        time_col: 時間列名
        velocity_col: 角速度列名
        peaks: ピークインデックスの配列
        
    Returns:
        base64エンコードされた画像文字列
    """
    fig, ax1 = plt.subplots(figsize=(12, 6))
    sns.set_style("whitegrid")
    
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Angular Velocity (rad/s)', color='tab:blue')
    ax1.plot(
        df[time_col],
        df[velocity_col],
        color='tab:blue',
        label='Angular Velocity'
    )
    ax1.plot(
        df.loc[peaks, time_col],
        df.loc[peaks, velocity_col],
        "x",
        color='red',
        markersize=10,
        label='Peak'
    )
    ax1.legend()
    plt.title('Angular Velocity and Detected Peaks')
    plt.grid(True)
    
    # base64エンコード
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    plt.close(fig)
    
    return img_b64



