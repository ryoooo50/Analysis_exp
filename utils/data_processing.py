"""
データ処理関連のユーティリティ関数
"""
import pandas as pd
import numpy as np
from typing import List, Tuple
from scipy.signal import butter, filtfilt


def load_csv_data(file_stream) -> pd.DataFrame:
    """
    CSVファイルを読み込み、列名の空白を削除
    
    Args:
        file_stream: ファイルストリーム
        
    Returns:
        読み込まれたDataFrame
        
    Raises:
        ValueError: 必要な列が見つからない場合
    """
    try:
        df = pd.read_csv(file_stream, encoding='utf-8')
    except UnicodeDecodeError:
        file_stream.seek(0)
        df = pd.read_csv(file_stream, encoding='shift_jis')
    
    df = df.rename(columns=lambda x: x.strip())
    return df


def calculate_angular_velocity(df: pd.DataFrame, time_col: str, angle_rad_col: str) -> pd.DataFrame:
    """
    角速度を計算してDataFrameに追加
    
    Args:
        df: データフレーム
        time_col: 時間列名
        angle_rad_col: 角度（ラジアン）列名
        
    Returns:
        角速度が追加されたDataFrame
    """
    time_diff = df[time_col].diff()
    angle_diff = df[angle_rad_col].diff()
    angular_velocity = np.where(time_diff > 0, angle_diff / time_diff, 0)
    df['angular_velocity_raw'] = np.abs(angular_velocity)
    return df


def apply_lowpass_filter(data: np.ndarray, cutoff: float, fs: float, order: int = 4) -> np.ndarray:
    """
    ローパスフィルタを適用
    
    Args:
        data: フィルタリングするデータ
        cutoff: カットオフ周波数
        fs: サンプリング周波数
        order: フィルタ次数
        
    Returns:
        フィルタリングされたデータ
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data


def parse_text_data(data_string: str) -> List[float]:
    """
    カンマ区切りのテキストを数値のリストに変換する
    
    Args:
        data_string: カンマ区切りの数値文字列
        
    Returns:
        数値のリスト
        
    Raises:
        ValueError: 数値に変換できない文字列が含まれる場合
    """
    if not data_string:
        return []
    
    # 全角カンマを半角に置換、空白を削除
    data_string = data_string.replace('，', ',').strip()
    
    values = []
    for item in data_string.split(','):
        item = item.strip()
        if item:
            try:
                values.append(float(item))
            except ValueError:
                raise ValueError(
                    f"データ '{item}' を数値に変換できません。"
                    "カンマ区切りで数値を入力してください。"
                )
    return values




