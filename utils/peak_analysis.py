"""
ピーク分析関連の関数
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any
from scipy.signal import find_peaks

from utils.data_processing import (
    load_csv_data,
    calculate_angular_velocity,
    apply_lowpass_filter
)
import config


def find_velocity_peaks(
    df: pd.DataFrame,
    velocity_col: str,
    height: tuple,
    prominence: float,
    distance: int
) -> tuple:
    """
    角速度データからピークを検出
    
    Args:
        df: データフレーム
        velocity_col: 角速度列名
        height: ピークの高さ範囲 (min, max)
        prominence: ピークの顕著さ
        distance: ピーク間の最小距離
        
    Returns:
        (ピークインデックス配列, ピークプロパティ辞書) のタプル
    """
    find_peaks_result = find_peaks(
        df[velocity_col],
        height=height,
        prominence=prominence,
        distance=distance
    )
    if find_peaks_result and len(find_peaks_result) >= 2:
        return find_peaks_result[0], find_peaks_result[1]
    else:
        return np.array([]), {}


def calculate_peak_averages(peak_info: pd.DataFrame, group_size: int = 10) -> List[Dict[str, Any]]:
    """
    ピークをグループ化して平均を計算
    
    Args:
        peak_info: ピーク情報のDataFrame
        group_size: グループサイズ
        
    Returns:
        平均情報のリスト
    """
    peak_averages = []
    if not peak_info.empty:
        num_peaks = len(peak_info)
        group_indices = np.arange(num_peaks) // group_size
        avg_data = peak_info.groupby(group_indices)['peak_velocity_rad_s'].mean()
        
        for i_raw, avg_val in avg_data.items():
            i = int(i_raw)
            start_peak = i * group_size + 1
            end_peak = min((i + 1) * group_size, num_peaks)
            peak_averages.append({
                "interval": f"Peaks {start_peak}-{end_peak}",
                "average_velocity": avg_val
            })
    return peak_averages


def filter_outliers_iqr(peak_heights: np.ndarray, multiplier: float = 1.5) -> np.ndarray:
    """
    IQR法で異常値を検出し、正常範囲のインデックスマスクを返す
    
    Args:
        peak_heights: ピーク高さの配列
        multiplier: IQR倍率（デフォルト1.5）
        
    Returns:
        正常値のブールマスク配列
    """
    if len(peak_heights) == 0:
        return np.array([], dtype=bool)
    
    q1 = np.percentile(peak_heights, 25)
    q3 = np.percentile(peak_heights, 75)
    iqr = q3 - q1
    upper_bound = q3 + multiplier * iqr
    
    return peak_heights <= upper_bound


def analyze_peak_data(file_stream, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    CSVファイルからピーク分析を実行
    
    Args:
        file_stream: CSVファイルストリーム
        params: ピーク検出パラメータ
            - min_peak_height: 最小ピーク高さ
            - max_peak_height: 最大ピーク高さ
            - peak_prominence: ピークの顕著さ
            - peak_distance: ピーク間の最小距離
            - target_joint: 対象関節名（オプション、デフォルトはconfig.ANGLE_COL）
            - filter_outliers: 異常値除去を行うか（オプション、デフォルトFalse）
        
    Returns:
        分析結果の辞書
        
    Raises:
        ValueError: 必要な列が見つからない場合
    """
    # 対象関節を取得（指定がなければデフォルト値を使用）
    target_joint = params.get('target_joint', config.ANGLE_COL)
    filter_outliers = params.get('filter_outliers', False)
    
    df = load_csv_data(file_stream, target_joint)
    
    # 必要な列のチェック
    if config.TIME_COL not in df.columns or target_joint not in df.columns:
        raise ValueError(
            f"CSVに '{config.TIME_COL}' または '{target_joint}' の列が見つかりません。"
        )
    
    # 角度をラジアンに変換
    df['angle_rad'] = np.deg2rad(df[target_joint])
    
    # 角速度の計算
    df = calculate_angular_velocity(df, config.TIME_COL, 'angle_rad')
    
    # サンプリング周波数の計算
    sampling_freq = 1 / df[config.TIME_COL].diff().mean()
    
    # ローパスフィルタの適用
    df['angular_velocity_filtered'] = apply_lowpass_filter(
        df['angular_velocity_raw'].fillna(0),
        cutoff=config.CUTOFF_FREQ,
        fs=sampling_freq,
        order=config.FILTER_ORDER
    )
    
    # ピーク検出
    peak_height_range = (params['min_peak_height'], params['max_peak_height'])
    peaks, properties = find_velocity_peaks(
        df,
        'angular_velocity_filtered',
        height=peak_height_range,
        prominence=params['peak_prominence'],
        distance=params['peak_distance']
    )
    
    # 異常値フィルタリング（オプション）
    removed_outliers_count = 0
    if filter_outliers and len(peaks) > 0:
        peak_heights = properties.get('peak_heights', np.array([]))
        if len(peak_heights) > 0:
            valid_mask = filter_outliers_iqr(peak_heights)
            removed_outliers_count = int(len(peaks) - np.sum(valid_mask))
            peaks = peaks[valid_mask]
            properties['peak_heights'] = peak_heights[valid_mask]
    
    # ピーク情報の作成
    peak_info = pd.DataFrame({
        'peak_time_s': df.loc[peaks, config.TIME_COL].to_numpy(),
        'peak_velocity_rad_s': properties.get('peak_heights', np.array([]))
    })
    
    # ピーク平均の計算
    peak_averages = calculate_peak_averages(peak_info, config.PEAK_GROUP_SIZE)
    
    return {
        "peaks": peak_info.to_dict(orient='records'),
        "peak_count": len(peaks),
        "peak_averages": peak_averages,
        "df": df,
        "peak_indices": peaks,
        "removed_outliers_count": removed_outliers_count
    }




