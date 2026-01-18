"""
データ処理関連のユーティリティ関数
"""
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict
from scipy.signal import butter, filtfilt
from datetime import datetime
import io


def detect_csv_format(file_stream) -> str:
    """
    CSVファイルのフォーマットを判別する
    
    Args:
        file_stream: ファイルストリーム
        
    Returns:
        'vrm_log': VRMログフォーマット（from,time,mode,data構造）
        'simple': シンプルフォーマット（time, angle列が直接ある）
    """
    # 先頭行を読み込んで判別
    start_pos = file_stream.tell()
    try:
        first_line = file_stream.readline()
        if isinstance(first_line, bytes):
            first_line = first_line.decode('utf-8', errors='replace')
        first_line = first_line.strip()
    except Exception:
        file_stream.seek(start_pos)
        return 'simple'
    
    file_stream.seek(start_pos)
    
    # VRMログフォーマットの判別: 最初の行が "from,time,mode,data" で始まる
    if first_line.lower().startswith('from,time,mode,data'):
        return 'vrm_log'
    
    return 'simple'


def get_available_joints_from_vrm_log(file_stream) -> List[str]:
    """
    VRMログファイルから利用可能な関節（rotation列）のリストを取得する
    
    Args:
        file_stream: ファイルストリーム
        
    Returns:
        利用可能な関節名のリスト（rotation_y列のみ）
    """
    start_pos = file_stream.tell()
    
    try:
        content = file_stream.read()
        if isinstance(content, bytes):
            content = content.decode('utf-8')
    except UnicodeDecodeError:
        file_stream.seek(start_pos)
        content = file_stream.read().decode('shift_jis')
    
    # ファイルポインタを元に戻す
    file_stream.seek(start_pos)
    
    lines = content.strip().split('\n')
    
    # モード13（ボディトラッキングデータ）のヘッダーを探す
    for line in lines:
        parts = line.split(',')
        if len(parts) >= 4 and parts[0] == 'header_define' and parts[2] == '13':
            columns = [col.strip() for col in parts[3:] if col.strip()]
            # rotation_y列のみを抽出（リーチング分析に適した回転データ）
            rotation_columns = [col for col in columns if '/rotation_y' in col]
            return rotation_columns
    
    return []


def parse_vrm_log_csv(file_stream, target_column: Optional[str] = None) -> pd.DataFrame:
    """
    VRMログフォーマットのCSVを解析し、標準的なDataFrameに変換する
    
    VRMログフォーマットの構造:
    - 最初の行: from,time,mode,data
    - header_define行: 各モードのカラム定義
    - データ行: clientReceive/clientSendなどから始まる
    
    Args:
        file_stream: ファイルストリーム
        target_column: 抽出する角度列名（Noneの場合は'Lower_Arm_R/rotation_y'）
        
    Returns:
        時間と角度データを含むDataFrame
    """
    if target_column is None:
        target_column = 'Lower_Arm_R/rotation_y'
    
    # ファイル全体を読み込む
    try:
        content = file_stream.read()
        if isinstance(content, bytes):
            content = content.decode('utf-8')
    except UnicodeDecodeError:
        file_stream.seek(0)
        content = file_stream.read().decode('shift_jis')
    
    lines = content.strip().split('\n')
    
    if len(lines) < 2:
        raise ValueError("VRMログファイルが空または不正な形式です")
    
    # モード13（ボディトラッキングデータ）のヘッダーを探す
    mode_13_columns: Optional[List[str]] = None
    
    for line in lines:
        parts = line.split(',')
        if len(parts) >= 4 and parts[0] == 'header_define' and parts[2] == '13':
            # モード13のカラム定義を取得
            mode_13_columns = [col.strip() for col in parts[3:] if col.strip()]
            break
    
    if mode_13_columns is None:
        raise ValueError("モード13（ボディトラッキングデータ）のヘッダー定義が見つかりません")
    
    # 対象列の位置を特定
    try:
        target_idx = mode_13_columns.index(target_column)
    except ValueError:
        raise ValueError(f"必要な列 '{target_column}' がモード13のデータに見つかりません")
    
    # データ行をフィルタリングして一括処理（最適化版）
    # clientReceiveで始まり、モード13のデータ行のみを抽出
    data_lines = []
    for line in lines:
        if line.startswith('clientReceive') and ',13,' in line:
            data_lines.append(line)
    
    if not data_lines:
        raise ValueError("有効なデータ行が見つかりませんでした")
    
    # pandasで一括処理
    # データをリスト形式で高速に抽出
    time_strings = []
    angle_values = []
    
    for line in data_lines:
        parts = line.split(',')
        if len(parts) >= 4 + target_idx:
            time_str = parts[1].strip()
            angle_str = parts[3 + target_idx].strip()
            if time_str and angle_str:
                try:
                    angle_val = float(angle_str)
                    time_strings.append(time_str)
                    angle_values.append(angle_val)
                except ValueError:
                    continue
    
    if not time_strings:
        raise ValueError("有効なデータ行が見つかりませんでした")
    
    # pd.to_datetimeで一括変換（datetime.strptimeより大幅に高速）
    time_datetimes = pd.to_datetime(time_strings, format='%Y/%m/%d %H:%M:%S.%f')
    
    # 時間を秒単位に変換（最初のタイムスタンプからの経過秒数）
    base_time = time_datetimes[0]
    time_seconds = (time_datetimes - base_time).total_seconds()
    
    # DataFrameを作成
    df = pd.DataFrame({
        'time': time_seconds,
        target_column: angle_values
    })
    
    return df


def load_csv_data(file_stream, target_column: Optional[str] = None) -> pd.DataFrame:
    """
    CSVファイルを読み込み、フォーマットを自動判別して処理
    
    Args:
        file_stream: ファイルストリーム
        target_column: VRMログの場合に抽出する角度列名（Noneの場合はデフォルト値を使用）
        
    Returns:
        読み込まれたDataFrame
        
    Raises:
        ValueError: 必要な列が見つからない場合
    """
    # フォーマットを判別
    csv_format = detect_csv_format(file_stream)
    
    if csv_format == 'vrm_log':
        # VRMログフォーマットの場合
        return parse_vrm_log_csv(file_stream, target_column)
    else:
        # シンプルフォーマットの場合（従来の処理）
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
    カンマ区切りまたは改行区切りのテキストを数値のリストに変換する
    
    Args:
        data_string: カンマ区切りまたは改行区切りの数値文字列
        
    Returns:
        数値のリスト
        
    Raises:
        ValueError: 数値に変換できない文字列が含まれる場合
    """
    if not data_string:
        return []
    
    # 全角カンマを半角に置換
    data_string = data_string.replace('，', ',')
    
    # 改行（\r\n, \n, \r）をカンマに変換して統一
    data_string = data_string.replace('\r\n', ',').replace('\n', ',').replace('\r', ',')
    
    # 空白を削除
    data_string = data_string.strip()
    
    values = []
    for item in data_string.split(','):
        item = item.strip()
        if item:
            try:
                values.append(float(item))
            except ValueError:
                raise ValueError(
                    f"データ '{item}' を数値に変換できません。"
                    "カンマ区切りまたは改行区切りで数値を入力してください。"
                )
    return values




