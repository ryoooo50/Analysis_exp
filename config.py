"""
アプリケーション設定ファイル
定数とデフォルト値を定義
"""
from typing import Dict, Any

# CSVファイルの列名
TIME_COL = 'time'
ANGLE_COL = 'Lower_Arm_R/rotation_y'

# フィルタリング設定
CUTOFF_FREQ = 15.0
FILTER_ORDER = 4

# ピーク検出のデフォルトパラメータ
DEFAULT_PEAK_PARAMS: Dict[str, Any] = {
    'min_peak_height': 1.0,
    'max_peak_height': 100.0,
    'peak_prominence': 1.0,
    'peak_distance': 50,
}

# 統計検定のデフォルト設定
DEFAULT_ALTERNATIVE = 'two-sided'
SIGNIFICANCE_LEVEL = 0.05

# ピーク平均のグループサイズ
PEAK_GROUP_SIZE = 10




