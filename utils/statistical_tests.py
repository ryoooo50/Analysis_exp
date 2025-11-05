"""
統計検定関連の関数
"""
import numpy as np
from typing import Dict, Any, Tuple
from scipy.stats import mannwhitneyu, wilcoxon, ttest_ind, shapiro, levene, norm


def run_shapiro(data: list) -> Dict[str, float]:
    """
    シャピロ・ウィルク検定を実行し、結果を辞書で返す
    
    Args:
        data: 検定するデータ
        
    Returns:
        {"stat": 統計量, "p": p値} の辞書
    """
    if len(data) < 3:
        return {"stat": 0, "p": 0}  # 3サンプル未満は検定不可
    stat, p = shapiro(data)
    return {"stat": stat, "p": p}


def calculate_z_and_r_from_p(p_value: float, N: int, alternative: str) -> Tuple[float, float]:
    """
    p値からZ値と効果量rを計算 (ノンパラメトリック検定用)
    
    Args:
        p_value: p値
        N: サンプルサイズ
        alternative: 検定の種類 ('two-sided', 'less', 'greater')
        
    Returns:
        (Z値, 効果量r) のタプル
    """
    if p_value == 0.0:
        Z_score = np.inf
    elif p_value == 1.0:
        Z_score = 0.0
    else:
        if alternative == 'two-sided':
            Z_score = np.abs(norm.ppf(p_value / 2))
        else:  # 'less' or 'greater'
            Z_score = np.abs(norm.ppf(p_value))
    
    if N > 0 and np.isfinite(Z_score):
        effect_size_r = Z_score / np.sqrt(N)
    else:
        effect_size_r = np.nan
    
    return Z_score, effect_size_r


def perform_mannwhitney_test(
    group1: list,
    group2: list,
    alternative: str = 'two-sided'
) -> Dict[str, Any]:
    """
    マン・ホイットニーU検定を実行
    
    Args:
        group1: グループ1のデータ
        group2: グループ2のデータ
        alternative: 検定の種類
        
    Returns:
        検定結果の辞書
    """
    u_stat, p_value = mannwhitneyu(group1, group2, alternative=alternative)
    
    # 効果量rの計算
    N = len(group1) + len(group2)
    Z_score, effect_size_r = calculate_z_and_r_from_p(p_value, N, alternative)
    
    return {
        "test_name": "マン・ホイットニーU検定",
        "stat_name": "U値",
        "stat": u_stat,
        "p_value": p_value,
        "z_approx": Z_score,
        "effect_size_r": effect_size_r
    }


def perform_wilcoxon_test(
    group1: list,
    group2: list,
    alternative: str = 'two-sided'
) -> Dict[str, Any]:
    """
    ウィルコクソン符号順位検定を実行
    
    Args:
        group1: グループ1のデータ
        group2: グループ2のデータ
        alternative: 検定の種類
        
    Returns:
        検定結果の辞書
        
    Raises:
        ValueError: データサイズが異なる場合
    """
    if len(group1) != len(group2):
        raise ValueError(
            f"ウィルコクソン検定は対応のあるデータ（同サイズ）が必要です。"
            f"Group 1: {len(group1)}件, Group 2: {len(group2)}件"
        )
    
    w_stat, p_value = wilcoxon(group1, group2, alternative=alternative)
    
    # 効果量rの計算
    n_pairs = len(group1)
    N_total_obs = n_pairs * 2
    Z_score, effect_size_r = calculate_z_and_r_from_p(p_value, N_total_obs, alternative)
    
    return {
        "test_name": "ウィルコクソン符号順位検定",
        "stat_name": "W値",
        "stat": w_stat,
        "p_value": p_value,
        "z_approx": Z_score,
        "effect_size_r": effect_size_r
    }


def perform_ttest(
    group1: list,
    group2: list,
    alternative: str = 'two-sided'
) -> Dict[str, Any]:
    """
    対応のないt検定を実行
    
    Args:
        group1: グループ1のデータ
        group2: グループ2のデータ
        alternative: 検定の種類
        
    Returns:
        検定結果の辞書
    """
    n1, n2 = len(group1), len(group2)
    mean1, mean2 = np.mean(group1), np.mean(group2)
    var1 = np.var(group1, ddof=1) if n1 > 1 else 0.0
    var2 = np.var(group2, ddof=1) if n2 > 1 else 0.0
    
    # 正規性の検定
    shapiro1 = run_shapiro(group1)
    shapiro2 = run_shapiro(group2)
    is_normal = (shapiro1['p'] > 0.05 and shapiro2['p'] > 0.05)
    
    message = ""
    if not is_normal:
        message = (
            "警告: データが正規分布に従っていないため、"
            "t検定の結果は信頼性が低い可能性があります。"
            "マン・ホイットニーU検定を推奨します。"
        )
    
    # 等分散性の検定
    levene_stat, levene_p = np.nan, np.nan
    equal_var = True
    if n1 > 1 and n2 > 1:
        levene_stat, levene_p = levene(group1, group2)
        equal_var = (levene_p > 0.05)
    
    # t検定の実施
    ttest_result = ttest_ind(group1, group2, equal_var=equal_var, alternative=alternative)
    t_statistic_val = float(ttest_result.statistic)
    p_value = float(ttest_result.pvalue)
    
    # 効果量とdfの計算
    cohen_d = np.nan
    if n1 > 1 and n2 > 1:
        s_pooled = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        cohen_d = (mean1 - mean2) / s_pooled if s_pooled > 0 else 0
    
    # 自由度の計算
    df = np.nan
    if equal_var:
        df = n1 + n2 - 2
    else:
        if n1 > 1 and n2 > 1:
            v1 = var1 / n1
            v2 = var2 / n2
            denom_df = (v1**2 / (n1 - 1)) + (v2**2 / (n2 - 1))
            df = (v1 + v2)**2 / denom_df if denom_df > 0 else (n1 + n2 - 2)
        else:
            df = n1 + n2 - 2
    
    # 効果量 r の計算
    t_stat = t_statistic_val
    effect_size_r = np.nan
    if np.isfinite(t_stat) and np.isfinite(df):
        denominator_sq = (t_stat**2) + df
        if denominator_sq > 0:
            denominator = np.sqrt(denominator_sq)
            effect_size_r = t_stat / denominator
        elif t_stat == 0:
            effect_size_r = 0.0
    
    return {
        "test_name": "対応のないt検定",
        "stat_name": "t値",
        "stat": t_statistic_val,
        "p_value": p_value,
        "df": df,
        "effect_size_cohen_d": cohen_d,
        "effect_size_r": effect_size_r,
        "shapiro1": shapiro1,
        "shapiro2": shapiro2,
        "normality": bool(is_normal),
        "levene": {"stat": levene_stat, "p": levene_p},
        "equal_var": bool(equal_var),
        "message": message
    }

