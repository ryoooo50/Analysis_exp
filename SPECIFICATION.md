# リーチング分析・統計アプリ 仕様書

## 1. 概要

本アプリケーションは、リーチング動作の解析と統計検定を行うWebアプリケーションです。CSVファイルから角速度データを読み込み、ピーク検出を行う「ピーク分析」機能と、2群間の統計検定を行う「統計検定」機能を提供します。

### 主な機能

1. **ピーク分析**: CSVファイルから角速度データを読み込み、ピークを自動検出・分析
2. **マン・ホイットニーU検定**: 対応のない2群間のノンパラメトリック検定
3. **ウィルコクソン符号順位検定**: 対応のある2群間のノンパラメトリック検定
4. **t検定**: 対応のない2群間のパラメトリック検定（正規性・等分散性の検定も含む）

## 2. 技術スタック

### バックエンド
- **Python**: 3.x
- **Flask**: 3.1.2 - Webフレームワーク
- **NumPy**: 2.3.3 - 数値計算
- **Pandas**: 2.3.3 - データ処理
- **SciPy**: 1.16.2 - 統計検定・信号処理
- **Matplotlib**: 3.10.6 - グラフ描画
- **Seaborn**: 0.13.2 - 統計的可視化

### フロントエンド
- **HTML5/CSS3/JavaScript**: ネイティブ実装（フレームワーク不使用）

### デプロイメント
- **Gunicorn**: 23.0.0 - WSGIサーバー

## 3. アーキテクチャ

### ディレクトリ構造

```
Analysis_exp/
├── app.py                    # メインアプリケーション（Flaskルーティング）
├── config.py                 # 設定定数
├── requirements.txt          # 依存パッケージ一覧
├── runtime.txt              # Pythonランタイムバージョン指定
├── SPECIFICATION.md         # 本仕様書
├── utils/
│   ├── __init__.py
│   ├── data_processing.py   # データ処理ユーティリティ
│   ├── peak_analysis.py      # ピーク分析ロジック
│   ├── plotting.py          # プロット生成
│   └── statistical_tests.py # 統計検定関数
└── templates/
    └── index.html           # フロントエンドUI
```

### モジュール構成

#### `app.py`
- Flaskアプリケーションのメインファイル
- ルーティング定義
- エラーハンドリング
- リクエストパラメータの抽出と検証

#### `config.py`
- アプリケーション全体で使用する定数
- CSVファイルの列名定義
- デフォルトパラメータ値
- 統計検定の設定

#### `utils/data_processing.py`
- CSVファイルの読み込み（UTF-8/Shift-JIS対応）
- 角速度の計算
- ローパスフィルタの適用
- テキストデータのパース（カンマ区切り数値）

#### `utils/peak_analysis.py`
- ピーク検出ロジック
- ピーク情報の集計
- ピーク平均の計算（10回ごとのグループ化）

#### `utils/plotting.py`
- 角速度とピークの可視化
- Base64エンコードされた画像の生成

#### `utils/statistical_tests.py`
- マン・ホイットニーU検定の実装
- ウィルコクソン符号順位検定の実装
- t検定の実装（正規性・等分散性検定含む）
- 効果量の計算（Cohen's d, r値）

## 4. API仕様

### 4.1 エンドポイント一覧

| メソッド | エンドポイント | 説明 |
|---------|--------------|------|
| GET | `/` | メインページ（HTML）を返す |
| POST | `/analyze-peak` | ピーク分析を実行 |
| POST | `/analyze-mann-whitney` | マン・ホイットニーU検定を実行 |
| POST | `/analyze-wilcoxon` | ウィルコクソン検定を実行 |
| POST | `/analyze-ttest` | t検定を実行 |

### 4.2 ピーク分析エンドポイント

#### リクエスト
- **URL**: `/analyze-peak`
- **Method**: POST
- **Content-Type**: `multipart/form-data`

**パラメータ**:
| パラメータ名 | 型 | 必須 | デフォルト値 | 説明 |
|------------|----|-----|------------|------|
| `file` | File | 必須 | - | CSVファイル |
| `min_peak_height` | float | 任意 | 1.0 | ピークの最小高さ (rad/s) |
| `max_peak_height` | float | 任意 | 100.0 | ピークの最大高さ (rad/s) |
| `peak_prominence` | float | 任意 | 1.0 | ピークの顕著さ |
| `peak_distance` | int | 任意 | 50 | ピーク間の最小距離（データ点数） |

#### レスポンス
```json
{
  "image": "base64エンコードされた画像データ",
  "peaks": [
    {
      "peak_time_s": 0.123,
      "peak_velocity_rad_s": 5.678
    },
    ...
  ],
  "peak_count": 150,
  "peak_averages": [
    {
      "interval": "Peaks 1-10",
      "average_velocity": 4.567
    },
    ...
  ]
}
```

**エラーレスポンス**:
```json
{
  "error": "エラーメッセージ"
}
```
- ステータスコード: 400 (バリデーションエラー), 500 (サーバーエラー)

#### CSVファイル形式
- 必須列: `time`, `Lower_Arm_R/rotation_y`
- エンコーディング: UTF-8 または Shift-JIS
- `time`: 時間（秒）
- `Lower_Arm_R/rotation_y`: 角度（度）

### 4.3 マン・ホイットニーU検定エンドポイント

#### リクエスト
- **URL**: `/analyze-mann-whitney`
- **Method**: POST
- **Content-Type**: `application/x-www-form-urlencoded`

**パラメータ**:
| パラメータ名 | 型 | 必須 | デフォルト値 | 説明 |
|------------|----|-----|------------|------|
| `data1` | string | 必須 | - | Group 1のデータ（カンマ区切り） |
| `data2` | string | 必須 | - | Group 2のデータ（カンマ区切り） |
| `alternative` | string | 任意 | "two-sided" | 検定の種類: "two-sided", "less", "greater" |

#### レスポンス
```json
{
  "test_name": "マン・ホイットニーU検定",
  "stat_name": "U値",
  "stat": 123.45,
  "p_value": 0.02345,
  "z_approx": 2.345,
  "effect_size_r": 0.456
}
```

### 4.4 ウィルコクソン符号順位検定エンドポイント

#### リクエスト
- **URL**: `/analyze-wilcoxon`
- **Method**: POST
- **Content-Type**: `application/x-www-form-urlencoded`

**パラメータ**:
| パラメータ名 | 型 | 必須 | デフォルト値 | 説明 |
|------------|----|-----|------------|------|
| `data1` | string | 必須 | - | Group 1のデータ（カンマ区切り） |
| `data2` | string | 必須 | - | Group 2のデータ（カンマ区切り、Group 1と同じサイズ） |
| `alternative` | string | 任意 | "two-sided" | 検定の種類: "two-sided", "less", "greater" |

#### レスポンス
```json
{
  "test_name": "ウィルコクソン符号順位検定",
  "stat_name": "W値",
  "stat": 234.56,
  "p_value": 0.01234,
  "z_approx": 2.567,
  "effect_size_r": 0.567
}
```

**エラー**: Group 1とGroup 2のサイズが異なる場合、エラーが返されます。

### 4.5 t検定エンドポイント

#### リクエスト
- **URL**: `/analyze-ttest`
- **Method**: POST
- **Content-Type**: `application/x-www-form-urlencoded`

**パラメータ**:
| パラメータ名 | 型 | 必須 | デフォルト値 | 説明 |
|------------|----|-----|------------|------|
| `data1` | string | 必須 | - | Group 1のデータ（カンマ区切り） |
| `data2` | string | 必須 | - | Group 2のデータ（カンマ区切り） |
| `alternative` | string | 任意 | "two-sided" | 検定の種類: "two-sided", "less", "greater" |

#### レスポンス
```json
{
  "test_name": "対応のないt検定",
  "stat_name": "t値",
  "stat": 3.456,
  "p_value": 0.00123,
  "df": 28.5,
  "effect_size_cohen_d": 1.234,
  "effect_size_r": 0.567,
  "shapiro1": {
    "stat": 0.987,
    "p": 0.876
  },
  "shapiro2": {
    "stat": 0.965,
    "p": 0.543
  },
  "normality": true,
  "levene": {
    "stat": 1.234,
    "p": 0.567
  },
  "equal_var": true,
  "message": ""
}
```

**レスポンス項目の説明**:
- `df`: 自由度（Welchのt検定の場合は非整数値）
- `effect_size_cohen_d`: Cohen's d（効果量）
- `effect_size_r`: 効果量r（r = t / sqrt(t² + df)）
- `shapiro1`, `shapiro2`: 各群のShapiro-Wilk検定結果
- `normality`: 両群が正規分布に従うか（p > 0.05）
- `levene`: Levene検定の結果（等分散性の検定）
- `equal_var`: 等分散性を仮定するか（p > 0.05）
- `message`: 警告メッセージ（正規性がない場合など）

## 5. 設定項目

### `config.py`の設定値

| 設定名 | 型 | デフォルト値 | 説明 |
|-------|---|------------|------|
| `TIME_COL` | str | `'time'` | CSVファイルの時間列名 |
| `ANGLE_COL` | str | `'Lower_Arm_R/rotation_y'` | CSVファイルの角度列名 |
| `CUTOFF_FREQ` | float | `5.0` | ローパスフィルタのカットオフ周波数 (Hz) |
| `FILTER_ORDER` | int | `4` | ローパスフィルタの次数 |
| `DEFAULT_PEAK_PARAMS` | dict | 下記参照 | ピーク検出のデフォルトパラメータ |
| `DEFAULT_ALTERNATIVE` | str | `'two-sided'` | 統計検定のデフォルト仮説 |
| `SIGNIFICANCE_LEVEL` | float | `0.05` | 有意水準 |
| `PEAK_GROUP_SIZE` | int | `10` | ピーク平均のグループサイズ |

**DEFAULT_PEAK_PARAMS**:
```python
{
    'min_peak_height': 1.0,
    'max_peak_height': 100.0,
    'peak_prominence': 1.0,
    'peak_distance': 50,
}
```

## 6. データ処理フロー

### 6.1 ピーク分析フロー

1. CSVファイルの読み込み（UTF-8/Shift-JIS自動判定）
2. 列名の空白削除
3. 角度をラジアンに変換（`deg2rad`）
4. 角速度の計算（時間微分）
5. ローパスフィルタの適用（カットオフ周波数: 5.0 Hz）
6. ピーク検出（SciPy `find_peaks`）
7. ピーク情報の集計
8. 10回ごとのピーク平均計算
9. グラフの生成（角速度とピーク位置）
10. 結果をJSON形式で返却

### 6.2 統計検定フロー

1. カンマ区切りテキストのパース
2. データの検証（空でない、数値のみ）
3. 検定の実行（統計量、p値の計算）
4. 効果量の計算
   - ノンパラメトリック検定: Z値 → 効果量r
   - t検定: Cohen's d, 効果量r
5. 正規性・等分散性の検定（t検定のみ）
6. 結果をJSON形式で返却

## 7. エラーハンドリング

### エラー種別

1. **バリデーションエラー (400)**
   - ファイルが指定されていない
   - データが空
   - データ形式が不正

2. **処理エラー (500)**
   - CSVファイルに必要な列がない
   - 数値変換エラー
   - 統計検定の実行エラー
   - その他の予期しないエラー

### エラーレスポンス形式

すべてのエラーは以下の形式で返されます：
```json
{
  "error": "エラーメッセージ"
}
```

## 8. 使用例

### 8.1 ピーク分析の使用例

**CSVファイル形式**:
```csv
time,Lower_Arm_R/rotation_y
0.0,10.5
0.01,11.2
0.02,12.1
...
```

**リクエスト例**:
```bash
curl -X POST http://localhost:5000/analyze-peak \
  -F "file=@data.csv" \
  -F "min_peak_height=1.0" \
  -F "max_peak_height=100.0" \
  -F "peak_prominence=1.0" \
  -F "peak_distance=50"
```

### 8.2 統計検定の使用例

**リクエスト例（マン・ホイットニーU検定）**:
```bash
curl -X POST http://localhost:5000/analyze-mann-whitney \
  -d "data1=4.1,4.4,4.2,4.5,4.3" \
  -d "data2=3.5,3.4,3.6,3.3,3.7" \
  -d "alternative=two-sided"
```

**データ入力形式**:
- カンマ区切り（全角カンマも可）
- 空白は自動削除
- 数値のみ（小数点対応）

## 9. 依存関係

### 主要パッケージ

| パッケージ | バージョン | 用途 |
|----------|----------|------|
| Flask | 3.1.2 | Webフレームワーク |
| NumPy | 2.3.3 | 数値計算 |
| Pandas | 2.3.3 | データ処理 |
| SciPy | 1.16.2 | 統計検定・信号処理 |
| Matplotlib | 3.10.6 | グラフ描画 |
| Seaborn | 0.13.2 | 統計的可視化 |
| Gunicorn | 23.0.0 | WSGIサーバー |

### インストール方法

```bash
pip install -r requirements.txt
```

## 10. デプロイメント

### ローカル開発環境

```bash
python app.py
```

開発サーバーが `http://localhost:5000` で起動します。

### 本番環境（Gunicorn使用）

```bash
gunicorn -w 4 -b 0.0.0.0:8000 app:app
```

### 環境変数

現在は環境変数の設定は不要です。必要に応じて `config.py` を編集してください。

## 11. 制約事項

1. **CSVファイル形式**
   - 必須列: `time`, `Lower_Arm_R/rotation_y`
   - エンコーディング: UTF-8 または Shift-JIS
   - 列名の前後の空白は自動削除されます

2. **統計検定**
   - ウィルコクソン検定: 2群のデータ数は同じである必要があります
   - t検定: サンプルサイズが小さい場合、正規性の検定ができない場合があります（n < 3）

3. **ピーク検出**
   - サンプリング周波数は自動計算されます
   - ローパスフィルタは4次バターワースフィルタを使用

## 12. 今後の拡張予定

現在のバージョンでは以下の拡張が検討されています：

- [ ] 複数CSVファイルの一括処理
- [ ] 統計検定結果のCSVエクスポート
- [ ] ピーク分析結果の詳細レポート生成
- [ ] グラフのカスタマイズオプション
- [ ] ユーザー認証機能
- [ ] データベース連携（分析履歴の保存）

## 13. ライセンス

本アプリケーションのライセンス情報については、プロジェクトルートのライセンスファイルを参照してください。

## 14. 更新履歴

### Version 1.0.0 (2024)
- 初回リリース
- ピーク分析機能
- 統計検定機能（マン・ホイットニー、ウィルコクソン、t検定）
- モジュール化によるリファクタリング完了

---

**作成日**: 2024年
**最終更新日**: 2024年
**バージョン**: 1.0.0




