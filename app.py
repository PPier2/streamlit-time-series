
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import json
import random
import pathlib
from io import BytesIO

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

# 👉 Colab 需要 nest_asyncio 來解決事件迴圈衝突
try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass

# 1. 版面與環境設定

st.set_page_config(page_title="XGBoost Time‑Series Trainer", layout="wide")
st.title("📈 XGBoost Time‑Series Trainer")

SEED_DEFAULT = 42

# 2. Feature‑engineering 函式

def create_features(data: pd.DataFrame, dynamic: bool = True):
    d = data.copy()

    # 時間週期特徵
    d["year"] = d["Date"].dt.year
    d["quarter"] = d["Date"].dt.quarter
    d["quarter_sin"] = np.sin(2 * np.pi * d["quarter"] / 4)
    d["quarter_cos"] = np.cos(2 * np.pi * d["quarter"] / 4)
    d["month"] = d["Date"].dt.month
    d["month_sin"] = np.sin(2 * np.pi * d["month"] / 12)
    d["month_cos"] = np.cos(2 * np.pi * d["month"] / 12)
    d["weekday"] = d["Date"].dt.weekday
    d["weekday_sin"] = np.sin(2 * np.pi * d["weekday"] / 7)
    d["weekday_cos"] = np.cos(2 * np.pi * d["weekday"] / 7)

    if dynamic:
        for k in [1, 5, 7, 20, 60]:
            d[f"lag{k}"] = d["B"].shift(k)

        for k in [5, 20, 60]:
            d[f"roll_mean{k}"] = d["B"].rolling(k).mean().shift(1)
            d[f"roll_std{k}"] = d["B"].rolling(k).std().shift(1)
            d[f"slope_MA{k}"] = d[f"roll_mean{k}"].diff()

        d["week_month"] = (d["roll_mean5"] - d["roll_mean20"])
        d["month_quarter"] = (d["roll_mean20"] - d["roll_mean60"])

    return d


def add_ema_macd(df: pd.DataFrame):
    d = df.copy()
    d["EMA12"] = d["B"].ewm(span=12, adjust=False).mean().shift(1)
    d["EMA26"] = d["B"].ewm(span=26, adjust=False).mean().shift(1)
    d["MACD"] = (d["EMA12"] - d["EMA26"])
    d["MACD_signal"] = d["MACD"].ewm(span=9, adjust=False).mean()
    d["MACD_diff"] = (d["MACD"] - d["MACD_signal"])
    return d

# --------------------------------------------------------------
# 3. 其他工具
# --------------------------------------------------------------

feature_cols = [
    "year", "quarter_sin", "quarter_cos",
    "month_sin", "month_cos",
    "weekday_sin", "weekday_cos",
    "lag1", "lag5", "lag7", "lag20", "lag60",
    "roll_mean5", "roll_mean20", "roll_mean60",
    "slope_MA5", "slope_MA20", "slope_MA60",
    "roll_std5", "roll_std20", "roll_std60",
    "week_month", "month_quarter",
    "EMA12", "EMA26", "MACD", "MACD_signal", "MACD_diff",
]

param_dist_default = {
    "n_estimators": list(range(100, 1001, 10)),
    "learning_rate": list(np.linspace(0.005, 0.05, 10)),
    "max_depth": list(range(2, 8)),
    "subsample": list(np.linspace(0.6, 1.0, 5)),
    "colsample_bytree": list(np.linspace(0.6, 1.0, 5)),
    "reg_alpha": list(np.linspace(0, 1.0, 6)),
    "reg_lambda": list(np.linspace(0.5, 3.0, 6)),
}


def sample_param(space: dict):
    return {k: random.choice(v) for k, v in space.items()}

# --------------------------------------------------------------
# 4. Iterative validation
# --------------------------------------------------------------

def iterative_validate(model, val_static: pd.DataFrame, train_hist: list):
    preds, hr_hist = [], train_hist.copy()
    for _, row in val_static.iterrows():
        lag1 = hr_hist[-1]
        lag5 = hr_hist[-5] if len(hr_hist) >= 5 else hr_hist[0]
        lag7 = hr_hist[-7] if len(hr_hist) >= 7 else hr_hist[0]
        lag20 = hr_hist[-20] if len(hr_hist) >= 20 else hr_hist[0]
        lag60 = hr_hist[-60] if len(hr_hist) >= 60 else hr_hist[0]

        win5 = hr_hist[-6:-1] if len(hr_hist) >= 5 else hr_hist
        roll_mean5 = float(np.mean(win5))
        roll_std5 = float(np.std(win5))
        win20 = hr_hist[-21:-1] if len(hr_hist) >= 20 else hr_hist
        roll_mean20 = float(np.mean(win20))
        roll_std20 = float(np.std(win20))
        win60 = hr_hist[-61:-1] if len(hr_hist) >= 60 else hr_hist
        roll_mean60 = float(np.mean(win60))
        roll_std60 = float(np.std(win60))

        hr_ser = pd.Series(hr_hist)
        ema12 = hr_ser.ewm(span=12, adjust=False).mean().iloc[-1]
        ema26 = hr_ser.ewm(span=26, adjust=False).mean().iloc[-1]
        macd = ema12 - ema26
        macd_sig = (
            hr_ser.ewm(span=12, adjust=False).mean() - hr_ser.ewm(span=26, adjust=False).mean()
        ).ewm(span=9, adjust=False).mean().iloc[-1]
        macd_diff = macd - macd_sig

        week_month = roll_mean5 - roll_mean20
        month_quarter = roll_mean20 - roll_mean60

        ma5 = hr_ser.rolling(5).mean().shift(1)
        ma20 = hr_ser.rolling(20).mean().shift(1)
        ma60 = hr_ser.rolling(60).mean().shift(1)
        slope_MA5 = ma5.diff().iloc[-1] if len(ma5.dropna()) >= 2 else 0.0
        slope_MA20 = ma20.diff().iloc[-1] if len(ma20.dropna()) >= 2 else 0.0
        slope_MA60 = ma60.diff().iloc[-1] if len(ma60.dropna()) >= 2 else 0.0

        base_vec = [
            row["year"], row["quarter_sin"], row["quarter_cos"],
            row["month_sin"], row["month_cos"],
            row["weekday_sin"], row["weekday_cos"],
            lag1, lag5, lag7, lag20, lag60,
            roll_mean5, roll_mean20, roll_mean60,
            slope_MA5, slope_MA20, slope_MA60,
            roll_std5, roll_std20, roll_std60,
            week_month, month_quarter,
            ema12, ema26, macd, macd_sig, macd_diff,
        ]

        pred = model.predict(np.array(base_vec).reshape(1, -1))[0]
        preds.append(pred)
        hr_hist.append(pred)

    return preds, r2_score(val_static["B"], preds)

# --------------------------------------------------------------
# --------------------------------------------------------------
# 5. 介面
# --------------------------------------------------------------

uploaded_file = st.file_uploader("上傳含有 'Date' 與 'B' 欄位的 Excel 檔", type=["xlsx", "xls"])
if uploaded_file is None:
    st.info("👆 請先上傳檔案。")
    st.stop()

raw_df = (
    pd.read_excel(uploaded_file, parse_dates=["Date"])
    .sort_values("Date")
    .reset_index(drop=True)
)

st.success(f"✅ 已讀取 {len(raw_df)} 筆資料。")

col1, col2, col3 = st.columns(3)
with col1:
    split_ratio = st.slider("訓練集比例 (%)", 50, 90, 80, step=5)
with col2:
    N_ITER = st.number_input("隨機搜尋迭代次數", 10, 2000, 200, step=10)
with col3:
    seed = st.number_input("Random Seed", 0, 10000, SEED_DEFAULT, step=1)

param_json = st.text_area(
    "可選：自訂 param_dist (JSON)", value="", height=140
)
param_dist = json.loads(param_json) if param_json.strip() else param_dist_default

start_training = st.button("🚀 開始訓練")

# --------------------------------------------------------------
# 6. 訓練
# --------------------------------------------------------------

if start_training:
    random.seed(seed)
    np.random.seed(seed)

    split_idx = int(len(raw_df) * split_ratio / 100)
    train_df = raw_df.iloc[:split_idx].reset_index(drop=True)
    val_df = raw_df.iloc[split_idx:].reset_index(drop=True)

    train_feat = (
        create_features(train_df, dynamic=True).pipe(add_ema_macd).dropna().reset_index(drop=True)
    )
    val_feat = create_features(val_df, dynamic=False).reset_index(drop=True)

    X_train, y_train = train_feat[feature_cols], train_feat["B"]

    progress = st.progress(0.0, text="等待開始…")
    best_r2, best_params, best_model, best_preds = -np.inf, None, None, None

    for i in range(int(N_ITER)):
        params = sample_param(param_dist)
        model = XGBRegressor(
            **params,
            objective="reg:squarederror",
            random_state=seed,
            tree_method="hist",
        )
        model.fit(X_train, y_train)
        preds, r2 = iterative_validate(model, val_feat, train_feat["B"].tolist())
        if r2 > best_r2:
            best_r2, best_params, best_model, best_preds = r2, params, model, preds
        progress.progress((i + 1) / N_ITER, text=f"迭代 {i+1}/{N_ITER} ‒ R²={r2:.4f}")

    st.subheader("最佳結果 (Validation)")
    st.metric("R²", f"{best_r2:.4f}")
    st.json(best_params)

    pathlib.Path("models").mkdir(exist_ok=True)
    joblib.dump(best_model, "models/xgb_best.pkl")

    out = val_df[["Date", "B"]].copy()
    out["B_pred"] = best_preds
    out.to_excel("models/valid_best_preds.xlsx", index=False)

    st.line_chart(out.set_index("Date"))

    with open("models/xgb_best.pkl", "rb") as f:
        st.download_button("下載模型 .pkl", f, file_name="xgb_best.pkl")
    with open("models/valid_best_preds.xlsx", "rb") as f:
        st.download_button("下載預測 .xlsx", f, file_name="valid_best_preds.xlsx")

    st.success("🎉 訓練完成！下載檔案或調整參數重新訓練。")

# ⬆⬆⬆ 直至程式結尾 ⬆⬆⬆

# 2) 建 requirements.txt


# 3) 初始化 Git


# 4) 在 GitHub 建立遠端 repo（透過 API）


# 5) 推送
