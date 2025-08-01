import streamlit as st
import pandas as pd
import numpy as np
import random
import tensorflow as tf
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import plotly.graph_objs as go

# Đặt seed để tái lập kết quả


def set_seed(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


set_seed(42)

st.set_page_config(page_title="LSTM Dự báo sản phẩm", layout="wide")
st.title("📦 Dự báo số lượng bán theo mã sản phẩm (LSTM)")

df = pd.read_excel("data_product.xlsx")
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df.dropna(subset=['date', 'numsell', 'code'], inplace=True)

st.subheader("📋 Xem trước dữ liệu")
st.dataframe(df.head())

product_codes = sorted(df['code'].unique())
selected_code = st.selectbox("🔍 Chọn mã sản phẩm", product_codes)

forecast_months = st.slider(
    "🔮 Số tháng cần dự báo", min_value=1, max_value=12, value=3)

df_product = df[df['code'] == selected_code]
df_monthly = df_product.resample('ME', on='date')[
    'numsell'].sum().fillna(0).reset_index()

# Thêm đặc trưng thời gian
df_monthly['month'] = df_monthly['date'].dt.month
df_monthly['quarter'] = df_monthly['date'].dt.quarter
df_monthly['year'] = df_monthly['date'].dt.year
df_monthly['days_in_month'] = df_monthly['date'].dt.days_in_month
df_monthly['is_holiday'] = df_monthly['month'].isin(
    [1, 2, 4, 9, 12]).astype(int)

st.subheader("📋 Dữ liệu sau khi xử lý")
st.dataframe(df_monthly.head())

# Biểu đồ lịch sử bán hàng
st.subheader("📈 Biểu đồ lịch sử bán hàng theo tháng")
fig_hist = go.Figure()
fig_hist.add_trace(go.Scatter(
    x=df_monthly['date'],
    y=df_monthly['numsell'],
    mode='lines+markers',
    name='Số lượng bán',
    hovertemplate='Tháng: %{x|%b %Y}<br>Số lượng: %{y}<extra></extra>'
))
fig_hist.update_layout(title=f"Lịch sử bán hàng của mã: {selected_code}",
                       xaxis_title="Thời gian",
                       yaxis_title="Số lượng bán")
st.plotly_chart(fig_hist, use_container_width=True)

if len(df_monthly) < 10:
    st.warning("⚠️ Không đủ dữ liệu để huấn luyện (cần ≥ 10 tháng).")
else:
    features = ['numsell', 'month', 'quarter',
                'year', 'days_in_month', 'is_holiday']
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df_monthly[features])

    def create_sequences(data, seq_len):
        X, y = [], []
        for i in range(len(data) - seq_len):
            X.append(data[i:i+seq_len])
            y.append(data[i+seq_len][0])
        return np.array(X), np.array(y)

    seq_len = 6
    X, y = create_sequences(scaled, seq_len)
    X_train, X_val = X[:-forecast_months], X[-forecast_months:]
    y_train, y_val = y[:-forecast_months], y[-forecast_months:]

    model = Sequential([
        LSTM(64, return_sequences=True, activation='relu',
             input_shape=(X.shape[1], X.shape[2])),
        Dropout(0.2),
        LSTM(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=200, batch_size=8,
              validation_data=(X_val, y_val), verbose=0)

    last_seq = scaled[-seq_len:]
    future_forecasts = []
    future_dates = []

    last_date = df_monthly['date'].max()

    for i in range(forecast_months):
        forecast_date = last_date + pd.DateOffset(months=i+1)
        month = forecast_date.month
        quarter = (month - 1) // 3 + 1
        year = forecast_date.year
        days_in_month = pd.Period(forecast_date, freq='M').days_in_month
        is_holiday = int(month in [1, 2, 4, 9, 12])

        input_scaled = scaler.transform(
            [[0, month, quarter, year, days_in_month, is_holiday]])
        input_time_features = input_scaled[0][1:]

        input_seq = last_seq.reshape(1, seq_len, X.shape[2])
        pred_scaled = model.predict(input_seq)
        pred_value = scaler.inverse_transform(
            np.hstack([pred_scaled, np.zeros((1, len(features)-1))])
        )[0][0]

        future_forecasts.append(pred_value)
        future_dates.append(forecast_date)

        next_scaled = np.concatenate(
            [[pred_scaled[0][0]], input_time_features])
        last_seq = np.vstack([last_seq[1:], next_scaled])

    y_pred = model.predict(X_val)
    real_dates = df_monthly['date'][-forecast_months:]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=real_dates,
                             y=scaler.inverse_transform(np.hstack(
                                 [y_val.reshape(-1, 1), np.zeros((len(y_val), len(features)-1))]))[:, 0],
                             mode='lines+markers', name='Giá trị thực'))

    fig.add_trace(go.Scatter(x=real_dates,
                             y=scaler.inverse_transform(
                                 np.hstack([y_pred, np.zeros((len(y_pred), len(features)-1))]))[:, 0],
                             mode='lines+markers', name='Dự báo'))

    fig.add_trace(go.Scatter(x=future_dates, y=future_forecasts,
                             mode='lines+markers', name='Dự báo tương lai',
                             line=dict(dash='dot', color='green')))

    fig.update_layout(title="📊 So sánh giá trị thực và dự báo",
                      xaxis_title="Thời gian", yaxis_title="Số lượng")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📅 Kết quả dự báo")
    forecast_table = pd.DataFrame({
        'Tháng': future_dates,
        'Dự báo số lượng': future_forecasts,
        'Tồn kho đề xuất (+20%)': [round(x * 1.2) for x in future_forecasts]
    })
    st.dataframe(forecast_table)

    # Phân tích top sản phẩm dự báo cao nhất
    st.subheader("🚀 Gợi ý sản phẩm có nhu cầu cao")
    df_all_monthly = df.copy()
    df_all_monthly['month'] = df_all_monthly['date'].dt.to_period(
        'M').dt.to_timestamp('M')
    top_agg = df_all_monthly.groupby(['month', 'code'])[
        'numsell'].sum().reset_index()
    top_codes = top_agg.groupby(
        'code')['numsell'].sum().nlargest(30).index.tolist()

    future_top_preds = []
    for code in top_codes:
        sub_df = df[df['code'] == code]
        sub_monthly = sub_df.resample('ME', on='date')[
            'numsell'].sum().fillna(0).reset_index()
        if len(sub_monthly) >= seq_len + forecast_months:
            sub_monthly['month'] = sub_monthly['date'].dt.month
            sub_monthly['quarter'] = sub_monthly['date'].dt.quarter
            sub_monthly['year'] = sub_monthly['date'].dt.year
            sub_monthly['days_in_month'] = sub_monthly['date'].dt.days_in_month
            sub_monthly['is_holiday'] = sub_monthly['month'].isin(
                [1, 2]).astype(int)

            sc = MinMaxScaler()
            sub_scaled = sc.fit_transform(sub_monthly[features])

            if len(sub_scaled) >= seq_len:
                last_seq = sub_scaled[-seq_len:]
                forecast_date = sub_monthly['date'].max(
                ) + pd.DateOffset(months=1)
                m, q, y = forecast_date.month, forecast_date.quarter, forecast_date.year
                d = pd.Period(forecast_date, freq='M').days_in_month
                h = int(m in [1, 2])
                time_input = sc.transform([[0, m, q, y, d, h]])[0][1:]

                input_seq = last_seq.reshape(1, seq_len, len(features))
                pred_scaled = model.predict(input_seq)
                pred_value = sc.inverse_transform(
                    np.hstack(
                        [pred_scaled, np.zeros((1, len(features)-1))])
                )[0][0]

                future_top_preds.append((code, round(pred_value)))

    top_demand_df = pd.DataFrame(future_top_preds, columns=[
        'Mã sản phẩm', 'Dự báo tháng tới'])
    top_demand_df = top_demand_df.sort_values(
        by='Dự báo tháng tới', ascending=False)
    st.dataframe(top_demand_df)
