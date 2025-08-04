import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import requests
from io import BytesIO
import plotly.express as px
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


st.set_page_config(page_title="📦 Dự báo đặt hàng kho", layout="wide")
st.title("📦 Dự báo đặt hàng kho theo doanh số 6 tháng gần nhất")

# Đường dẫn file Excel trên GitHub (raw link)
excel_url = "data_product.xlsx"

# Đọc file Excel
df = pd.ExcelFile("data_product.xlsx")

df_sales = df.parse("sales")
df_tonkho = df.parse("tonkho")


if df_sales is not None and df_tonkho is not None:
    df_sales['date'] = pd.to_datetime(df_sales['date'], errors='coerce')
    latest_date = df_sales["date"].max()
    six_months_ago = latest_date - pd.DateOffset(months=6)
    df_recent_sales = df_sales[df_sales["date"] >= six_months_ago]

    sales_summary = df_recent_sales.groupby("spcode")["numsell"].sum().reset_index()
    sales_summary.columns = ["spcode", "total_6m_sales"]
    sales_summary["monthly_avg"] = (sales_summary["total_6m_sales"] / 6).round()
    sales_summary["forecast_qty"] = (sales_summary["monthly_avg"] * 3.5).round()

    df_result = pd.merge(sales_summary, df_tonkho[["spcode", "tonkho", "hangve","conlai"]], on="spcode", how="left")
    df_result["tonkho"] = df_result["tonkho"].fillna(0)
    df_result["hangve"] = df_result["hangve"].fillna(0)
    df_result["conlai"] = df_result["conlai"].fillna(0)
    df_result["available"] = df_result["tonkho"] + df_result["hangve"]
    df_result["need_order"] = df_result["available"] < df_result["forecast_qty"]

    df_need_order = df_result[df_result["need_order"] == True]

    st.subheader("📋 Danh sách sản phẩm")
    st.dataframe(df_result)

    # Combobox chọn mã sản phẩm
    st.subheader("📈 Lịch sử bán theo tháng")
    selected_spcode = st.selectbox("Chọn mã sản phẩm để xem biểu đồ:", df_result['spcode'].unique())   
    
    df_info = df_result[df_result["spcode"] == selected_spcode]
    
    # Hiển thị bảng thông tin sản phẩm đã chọn
    st.write("📌 **Thông tin sản phẩm đã chọn:**")
    st.dataframe(df_info)
    
    # Lọc và nhóm dữ liệu theo tháng cho sản phẩm được chọn
    df_selected = df_sales[df_sales['spcode'] == selected_spcode].copy()
    df_selected['month'] = df_selected['date'].dt.to_period('M').dt.to_timestamp()
    monthly_sales = df_selected.groupby('month')['numsell'].sum().reset_index()
    
    # Hiển thị biểu đồ lịch sử
    fig = px.line(
        monthly_sales,
        x="month",
        y="numsell",
        markers=True,
        title=f"Lịch sử bán hàng theo tháng - {selected_spcode}",
        labels={"month": "Tháng", "numsell": "Số lượng bán"},
        hover_data={"month": True, "numsell": True}
    )
    fig.update_traces(hovertemplate='Tháng: %{x}<br>Số lượng: %{y}')
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)
    
    # Chỉ chạy LSTM nếu có đủ dữ liệu (> 10 tháng)
    if len(monthly_sales) > 10:
        # Chuẩn hóa
        data = monthly_sales[['numsell']].values.astype('float32')
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_scaled = scaler.fit_transform(data)
    
        # Tạo chuỗi thời gian dạng sliding window
        def create_dataset(dataset, look_back=6, predict_forward=3):
            X, y = [], []
            for i in range(len(dataset) - look_back - predict_forward + 1):
                X.append(dataset[i:i + look_back, 0])
                y.append(dataset[i + look_back:i + look_back + predict_forward, 0])
            return np.array(X), np.array(y)
    
        look_back = 6
        predict_forward = 3
        X, y = create_dataset(data_scaled, look_back, predict_forward)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
        # Tạo và huấn luyện mô hình LSTM
        model = Sequential()
        model.add(LSTM(50, input_shape=(look_back, 1)))
        model.add(Dense(predict_forward))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(X, y, epochs=100, batch_size=8, verbose=0)
    
        # Dự báo 3 tháng tiếp theo
        last_input = data_scaled[-look_back:]
        last_input = np.reshape(last_input, (1, look_back, 1))
        prediction_scaled = model.predict(last_input)
        prediction = scaler.inverse_transform(prediction_scaled).flatten()
    
        # Hiển thị kết quả dự báo
        last_date = monthly_sales['date'].max()
        future_months = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=3, freq='MS')
        df_forecast = pd.DataFrame({'Tháng': future_months.strftime('%Y-%m'), 'Dự báo số lượng bán': prediction.astype(int)})
    
        st.subheader("🔮 Dự báo số lượng bán 3 tháng tiếp theo")
        st.dataframe(df_forecast)
    
    else:
        st.warning("Không đủ dữ liệu để dự báo với LSTM (cần > 10 tháng).")

else:
    st.warning("Dữ liệu chưa được tải.")












