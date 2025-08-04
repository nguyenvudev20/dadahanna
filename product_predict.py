import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import requests
from io import BytesIO

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
    
    # Dữ liệu lịch sử của mã sản phẩm
    df_selected = df_sales[df_sales['spcode'] == selected_spcode].copy()
    df_selected['month'] = df_selected['date'].dt.to_period('M').astype(str)
    monthly_sales = df_selected.groupby('month')['numsell'].sum().reset_index()
    
    # Biểu đồ tương tác với tooltip
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


else:
    st.warning("Dữ liệu chưa được tải.")







