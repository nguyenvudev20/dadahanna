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

    df_result = pd.merge(sales_summary, df_tonkho[["spcode", "tonkho", "hangve"]], on="spcode", how="left")
    df_result["tonkho"] = df_result["tonkho"].fillna(0)
    df_result["hangve"] = df_result["hangve"].fillna(0)
    df_result["available"] = df_result["tonkho"] + df_result["hangve"]
    df_result["need_order"] = df_result["available"] < df_result["forecast_qty"]

    df_need_order = df_result[df_result["need_order"] == True]

    st.subheader("📋 Danh sách sản phẩm cần đặt hàng")
    st.dataframe(df_need_order)

    top20 = df_need_order.sort_values(by="forecast_qty", ascending=False).head(20)

    if not top20.empty:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(top20["spcode"], top20["forecast_qty"], label="Forecast Qty (3.5 tháng)")
        ax.bar(top20["spcode"], top20["available"], label="Tồn kho + Hàng về")
        ax.set_title("Top 20 sản phẩm cần đặt hàng")
        ax.set_ylabel("Số lượng")
        ax.set_xticklabels(top20["spcode"], rotation=90)
        ax.legend()
        st.pyplot(fig)
else:
    st.warning("Dữ liệu chưa được tải.")

