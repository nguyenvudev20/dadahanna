import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import requests
from io import BytesIO

st.set_page_config(page_title="📦 Dự báo đặt hàng kho", layout="wide")
st.title("📦 Dự báo đặt hàng kho theo doanh số 6 tháng gần nhất")

# Đường dẫn file Excel trên GitHub (raw link)
excel_url = "https://raw.githubusercontent.com/nguyenvudev20/dadahanna/main/data_product.xlsx"

@st.cache_data
def load_data(url):
    response = requests.get(url)
    if response.status_code != 200:
        st.error("Không thể tải file từ GitHub.")
        return None, None
    xls = pd.ExcelFile(BytesIO(response.content))
    return xls.parse("sales"), xls.parse("tonkho")

df_sales, df_tonkho = load_data(excel_url)

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
    df_result["hangve"] =_
