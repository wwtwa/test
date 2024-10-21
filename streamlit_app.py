import streamlit as st
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import japanize_matplotlib  # 日本語表示のため
from janome.tokenizer import Tokenizer
import os
from dotenv import load_dotenv
from notion_client import Client
import plotly.express as px
import numpy as np

# 環境変数のロード
load_dotenv()
NOTION_API_KEY = os.environ.get("NOTION_API_KEY")
DATABASE_ID = os.environ.get("DATABASE_ID")

# Notion APIクライアントを初期化
notion = Client(auth=NOTION_API_KEY)

# クエリを作成してデータを取得、データフレームに変換
query = notion.databases.query(database_id=DATABASE_ID)
df = pd.DataFrame(query["results"])

# データ加工用の関数定義
def extract_property_value(row, property_name, sub_property=None):
   prop = row["properties"].get(property_name, {})
   if sub_property and isinstance(prop, dict):
       return prop.get(sub_property, {}).get("name", None)
   return None

# Plotlyのダークテーマを設定
px.defaults.template = "plotly_dark"

# アプリケーションのタイトル設定とスタイル調整
st.title("Notion Reading List Visualization")
st.markdown(
   """
   <style>
   .big-font {
       font-size:20px !important;
   }
   </style>
   """,
   unsafe_allow_html=True,
)

# 2行2列のグリッドレイアウトを作成
grid = [[None, None], [None, None]]
font_path = 'fonts/ヒラギノ明朝 ProN.ttc'  # フォントパスを指定


# ステータス別の円グラフ
status_counts = df.apply(lambda row: extract_property_value(row, "ステータス", "status"), axis=1).value_counts()
fig = px.pie(values=status_counts.values, names=status_counts.index, title="<b>ステータス別分布</b>")
grid[0][1] = fig

# 大分類別の分布
category_counts = df.apply(lambda row: ', '.join([x["name"] for x in row["properties"].get("大分類", {}).get("multi_select", []) if x]), axis=1).value_counts()
fig = px.bar(x=category_counts.index, y=category_counts.values, labels={'x': "<b>大分類</b>", 'y': "<b>件数</b>"}, title="<b>大分類別の件数</b>")
grid[1][0] = fig

# 小分類別の分布
subcategory_counts = df.apply(lambda row: ', '.join([x["name"] for x in row["properties"].get("小分類", {}).get("multi_select", []) if x]), axis=1).value_counts()
fig = px.bar(x=subcategory_counts.index, y=subcategory_counts.values, labels={'x': "<b>小分類</b>", 'y': "<b>件数</b>"}, title="<b>小分類別の件数</b>")
grid[1][1] = fig

x = list(range(50))
y = np.random.randn(50)

plt.show()
plt.plot(x, y)
plt.title('Matplotlib and Streamlit')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# Streamlitで表示
st.pyplot(plt)
