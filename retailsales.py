import pandas as pd
import streamlit as st
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# # page_configuration:
# icon = Image.open("machine-learning_logo.png")
st.set_page_config(page_title="Retail sales Forecast",
                   page_icon=':star:',
                   layout='centered',
                   )
st.markdown(f'<h1 style="text-align: center;">Retail sales Forecast</h1>', unsafe_allow_html=True)
# #dataset:
store = pd.read_csv("stores_data_set.csv")
feature = pd.read_csv("Features_data_set.csv")
sales = pd.read_csv("sales_data_set.csv")

enc = OrdinalEncoder()

# concat:
df = pd.merge(sales, feature, how='left', on=['Store', 'Date', 'IsHoliday'])
df = pd.merge(df, store, how='left', on=['Store'])
df.fillna(0)
# df['IsReturn']=((df['Weekly_Sales']<0))
df['IsHoliday'] = enc.fit_transform(df[['IsHoliday']])
df["Type"] = enc.fit_transform(df[["Type"]])
df = df.drop(['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5'], axis=1)
df[["day", "month", "year"]] = df["Date"].str.split("/", expand=True)
# to change df info as int,datetime:
df.drop(columns=df.columns[2], axis=1, inplace=True)
df["day"] = df["day"].astype(str).astype(int)
df["month"] = df["month"].astype(str).astype(int)
df["year"] = df["year"].astype(str).astype(int)

# random regressor:
x = np.array(df.drop(['Weekly_Sales', 'Store', 'Dept'], axis=1))
y = np.array(df['Weekly_Sales'])
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
reg = RandomForestRegressor()
model = reg.fit(x_train, y_train)
y_pred = reg.predict(x_test)
y_train_pred = reg.predict(x_train)
print("RandomTreeRegressor_score :", reg.score(x_test, y_test))
print("Mean square error:%.2f", np.mean((y_test-y_pred)**2))


year_week_dept = df.groupby(['Dept', 'year'])["Weekly_Sales"].sum()
year_sale = year_week_dept.unstack()

col1, col2 = st.columns(2, gap='large')
with col1:
    year = st.selectbox(label='Year', options=['2010', '2011', '2012'])
    dept = st.selectbox(label='Dept', options=df['Dept'])
    temperature = st.selectbox(label='Temperature', options=df["Temperature"])
    size = st.selectbox(label='Size', options=df['Size'])
with col2:
    Type = st.selectbox(label='Type', options=['A', 'B', 'C'])
    Type_dict = {'A': '0.0', 'B': '1.0', 'C': '2.0'}
    store = st.selectbox(label='Store', options=df["Store"])
    cpi = st.selectbox(label='CPI', options=df["CPI"])
    fuel_price = st.selectbox(label='Fuel_price', options=df['Fuel_Price'])

st.write('')
submit = st.button(label="Submit")
st.write('')

if submit:
    user_data = np.array([year, dept, Type_dict[Type], store, temperature, cpi, size, fuel_price])
    test_result = model.predict(user_data)
    if test_result == 0:
        st.success('Result: is_not_holiday')
        st.balloons()
    else:
        st.error('Result: is_holiday')
