import streamlit as st
import numpy as np
import pandas as pd
import pickle

rfr=pickle.load(open('rfr.pkl','rb'))
X_train=pd.read_csv('X_train.csv')

st.title("Calories burnt prediction")

Gender=st.selectbox('Gender',X_train['Gender'])
Age=st.selectbox('Age',X_train['Age'])
Height=st.selectbox('Height',X_train['Height'])
Weight=st.selectbox('Weight',X_train['Weight'])
Duration=st.selectbox('Duration',X_train['Duration'])
Heart_rate=st.selectbox('Heart rate',X_train['Heart_Rate'])
Body_temp=st.selectbox('Body temp',X_train['Body_Temp'])

def pred(Gender,Age,Height,Weight,Duration,Heart_rate,Body_temp):
    features=np.array([[Gender,Age,Height,Weight,Duration,Heart_rate,Body_temp]])
    prediction=rfr.predict(features).reshape(1,-1)
    return prediction[0]
result=pred(Gender,Age,Height,Weight,Duration,Heart_rate,Body_temp)

if st.button('predict'):
    if result:
        st.write("Amount of calorie burnt:",result)

