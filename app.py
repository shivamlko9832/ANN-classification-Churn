import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pickle

#Load the trained model
model  = tf.keras.models.load_model('churn_model.h5')

#Load the encoder and Scaler
with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender = pickle.load(file)
    
with open('onehot_encoder_geo.pkl','rb') as file:
    onehot_encoder_geo = pickle.load(file)
    
with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)
    
    
## Streamlit App
st.title('Customer Churn Prediction')

#User Input
geography = st.selectbox('Geography',onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age',18,100,30)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure= st.slider('Tenure',0,10,5)
num_of_products = st.slider('Number of Products',1,4,1) 
has_credit_card = st.selectbox('Has Credit Card',[0,1])
is_active_member = st.selectbox('Is Active Member',[0,1])


#Prepare the input data
input_data = pd.DataFrame({
    'CreditScore':[credit_score],
    'Gender':[label_encoder_gender.transform([gender])[0]],
    'Age':[age],
    'Tenure':[tenure],
    'Balance':[balance],
    'NumOfProducts':[num_of_products],
    'HasCrCard':[has_credit_card],
    'IsActiveMember':[is_active_member],
    'EstimatedSalary':[estimated_salary],
})

##One hot encode 'Geography'
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded,columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

#Concatenate the input data and the encoded geography
input_data = pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)

#Scale the input data
input_data_scaled = scaler.transform(input_data)

#Predict the churn probability
prediction = model.predict(input_data_scaled)[0][0]
st.write('Churn Probability:',prediction)


if prediction >= 0.5:
    st.write('The customer is likely to churn')
else:
    st.write('The customer is not likely to churn')


