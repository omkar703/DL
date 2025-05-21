import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🔄",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        margin-top: 20px;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
    }
    </style>
    """, unsafe_allow_html=True)



## load the train model
model = tf.keras.models.load_model('model.h5')

## load the encoders and scalers
with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)

with open('geo_encoder.pkl', 'rb') as f:
    ohe_geo = pickle.load(f)

with open('scaler_pickle.pkl', 'rb') as f:
    scaler = pickle.load(f)

st.title('Customer Churn Prediction')



# Main title with emoji
st.title('🔄 Customer Churn Prediction System')
st.markdown('---')

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Customer Demographics")
    geographic = st.selectbox('📍 Geography', ohe_geo.categories_[0])
    gender = st.selectbox('👤 Gender', label_encoder_gender.classes_)
    age = st.number_input('🎂 Age', 
                                    min_value=18, max_value=100, value=40)
    
    credit_score = st.number_input('💳 Credit Score', min_value=300, max_value=850, value=600, 
                                 help="Customer's credit score (300-850)")

with col2:
    st.subheader("💼 Account Information")
    tenure = st.number_input('⏳ Tenure (years)', min_value=0, max_value=10, value=3)
    balance = st.number_input('💰 Balance ($)', min_value=0.0, max_value=250000.0, value=60000.0, 
                            format="%.2f")
    num_of_products = st.number_input('🏦 Number of Products', min_value=1, max_value=4, value=2)
    estimated_salary = st.number_input('💵 Estimated Salary ($)', 
                                    min_value=0.0, max_value=200000.0, value=50000.0, format="%.2f")

# Create expandable section for additional features
with st.expander("Additional Features"):
    has_cr_card = st.radio('💳 Has Credit Card?', 
                          ['Yes', 'No'], horizontal=True)
    is_active_member = st.radio('✅ Is Active Member?', 
                               ['Yes', 'No'], horizontal=True)

# Convert Yes/No to 1/0
has_cr_card = 1 if has_cr_card == 'Yes' else 0
is_active_member = 1 if is_active_member == 'Yes' else 0



input_data = pd.DataFrame({
    'Geography': [geographic],
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary' : [estimated_salary]
})


   
geo_encoded = ohe_geo.transform([[geographic]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=ohe_geo.get_feature_names_out(['Geography']))

## concatination one hot encoded 
input_df=pd.concat([input_data.drop("Geography",axis=1),geo_encoded_df],axis=1)
cols = input_df.columns.tolist()
new_order = cols[-3:] + cols[:-3]
input_df = input_df[new_order]

st.write(input_df)


# # Scale the data
input_scaled = scaler.transform(input_df)

# ## Predict the churn
prediction = model.predict(input_scaled)

prediction_probaa = prediction[0][0]
st.write('Predicted churn probability is' + ' -- ' + str(prediction_probaa))

if prediction_probaa > 0.5:
    st.write('The Customer is likely to churn.')
else:
    st.write('the Customer is not likely to churn.')
