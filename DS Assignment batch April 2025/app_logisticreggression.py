import pickle
import streamlit as st
import numpy as np

# Load trained model
with open('logistic_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Title
st.title('Titanic Survival Prediction')

# User inputs
pclass = st.selectbox('Passenger Class', [1, 2, 3])
sex = st.selectbox('Sex', ['male', 'female'])
age = st.slider('Age', 0, 100, 25)
sibsp = st.number_input('Number of Siblings/Spouses Aboard', 0, 10, 0)
parch = st.number_input('Number of Parents/Children Aboard', 0, 10, 0)
fare = st.number_input('Fare', 0.0, 500.0, 50.0)
embarked = st.selectbox('Port of Embarkation', ['C', 'Q', 'S'])

# Encode inputs
sex_encoded = 1 if sex == 'male' else 0
embarked_mapping = {'C': 0, 'Q': 1, 'S': 2}
embarked_encoded = embarked_mapping[embarked]

# Create input array
input_data = np.array([[pclass, sex_encoded, age, sibsp, parch, fare, embarked_encoded]])

# Prediction
if st.button('Predict'):
    prediction = model.predict(input_data)
    result = 'Survived' if prediction[0] == 1 else 'Did Not Survive'
    st.write(f'Prediction: {result}')