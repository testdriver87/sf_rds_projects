from pycaret.regression import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np

model = load_model('rds_diploma_061220')

def predict(model, input_df):
    predictions_df = predict_model(estimator=model, data=input_df)
    predictions = predictions_df['Label'][0]
    return predictions

def run():

    add_selectbox = st.sidebar.selectbox("How would you like to predict?", ("Online", "Batch"))

    st.sidebar.info('Heart Failure Prediction')

    st.title("Heart Failure Prediction")

    if add_selectbox == 'Online':

        age = st.number_input('Age', min_value=1, max_value=100, value=60)
        anaemia = st.selectbox('Anaemia', ['0', '1'])
        creatinine_phosphokinase = st.number_input('Creatinine Phosphokinase', min_value=0, max_value=10000, value=582)
        diabetes = st.selectbox('Diabetes', ['0', '1'])
        ejection_fraction = st.number_input('Ejection Fraction', min_value=0, max_value=100, value=35)
        high_blood_pressure = st.selectbox('High Blood_Pressure', ['0', '1'])
        platelets = st.number_input('Platelets', min_value=0, max_value=1000000, value=250000)
        serum_creatinine = st.number_input('Serum Creatinine', min_value=0, max_value=10, value=1)
        serum_sodium = st.number_input('Serum Sodium', min_value=100, max_value=150, value=135)
        sex = st.selectbox('Sex', ['0', '1'])
        smoking = st.selectbox('Smoking', ['0', '1'])
        time = st.number_input('Time', min_value=0, max_value=300, value=200)

        output = ""

        input_dict = {'age' : age, 'anaemia' : anaemia, 'creatinine_phosphokinase' : creatinine_phosphokinase, 'diabetes' : diabetes, 
                      'ejection_fraction' : ejection_fraction, 'high_blood_pressure' : high_blood_pressure, 'platelets' : platelets, 
                      'serum_creatinine' : serum_creatinine, 'serum_sodium' : serum_sodium, 'sex' : sex, 'smoking' : smoking, 'time' : time}
        input_df = pd.DataFrame([input_dict])

        if st.button("Predict"):
            output = predict(model=model, input_df=input_df)
            output = '$' + str(output)

        st.success('The output is {}'.format(output))

    if add_selectbox == 'Batch':

        file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])

        if file_upload is not None:
            data = pd.read_csv(file_upload)
            predictions = predict_model(estimator=model,data=data)
            st.write(predictions)

if __name__ == '__main__':
    run()