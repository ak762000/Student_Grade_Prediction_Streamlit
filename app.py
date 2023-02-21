import pickle
import numpy as np
import pandas as pd
import streamlit as st

# loading the saved model
loaded_model = pickle.load(open('Student_Grades_Prediction.pkl', 'rb'))

# loading the scalar model
load_scaler = pickle.load(open('Standard_Scalar.pkl', 'rb'))


# creating a function for Prediction

def student_prediction(input_data):
    

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(load_scaler.transform(input_data_reshaped))


    if (prediction[0] <= 2):
         return 'The Final result is given by : ',prediction[0]
    if(prediction[0] <=6):
        return 'The Final result is 1given by : ',prediction[0]
    else:
        return 'The Final result is 2given by : ',prediction[0]
   
    
def main():
    
    
    # giving a title
    st.title('Student Grades Prediction')
    
    
    # getting the input data from the user
    
    
    G1 = st.text_input('Enter G1')
    G2 = st.text_input('Enter G2')
    Absences = st.text_input('Enter Absences')
    Activities = st.text_input('Enter Activities')
    Paid = st.text_input('Enter Paid Details')
    Failures = st.text_input('Enter Failures')
    StudyTime = st.text_input('Enter studytime')
    
    
    # code for Prediction
    grade = ''
    
    # creating a button for Prediction
    
    if st.button('Student Grade Test Result'):
        grade = student_prediction([G1,G2,Absences,Activities,Paid,Failures,StudyTime])
        
        
    st.success(grade) 
    
if __name__ == '__main__':
    main()
    
