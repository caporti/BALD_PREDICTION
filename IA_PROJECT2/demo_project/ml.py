# Import our packages
import streamlit as st


import joblib
import os
import numpy as np


attribute = ("""

## Attribute Information

age: age in years

sex: sex (1 = male; 0 = female)

cp: chest pain type
-- Value 1: typical angina
-- Value 2: atypical angina
-- Value 3: non-anginal pain
-- Value 4: asymptomatic

trestbps: resting blood pressure (in mm Hg on admission to the hospital)

chol: serum cholestoral in mg/dl

fbs: (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)

restecg: resting electrocardiographic results
-- Value 0: normal
-- Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)
-- Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria

thalach: maximum heart rate achieved

exang: exercise induced angina (1 = yes; 0 = no)

oldpeak = ST depression induced by exercise relative to rest

 slope: the slope of the peak exercise ST segment
-- Value 1: upsloping
-- Value 2: flat
-- Value 3: downsloping

ca: number of major vessels (0-3) colored by flourosopy

thal: 3 = normal; 6 = fixed defect; 7 = reversable defect

num: diagnosis of heart disease (angiographic disease status)
-- Value 0: < 50% diameter narrowing
-- Value 1: > 50% diameter narrowing

""")


encoded_values = {"Female":0, "Male":1,'typical angina':0, 'atypical angina':1,"non-anginal pain":2, "asymptomatic":3,
"lower than 120mg/ml":0,"greater than 120mg/ml":1,'normal':0, 'ST-T wave abnormality':1, 
'left ventricular hypertrophy':2,"no":0, "yes":1, "upsloping":0, "flat":1, "downsloping":2,"normal":1, 
"fixed defect":2, "reversable defect":3}

# Function for encoding
def a(val, my_dict):
	for key, value in my_dict.items():
		if val == key:
			return value

def ml():
	st.header("This is our Prediction section")

	st.markdown(attribute)

	st.header("Give Your Input")
 

	col1, col2 = st.columns(2)

	with col1:
		age = st.number_input("Enter your Age", 29, 80, 54)
		chest_pain_type = st.selectbox("Your Chest pain type", ['typical angina','atypical angina', "non-anginal pain",
			"asymptomatic"])
		cholestrol = st.number_input("Cholesterol Value", 120, 575, 245)
		rest_ecg = st.selectbox("Your Rest ECG type", ['normal', 'ST-T wave abnormality','left ventricular hypertrophy'])
		exercise_induced_angina = st.radio("Exercise Induced Angina", ["yes", "no"])
		st_slope = st.selectbox("Your ST Slope type", ['upsloping', 'flat', 'downsloping'])

	with col2:
		sex = st.radio("What is your Gender", ["Male", "Female"])
		resting_blood_pressure = st.number_input("Resting Blood Pressure value", 90,200,131)
		fasting_blood_sugar = st.radio("Your Fasting blood sugar value is", ["lower than 120mg/ml", 'greater than 120mg/ml'])
		max_heart_rate_achieved = st.number_input("Your maximum heart rate achieved value", 70, 210, 109)
		st_depression = st.number_input("Your ST depression value", 0.0,7.0,1.0,step = 0.1)
		num_major_vessels = st.number_input("How many major vessels do you have?", 0, 3, 1)

	thalassemia = st.selectbox("Your Thalassemia type", ["normal", "fixed defect", "reversable defect"])


	with st.expander("Your selected options"):
		so = {"age":age,
		"sex":sex, "chest_pain_type":chest_pain_type, "resting_blood_pressure":resting_blood_pressure,
		"cholestrol":cholestrol, "fasting_blood_sugar":fasting_blood_sugar, "rest_ecg":rest_ecg,
		"max_heart_rate_achieved":max_heart_rate_achieved, "exercise_induced_angina":exercise_induced_angina,
		"st_depression":st_depression, "st_slope":st_slope, "num_major_vessels":num_major_vessels,
		"thalassemia":thalassemia}

		st.write(so)


		result = []

		for i in so.values():
			if type(i) == int or type(i) == float:
				result.append(i)

			else:
				res = a(i, encoded_values)
				result.append(res)

		#st.write(result)

	with st.expander("Prediction Results"):

		input = np.array(result).reshape(1,-1)
		#st.write(input)

		m = joblib.load("lr_model")

		prediction = m.predict(input)
		st.write(prediction)

		prob = m.predict_proba(input)
		#st.write(prob)

		if prediction == 1:
			st.warning("Positive Risk!!! Be Careful")
			prob_score = {"Positive Risk": prob[0][1],
			"Negative Risk": prob[0][0]}
			st.write(prob_score)

		else:
			st.success("Negative Risk!!! Enjoy!!")
			prob_score = {"Negative Risk": prob[0][0],
			"Positive Risk": prob[0][1]}
			st.write(prob_score)
