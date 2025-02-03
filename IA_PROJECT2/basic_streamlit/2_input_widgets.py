import streamlit as st 
import pandas as pd
import numpy as np
import os


# IE students lets create a button widget
button = st.button('BUTTON')
st.write('button =', button)

#load the data that we will use
data = pd.read_csv('tips.csv')

#Create a funcion to randomly show us 3 rows of a dataframe
def display_random(df):
	sample=df.sample(3)
	return sample

st.markdown("---")

#we create the button to submit the function
st.subheader('Displaying 3 rows')
st.caption('click on the button to display')
new_button=st.button('Display random 3 rows')
if new_button:
	sample = display_random(data)
	st.dataframe(sample)


st.markdown('---')
st.subheader(' we will use checkbox')
yes = st.checkbox('I agree the terms and conditions')
st.write('status=',yes)

#now we will use multiple checkbox
with st.container():
	st.info('which is you favourite tech :)')
	python=st.checkbox('Python')
	sql=st.checkbox('SQL')
	ai_ml=st.checkbox('AI/ML')
	datascience=st.checkbox('Data Science')

	techButton = st.button('Submit')
	if techButton:
		tech_dict={
			'Python':python,
			'SQL':sql,
			'AI/ML':ai_ml,
			'Data Science':datascience,

		}
		st.json(tech_dict)

# Now IE Students lets see the Radio Buttons!!!!!
st.markdown('---')
st.subheader('Lets play with Radio Buttons :smile:')
radio=st.radio('which is your favourite team?',
	('Real Madrid','Barcelona','Atletico de Madrid', 'Sevilla', 'Valencia'))

st.write('Your team is:',radio)

#Lets see the Select Box
st.markdown('---')
st.subheader('Lets learn how to use Select Box:clap:')
select_box = st.selectbox('what skill you want to learn',('Java','Python','C','C++'))
st.write('your selection is:', select_box)

#Now we will have fun with Multi Select
st.markdown('---')
st.subheader('How it works:pushpin:')
genres=st.multiselect('What kind of movies you like?',
	['Comedy','Action','Love','Thriller','Scify'])
st.write('You like', genres)

#Now lets talk about slider
st.markdown('---')
st.subheader('How to work with sliders:on:')
stars = st.slider('Rate this course (1 to 5 ⭐):', 1, 5, 1)
st.write('Your rating is:', '⭐' * stars)

#lets see how to work with data introduced by the user
st.markdown('---')
st.subheader('How we manage data introduced by the users')
name=st.text_input('Please IE Student introduce your name')
age=st.number_input('Which is your age?:',min_value=0,max_value=99,value=25,step=1)
describe=st.text_area(height=150, label="what do you expect from this course")
birth=st.date_input('Tell us your date of birth')


st.markdown('---')
#we will refactor the code to get the responses saved using a button

with st.container():
	name=st.text_input('Please Student introduce your name')
	age=st.number_input('Tell us your age?:',min_value=0,max_value=99,value=25,step=1)
	describe=st.text_area(height=150, label="what do you want to learn on this course")
	birth=st.date_input('When is your date of birth')

	submit_button=st.button("Submit your info")
	if submit_button:
		info={
		'Name':name,
		'Age': age,
		'Birth': birth,
		'Expectations': describe
		}
		st.json(info)

