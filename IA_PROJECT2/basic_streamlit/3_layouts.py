import streamlit as st
import pandas as pd
import time

side_bar=st.sidebar

side_bar.write('Conchita creates a sidebar')

side_bar.header('This will be my sidebar')
side_bar.caption('elements that added in the sidebard are pined to the left')

st.write('this will be outside the sidebard')

st.markdown('---')

#lets continue with layouts columns

st.header('Lets see columns in action:classical_building:')
col1,col2,col3=st.columns(3)

with col1:
	st.subheader('The column 1')
	st.image('./media/column1.jpg')
with col2:
	st.subheader('The column 2')
	st.image('./media/column2.jpg')
with col3:
	st.subheader('The column 3')
	st.image('./media/column3.jpg')

st.markdown('---')

#Now lets see expander

st.header('Now we will play with expander')
with st.expander('Lets see an example'):
	st.write("""
		insert a multi element container that can be expanded or colapsed 
		by the user """)

st.markdown('---')

st.header('Time to see containers')
with st.container():
	st.write('you are inside the container')

st.markdown('---')
st.header('Lets finish with empty')
placeholder=st.empty()
placeholder.write('Hello EDEM Students!!!!!!')
time.sleep(0.1)
placeholder.empty()