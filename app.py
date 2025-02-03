import streamlit as st
import pandas as pd
import numpy as np

# display almost anything with st.write

st.write("Hello World EDEM students from Conchita")

st.write("Welcome to our streamlit sessions")

st.write(123455154125151513)
st.write('Inglés o español?')

st.markdown("### eueuee")

st.balloons()


df=pd.DataFrame({
	"first_column":[1,2,3,4,5],
	"second_column": [10,20,30,40,50]
})

st.write(df)

st.write(np.array([1,2,3,4]))

