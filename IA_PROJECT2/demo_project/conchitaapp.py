# importing the packages
import streamlit as st
from PIL import Image


# importingthe files
from eda import eda
from ml import ml

def main():

	menu = ["Home", "Exploratory Data Analysis Section", "Prediction Section", "About"]

	choice = st.sidebar.selectbox("Menu", menu)

	if choice=="Home":

		st.write("# Early Stage heart attack risk predictor app")
		img = Image.open("heart.jpeg")
		st.image(img)



		st.write("""

The data that we use in this particular app contain the features of a newly heart disease patient 
or a diabetic patient.

### DataScource

	- https://archive.ics.uci.edu/ml/datasets/heart+disease

### App Content

	- This app has four sections
	1) Home Page - The page you are currently in

	2) Exploratory Data Analysis - The page in which you will find all the Data Analysis and Visualization Parts

	3) Prediction- The page in which you will be asked to give the information on all the medical aspects
		and we will predict the desired the output

	4) About - This Page is about me

			""")
	elif choice=="Exploratory Data Analysis Section":
		eda()
	elif choice == "Prediction Section":
		ml()
	else:
		st.subheader("About")
		
		st.write("### Conchita - The Google AI Trainer")
		img = Image.open("conchita.jpeg")
		st.image(img)

		st.text("""
		Concepcion Diaz Cantarero (Conchita)  es la responsable de formación de Machine Learning y Big Data en Europa de Google Cloud.  

		Apasionada con la enseñanza y la  tecnología, durante sus más de 20 años de carrera, ha formado a más de 20000 personas alrededor del mundo. 

		Conchita es autora de 10 manuales de tecnología usados por toda Europa.

		Es una de las 5 personas en el mundo en tener las 6 certificaciones de
		Cisco CCIE activas (y la única mujer). 

		Dada su pasión por la enseñanza, Conchita es ponente en numerosas conferencias
		a nivel mundial relacionadas con la inteligencia artificial y profesora visitante de varias universidades europeas,
		como por ejemplo la Universidad de Oxford

		""")

		socials = ["LinkedIn", "Github", "GMail"]
		linkedin = "https://www.linkedin.com/in/concepcion-conchita-diaz-cantarero-4a841819/"
		github = "https://github.com/conchitagoogle"
		mail = "conchitacloudtrainer@gmail.com"
		
		with st.expander("Links to all my Socials"):
			a = st.selectbox("Socials", socials)
			if a =="LinkedIn":
				st.write(linkedin)
			elif a =="Github":
				st.write(github)
			elif a=="GMail":
				st.write(mail)

			

			
			


main()
