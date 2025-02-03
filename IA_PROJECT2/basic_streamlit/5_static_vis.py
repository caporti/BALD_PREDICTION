import streamlit as st
import pandas as pd
import numpy as np

#lets create our first static visualizations using Matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

st.header("Matplotlib Visualizations Streamlit:white_check_mark:")

#First we need to load our data

df=pd.read_csv('./tips.csv')
st.dataframe(df.head(6))

st.markdown('---')

st.subheader('Lets find the distribution between males and females	:man-girl:')
value_count=df['sex'].value_counts()
st.dataframe(value_count)


#create a pie chart plot to represent that 

with st.container():
	value_count=df['sex'].value_counts()
	fig,ax=plt.subplots()
	ax.pie(value_count,autopct='%0.2f%%',labels=['Male','Female'])
	st.pyplot(fig)


#now we draw a bar chart
	st.write(value_count.index)
	fig,ax=plt.subplots()
	ax.bar(value_count.index,value_count)
	st.pyplot(fig)



st.markdown('---')
#now i will represent all of that but in a more beautiful representation using layouts

with st.container():
	value_count=df['sex'].value_counts()
	col1,col2=st.columns(2)
	
	with col1:
		st.subheader('Pie Chart')
		fig,ax=plt.subplots()
		ax.pie(value_count,autopct='%0.2f%%',labels=['Male','Female'])
		st.pyplot(fig)

	with col2:
		st.subheader('Bar Chart')
		fig,ax=plt.subplots()
		ax.bar(value_count.index,value_count)
		st.pyplot(fig)

#Finally i will show the distribution using an expander

	with st.expander('Click here to see the value counts'):
		st.dataframe(value_count)


st.markdown('---')
#now lets add widget to select the features to plot


data_types = df.dtypes
cat_cols=tuple(data_types[data_types=='object'].index)

with st.container():
	feature=st.selectbox('Select the feature to display',
						cat_cols
						)
	value_count=df[feature].value_counts()
	col1,col2=st.columns(2)
	
	with col1:
		st.subheader('Pie Chart')
		fig,ax=plt.subplots()
		ax.pie(value_count,autopct='%0.2f%%',labels=value_count.index)
		st.pyplot(fig)

	with col2:
		st.subheader('Bar Chart')
		fig,ax=plt.subplots()
		ax.bar(value_count.index,value_count)
		st.pyplot(fig)

st.markdown('---')
st.header("Seaborn Visualizations Streamlit	:cool:")

st.subheader('First Lets find the distribution of spent between males and females:moneybag:')

with st.container():
	fig,ax=plt.subplots()
	sns.boxplot(x='sex', y='total_bill', data=df, ax=ax)
	st.pyplot(fig)

#Lets allow the user select the chart that want to use :)


with st.container():
	#box, violin, kdeplot, histogram
	chartselection=('box','violin','kdeplot','histogram')
	chart_selection=st.selectbox('Chart type to use:',chartselection)
	fig,ax=plt.subplots()

	if chart_selection=='box':
		sns.boxplot(x='sex', y='total_bill', data=df, ax=ax)

	elif chart_selection=='violin':
		sns.violinplot(x='sex', y='total_bill', data=df, ax=ax)

	elif chart_selection=='kdeplot':
		sns.violinplot(x=df['total_bill'], hue=df['sex'], ax=ax)

	else:
		sns.histplot(x='total_bill', hue='sex', data=df, ax=ax)

	st.pyplot(fig)



st.markdown('---')
st.header("Pandas Visualizations Streamlit:panda_face:")

st.subheader('First the distribution average of bills across each day by  males and females:calendar:')

features_to_groupby=['day','sex']
feature=['total_bill']
select_cols=feature + features_to_groupby

avg_total_bill=df[select_cols].groupby(features_to_groupby).mean()

#lets  unstack the graph

avg_total_bill=avg_total_bill.unstack()
st.dataframe(avg_total_bill)


#lets create the graph

fig,ax=plt.subplots()
avg_total_bill.plot(kind='bar',ax=ax)


#lets move the legend
ax.legend(loc='center left',bbox_to_anchor=(1.0,0.5))
st.pyplot(fig)


st.subheader("lets play with widgets to customized all:partying_face:")

with st.container():
	#1. we will use a multichart to include all categorical features
	#2. we will use a select box to select the visual to use (bar, area or line)
	#3. we will use a radiobutton to display the graph stack or unstack

	#we will use 3 columns to separate all

	col1,col2,col3=st.columns(3)

	with col1:
		group_cols=st.multiselect("Which features you want to use", cat_cols, cat_cols[0])
		features_to_groupby=group_cols
		n_features=len(features_to_groupby)

	with col2:
		chart_type=st.selectbox("Which chart you want to plot",('bar','area','line'))

	with col3:
		stack_option=st.radio("You want to stack", ('Yes', 'No'))
		if stack_option=='Yes':
			stacked = True
		else:
			stacked = False

#And now lets create the visual part

feature=['total_bill']
select_cols=feature+features_to_groupby
avg_total_bill=df[select_cols].groupby(features_to_groupby).mean()

if n_features>1:
	for i in range(n_features - 1):
		avg_total_bill=avg_total_bill.unstack()


fig,ax=plt.subplots()
avg_total_bill.plot(kind=chart_type, ax=ax, stacked=stacked)
ax.legend(loc='center left', bbox_to_anchor=(1.0,0.5))
st.pyplot(fig)

#finally we put the dataframe inside of one expander
with st.expander('Click here to view the values'):
	st.dataframe(avg_total_bill)


st.markdown('---')
st.header('Finally we will find the relation between total_bill and tip on time')

fig,ax=plt.subplots()
sns.scatterplot(x='total_bill', y='tip', hue='time', ax=ax, data=df)
st.pyplot(fig)


st.write('Lets play with widgets to dinamically change the hue category')

fig,ax=plt.subplots()
hue_type=st.selectbox('Which feature you want to use for hue', cat_cols)
sns.scatterplot(x='total_bill', y='tip', hue=hue_type, ax=ax, data=df)
st.pyplot(fig)
























































