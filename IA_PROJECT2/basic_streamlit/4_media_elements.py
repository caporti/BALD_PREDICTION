import streamlit as st 

trikilin = st.image

st.header('Display my lovely dog image using st.image')

trikilin('./media/fotito.jpeg', caption='Beautiful Trosko', width=500)

st.header('Display video of my Google Team')

video_file = open('./media/google_team.mp4','rb')
video_bytes = video_file.read()
st.video(video_bytes)

st.header('Display an audio track')
audio_file = open('./media/audio.mp3','rb')
audio_bytes = audio_file.read()
st.audio(audio_bytes,format='audio/ogg')