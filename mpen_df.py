#
# MPEN Demand Flexibility Tool Landing Page
# Authors: Anoop R Kulkarni, Kaustubh Arekar
# Version 1.0
# Dec 11, 2023

import mpen_df_model
import mpen_df_analysis
#import mpen_df_viz
import streamlit as st
import toml

sites = {}
sites['Demo (Test data)'] = {'prefix':'demo'}

PAGES = {
    "About": mpen_df_model,
    "Analysis": mpen_df_analysis,
    #"Visualization": mpen_df_viz,
}

header = st.container()

primaryColor = toml.load("config.toml")['theme']['primaryColor']
s = f"""
<style>
div.stButton > button:first-child {{ border: 5px solid {primaryColor}; border-radius:20px 20px 20px 20px; }}
<style>
"""
st.markdown(s, unsafe_allow_html=True)

with header:
   mpen, title, ae = st.columns([1,5,1])
   with mpen:
      st.image('res/logo-mpen.png', width=200)

st.title("Demand Flexibility")

if 'explored' not in st.session_state:
   st.session_state.explored = False

lounge = st.empty()
with lounge.container():
      st.image ('res/demand_flex.jpg')

      st.sidebar.title('Selection')
      selection = st.sidebar.radio("Select variant", sites.keys())
   
      explored = st.sidebar.button("Explore!")
   
      if explored:
         st.session_state.explored = True
      
      if not st.session_state.explored:
         with lounge.container():
            st.image('res/demand_flex.jpg')
   
      if st.session_state.explored:
         sel_category = st.sidebar.radio("Select Category", list(PAGES.keys()))
         lounge.empty()
         with lounge.container():
            PAGES[sel_category].app(sites[selection])
