import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.plot import add_changepoints_to_plot
import holidays
from datetime import date, datetime, timedelta
import streamlit as st
import tensorflow as tf
from PIL import Image
import os
from dotenv import load_dotenv
import google.generativeai as genai
from streamlit_option_menu import option_menu
# from keras.saving import register_keras_serializable

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


st.set_page_config(
        page_title="Forecasting",
)

import main2
import bot2


class MultiApp:

    def __init__(self):
        self.apps = []

    def add_app(self, title, func):

        self.apps.append({
            "title": title,
            "function": func
        })

    def run():
        # app = st.sidebar(
        with st.sidebar:        
            app = option_menu(
                menu_title='Forecasting ',
                options=['Main','Querying'],
                icons=['house-fill','person-circle'],
                menu_icon='chat-text-fill',
                default_index=1,
                styles={
                    "container": {"padding": "5!important", "background-color": "white"},
                    "icon": {"color": "black", "font-size": "default"}, 
                    "nav-link": {"color": "black", "font-size": "default", "text-align": "left", "margin": "0px", "--hover-color": "lightgray"},
                    "nav-link-selected": {"background-color": "#02ab21", "color": "white"},
                }
            )

        
        if app == "Main":
            main2.app()
        if app == "Querying":
            bot2.app()
             
          
             
    run()      