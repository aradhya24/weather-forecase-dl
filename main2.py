# app.py
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
from io import BytesIO
# from keras.saving import register_keras_serializable

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []


st.markdown("""
<style>
.user-message {
    background-color: #e0f7fa;
    padding: 10px;
    border-radius: 5px;
    margin-bottom: 10px;
}
.bot-message {
    background-color: #f1f8e9;
    padding: 10px;
    border-radius: 5px;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)




from prophet_script2 import read_process, evaluate, forecast
from nbeats2 import read_and_process_nbeats, make_forecast_dates, make_future_forecast, NBeatsBlock, plot_time_series, WINDOW_SIZE

def save_fig_to_bytes(fig):
    """ Save a Matplotlib figure to a BytesIO object. """
    img_bytes = BytesIO()
    fig.savefig(img_bytes, format='png')
    img_bytes.seek(0)
    return img_bytes



def add_fig(fig):
    # image = Image.open(fig)
    # st.image(image, caption="Uploaded Image.", use_column_width=True)

    # Convert the image to bytes
    # img_byte_arr = io.BytesIO()
    # image.save(img_byte_arr, format=image.format)
    # img_byte_arr = img_byte_arr.getvalue()

    # Add chat message for the initial prompt
    st.chat_message("📈").write("Analyze the trends in this graph.")

    user_prompt = st.chat_input("Enter your prompt here:")
    
    if user_prompt:
        # Hardcoded default prompt for forecasting graphs and trend analysis
        default_prompt = """
            You are an expert in analyzing forecasting graphs for trend analysis.
            You will receive input images as graphs and you will have to answer questions based on the observed trends.
        """
        
        # Combine the default prompt with the user-provided prompt
        combined_prompt = f"{default_prompt}\n{user_prompt}"

        # Pass the combined prompt and image bytes to the model
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content([combined_prompt, fig], stream=True)
        response.resolve()

        st.session_state.conversation_history.append(("👦🏻", user_prompt, "user-message"))
        st.session_state.conversation_history.append(("🤖", response.text, "bot-message"))

        # Display the conversation history
        for speaker, message, css_class in st.session_state.conversation_history:
            st.markdown(f'<div class="{css_class}">{speaker} : {message}</div>', unsafe_allow_html=True)



def app():
    st.title("Generate Forecasts")
    st.sidebar.title("File Upload")

    uploaded_file = st.sidebar.file_uploader("Upload a File", type="csv")
    if uploaded_file is not None:
        end_date = st.date_input("Enter Last Date to be Forecasted", datetime(2019, 7, 6))

        model_selection = st.selectbox("Select Model to generate Forecasts", ["Prophet", "N-Beats"], index=None, placeholder="Models..")

        if st.button("Generate Forecasts?"):
            if model_selection == "Prophet":
                model = Prophet()
                df = read_process(uploaded_file)
                timesteps, freq = evaluate(df, end_date)
                fut, last_idx, fig = forecast(model, df, timesteps, freq)
                download_file = pd.DataFrame()
                download_file["Date"] = fut["ds"][last_idx:]
                download_file["Predictions"] = fut["yhat"][last_idx:]
                download_file = download_file.reset_index(drop=True)
                csv = download_file.to_csv(index=False).encode('utf-8')
                st.success("Forecasts Generated")
                st.download_button(label="Download Forecasts as CSV", data=csv, file_name='prophet_forecasts.csv', mime='text/csv')
                fig_bytes = save_fig_to_bytes(fig)
                st.download_button(label="Download Forecast as Image", data=fig_bytes, file_name='prophet_forecasts.png', mime='image/png')
                add_fig(fig)
            

            elif model_selection == "N-Beats":
                nbeats_model = tf.keras.models.load_model("C:/Users/Siddharth/Desktop/woodpeckers/nbeats.keras", custom_objects={'NBeatsBlock': NBeatsBlock})
                df = pd.read_csv(uploaded_file, parse_dates=["Date"])
                df['Date'] = df['Date'].dt.strftime("%m/%d/%Y")
                a, b = read_and_process_nbeats(df)
                x, y = make_forecast_dates(df, end_date)
                preds = make_future_forecast(b, nbeats_model, y, WINDOW_SIZE)
                forecast_df = pd.DataFrame()
                forecast_df["Date"] = x
                forecast_df["Predictions"] = preds
                forecast_df = forecast_df.reset_index(drop=True)
                fig = plot_time_series(timesteps=forecast_df["Date"], values=forecast_df["Predictions"])
                csv = forecast_df.to_csv(index=False).encode('utf-8')
                st.success("Forecasts Generated")
                st.download_button(label="Download Forecasts as CSV", data=csv, file_name='nbeats_forecasts.csv', mime='text/csv')
                fig_bytes = save_fig_to_bytes(fig)
                st.download_button(label="Download Forecast as Image", data=fig_bytes, file_name='nbeats_forecasts.png', mime='image/png')
                add_fig(fig)

            

if __name__ == "__main__":
    app()
