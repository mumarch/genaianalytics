# numpy version:1.26.4 is compatable with pandasai
# pip install numpy==1.26.4  pyyaml

#app4.1 version at local disk

import pandas as pd
from pandasai.llm.openai import OpenAI
import streamlit as st # type: ignore
#import numpy as np

#from pandasai.llm.local_llm import LocalLLM
from pandasai import SmartDataframe



st.title("Data Analysis using Prompts")

api_key = st.text_input("Enter your OpenAI Api Key")

if st.button("Submit"):
    OPENAI_API_KEY = api_key
    model = OpenAI(api_token=OPENAI_API_KEY)

    uploaded_file = st.file_uploader("Upload a CSV file", type=['csv'])

    # reading the uploaded dataset
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write(data.head(2))

        df = SmartDataframe(data, config={"llm":model})

        prompt = st.text_area("Enter your prompt")
    
        if st.button("Generate"):
            if prompt:
                with st.spinner("Generating response..."):
                    st.write(df.chat(prompt))
                





