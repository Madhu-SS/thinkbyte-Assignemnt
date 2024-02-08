%%writefile main.py
import streamlit as st
import os
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

st.title("AI-Assisted Analysis of Renewable Energy Sector Trends")
text=st.text_input("Please enter your Querry")


load_dotenv(".env")
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

llm = ChatGoogleGenerativeAI(model="gemini-pro")

prompt_template_name = PromptTemplate(
    input_variables=["question"],
    template="For my {question}, kindly provide information on the renewable energy sector, emphasizing recent advancements, policy impacts, public sentiment, and future trends. The objective is to create a comprehensive report that offers valuable insights into the sector."
)

chain=LLMChain(llm=llm,prompt=prompt_template_name)

if st.button('Submit'):
    # Check if the user has entered a review
    if not text:
        st.warning("Please enter your Querry")
    else:
        result=chain.run(text)
        st.write(result)