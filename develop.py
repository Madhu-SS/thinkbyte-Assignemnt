import streamlit as st
import os
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import SequentialChain
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

st.title("AI-Assisted Analysis of Renewable Energy Sector Trends")
# text=st.text_input("Please enter your Querry")


load_dotenv()
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

llm = ChatGoogleGenerativeAI(model="gemini-pro")

prompt_template_first = PromptTemplate(
    input_variables=["question"],
    template="For my {question}, kindly provide key information from a collection of articles related to renewable energy sector trends")

first_chain=LLMChain(llm=llm, prompt=prompt_template_first,output_key="information")

prompt_template_second = PromptTemplate(
    input_variables=["information"],
    template="Synthesize the gathered {information}into a structured report,highlighting key findings,emerging trends, and providing recommendations for stakeholders as a final result")

second_chain=LLMChain(llm=llm, prompt=prompt_template_second,output_key="final_result")

prompt_template_third = PromptTemplate(
     input_variables=["final_result"],
    template="Summarize the {final_result} within 200 words and provide the answer in paragraph as the final_answer .")

third_chain=LLMChain(llm=llm, prompt=prompt_template_third,output_key="final_answer")

chain=SequentialChain(chains=[first_chain, second_chain,third_chain],
    input_variables=["question"],
    output_variables=["information","final_result","final_answer"])



text = st.text_input("Please enter your Query")
if st.button('Submit'):
    if not text:
        st.warning("Please enter your Query")
    else:
        with st.spinner("Please wait, it may take some time to run..."):
            result = chain({"question": text})
            st.write(result.get("final_answer"))
