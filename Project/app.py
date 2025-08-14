# app.py (Final Version for Local and Deployed)

import streamlit as st
import os
import nest_asyncio

nest_asyncio.apply()

from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AI Judicial Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- FINAL, ROBUST API KEY SETUP ---
st.sidebar.title("Configuration")

api_key = None
# This block intelligently handles the API key for both deployed and local environments
try:
    # This will work when the app is deployed on Streamlit Community Cloud
    api_key = st.secrets["GOOGLE_API_KEY"]
    st.sidebar.success("API Key loaded from secrets!", icon="✅")
except:
    # This will work when the app is run locally
    st.sidebar.warning("Could not find Streamlit secret. Checking for local api.txt file.")
    if os.path.exists("api.txt"):
        api_key = open("api.txt").read().strip()
        st.sidebar.success("API Key loaded from local api.txt!", icon="✅")
    else:
        api_key = st.sidebar.text_input("Please enter your Google API Key:", type="password")

if not api_key:
    st.info("Please provide your Google API Key in Streamlit Secrets or a local api.txt file.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = api_key


# --- CACHED AI MODELS AND DATA ---
# (The rest of your code remains exactly the same)
@st.cache_resource
def load_resources():
    try:
        folder_path = 'poc_bail_cases'
        if not os.path.exists(folder_path):
            return "Error: Directory '{}' not found. Please ensure your 'poc_bail_cases' folder is in your GitHub repository.".format(folder_path)

        case_documents = []
        for filename in os.listdir(folder_path):
            if filename.endswith(".txt"):
                file_path = os.path.join(folder_path, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    case_documents.append(file.read())

        if not case_documents:
            return "Error: No '.txt' files found in '{}'.".format(folder_path)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        docs = text_splitter.create_documents(case_documents)

        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_documents(docs, embeddings)
        llm = GoogleGenerativeAI(model="gemini-1.5-flash-latest")
        
        return {"llm": llm, "vector_store": vector_store}

    except Exception as e:
        return f"An error occurred while loading resources: {e}"

# Load resources and handle potential errors
resources = load_resources()
if isinstance(resources, str):
    st.error(resources)
    st.stop()
llm = resources["llm"]
vector_store = resources["vector_store"]


# --- USER INTERFACE ---
st.title("⚖️ AI Judicial Assistant")
st.markdown("A powerful tool to assist with legal research and analysis. Choose a function below.")

# Create tabs for each function
tab1, tab2, tab3 = st.tabs(["📄 Summarize Judgment", "📚 Find Precedents", "🔮 Predictive Analysis"])


# --- TAB 1: SUMMARIZE JUDGMENT ---
with tab1:
    st.header("Automated Case Summarization")
    st.markdown("Upload a judgment as a `.txt` file or paste the text below.")
    
    uploaded_file = st.file_uploader("Upload a .txt file", type="txt")
    pasted_text = st.text_area("Or Paste Judgment Text Here:", height=250)
    
    if st.button("Generate Summary", key="summarize"):
        judgment_text = None
        if uploaded_file:
            judgment_text = uploaded_file.getvalue().decode("utf-8")
        elif pasted_text:
            judgment_text = pasted_text
        
        if judgment_text:
            with st.spinner("Analyzing the document..."):
                summary_prompt = PromptTemplate.from_template("Provide a detailed, multi-point summary of this judgment: {judgment}")
                summary_chain = summary_prompt | llm | StrOutputParser()
                summary_result = summary_chain.invoke({"judgment": judgment_text})
                st.info("Summary Generation Complete!")
                with st.expander("**View AI-Generated Summary**", expanded=True):
                    st.markdown(summary_result)
        else:
            st.warning("Please upload a file or paste text to generate a summary.")

# --- TAB 2: FIND PRECEDENTS ---
with tab2:
    st.header("Intelligent Precedent Retrieval")
    st.markdown("Describe a case scenario to find the most relevant precedents from the database.")
    
    scenario_input = st.text_area("Enter a brief case scenario:", height=200)
    
    if st.button("Find Similar Cases", key="precedent"):
        if scenario_input:
            with st.spinner("Searching database and analyzing precedents..."):
                rag_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=vector_store.as_retriever(),
                    return_source_documents=True
                )
                question = f"Analyze the following scenario and find relevant precedents. For each precedent, explain in detail why it is relevant, quoting key phrases from the document that support your reasoning. Scenario: '{scenario_input}'"
                rag_result = rag_chain.invoke(question)

                st.success("Precedent Analysis Complete!")
                st.subheader("Analysis Results:")
                st.markdown(rag_result['result'])
                
                with st.expander("**View Retrieved Source Documents**"):
                    for doc in rag_result['source_documents']:
                        st.markdown("---")
                        st.write(doc.page_content[:500] + "...")
        else:
            st.warning("Please enter a scenario to find precedents.")

# --- TAB 3: PREDICTIVE ANALYSIS ---
with tab3:
    st.header("Predictive Scenario Analysis")
    st.warning("⚠️ **Disclaimer:** This is a demonstrative feature. Predictions are based on historical patterns and are not a guarantee of future outcomes.", icon="🤖")
    
    predict_scenario = st.text_area("Enter a case brief for predictive analysis:", height=200)
    
    if st.button("Generate Prediction", key="predict"):
        if predict_scenario:
            with st.spinner("Running predictive models..."):
                predictive_prompt = f"You are a legal predictive analyst. Based on patterns in provided documents, analyze the following scenario. Provide a report covering: 1. Predicted Outcome Probability (e.g., bail granted/rejected). 2. Key Influencing Factors. 3. Potential Future Events. Scenario: \"{predict_scenario}\""
                rag_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_store.as_retriever())
                prediction_result = rag_chain.invoke(predictive_prompt)
                
                st.subheader("Predictive Analysis Report")
                st.markdown(prediction_result['result'])
        else:
            st.warning("Please enter a case brief to generate a prediction.")