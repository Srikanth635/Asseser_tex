import time
import streamlit as st

# Streamlit UI Initialization
st.set_page_config(page_title="RAG-Based Legal Assistant")
col1, col2, col3 = st.columns([1, 25, 1])
with col2:
    st.title("RAG-Based Legal Assistant")

# Environment Setup
import os
from dotenv import load_dotenv
load_dotenv()

# LangChain Dependencies
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.llm import LLMChain
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain

# File Paths
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir,"..", "files")
persistent_directory = os.path.join(current_dir, "data-ingestion-local")

# LLM Configuration
chat_model = ChatOpenAI(model_name="gpt-4o", temperature=0.15)
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Load the Vector Database
vector_db = FAISS.load_local(persistent_directory, embedding_model,allow_dangerous_deserialization=True)

# Retriever Setup
retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# History-Aware Retriever (Rephrasing Query Template)
rephrasing_template = """
**Objective**: Assess the writing style of the Collaborative Research Centre (CRC) proposal for the EASE project. Evaluate the clarity, coherence, precision, and professional tone of the proposal across key sections. Provide detailed ratings, constructive feedback, and actionable recommendations for improvement. The final output should be formatted as a **LaTeX document** with the following specifications:
- **Font**: 11pt Helvetica.
- **Page Style**: Full-page margins using the `fullpage` package.
- **Sections**: Organize using `\section` and `\subsection` headings.
- **Tables**: Summarize key findings and ratings in a formatted table.
- **Lists**: Use `itemize` for bullet points to present strengths, weaknesses, and recommendations.

---

### **Assessment Criteria**

Evaluate the proposal based on these aspects of writing style:

1. **Clarity**:
   - Are technical concepts and methodologies explained clearly and concisely?
   - Is the language accessible to an interdisciplinary audience while maintaining technical rigor?
   - Are acronyms and specialized terms consistently defined and explained?

2. **Precision**:
   - Does the writing avoid vague or ambiguous statements?
   - Are objectives, methodologies, and outcomes described with sufficient detail and specificity?
   - Are claims supported with appropriate data, evidence, or references?

3. **Coherence and Flow**:
   - Does the document follow a logical sequence of ideas?
   - Are transitions between sections and subsections smooth and intuitive?
   - Are repetitive or disjointed elements minimized?

4. **Professional Tone**:
   - Is the writing formal and consistent with academic and scientific standards?
   - Does the tone reflect the ambition and significance of the research without overstating claims?
   - Are any informal phrases or colloquialisms avoided?

5. **Engagement and Persuasiveness**:
   - Does the proposal capture the reader’s attention with compelling arguments?
   - Are the societal, scientific, and economic impacts of the research effectively communicated?
   - Are unique aspects of the research emphasized to distinguish it from similar initiatives?

6. **Grammar, Syntax, and Formatting**:
   - Are sentences well-constructed, grammatically correct, and free of typographical errors?
   - Does the formatting enhance readability, with consistent headings, bullet points, and lists?
   - Are visuals, tables, and charts (if applicable) integrated effectively into the narrative?

---

### **Output Requirements**

1. Assign a **numerical rating (1–10)** for each criterion.
2. Provide **detailed explanations** for each rating, including:
   - Specific examples of strong and weak writing practices.
   - Suggestions for improving clarity, tone, or engagement.
3. Summarize ratings and key findings in a **LaTeX table**.
4. Write the report as a **LaTeX document** formatted as follows:
   - Use `\section` and `\subsection` for structure.
   - Summarize ratings and observations in a formatted table using `\tabular`.
   - Include bullet points (`itemize`) to list strengths, weaknesses, and recommendations for each criterion.

---

### **Example LaTeX Report Structure**

1. **Title**: Assessment of Proposal Writing Style.
2. **Introduction**: Brief description of the purpose and scope of the writing style assessment.
3. **Assessment Sections**:
   - Each section corresponds to one criterion (e.g., Clarity, Precision).
   - Include a numerical rating, detailed evaluation, and recommendations.
4. **Summary Table**: Present ratings and observations in a formatted table.
5. **Conclusion**: Summarize key takeaways and provide a roadmap for improving the proposal’s writing style. 
"""

rephrasing_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", rephrasing_template),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Conversational Retrieval Chain
conversational_rag_chain = LLMChain.from_llm(
    llm=chat_model,
    retriever=retriever,
    return_source_documents=False
)

# Streamlit Session State Management
if "messages" not in st.session_state:
    st.session_state["messages"] = []

def reset_conversation():
    st.session_state["messages"] = []

# Display Existing Messages
for message in st.session_state.messages:
    with st.chat_message(message.type):
        st.write(message.content)

# Accept User Query
user_query = st.chat_input("Ask me anything...")

if user_query:
    # Display User Query
    with st.chat_message("user"):
        st.write(user_query)

    # Generate Assistant Response
    with st.chat_message("assistant"):
        with st.spinner("Generating response..."):
            result = conversational_rag_chain(
                {"question": user_query, "chat_history": st.session_state["messages"]}
            )
            response = result["answer"]

        # Add Reset Button
        st.button("Reset Conversation 🗑️", on_click=reset_conversation)

    # Save Conversation in Session State
    st.session_state["messages"].extend(
        [
            {"type": "user", "content": user_query},
            {"type": "assistant", "content": response},
        ]
    )
