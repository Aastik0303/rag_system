import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import tempfile
from datetime import datetime

# LangChain Core & Google
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# Agent & Tools
from langchain.agents.agent import AgentExecutor
from langchain.agents.tool_calling_agent.base import create_tool_calling_agent
from langchain.tools import tool
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

# RAG Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ═════════════════════════════════════════════
# CUSTOM CSS FOR ENHANCED UI
# ═════════════════════════════════════════════

def load_custom_css():
    st.markdown("""
    <style>
    /* Main app styling */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3c72 0%, #2a5298 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* Card-like containers */
    .stApp > header {
        background-color: transparent;
    }
    
    /* Chat messages */
    .stChatMessage {
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Headers */
    h1, h2, h3 {
        color: white !important;
        font-weight: 700 !important;
    }
    
    /* Input boxes */
    .stTextInput > div > div > input {
        border-radius: 10px;
        border: 2px solid #667eea;
    }
    
    /* Buttons */
    .stButton > button {
        border-radius: 20px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        border: none;
        padding: 10px 24px;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);
    }
    
    /* File uploader */
    .stFileUploader {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 15px;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 28px;
        color: #667eea;
        font-weight: 700;
    }
    
    /* Info boxes */
    .stAlert {
        border-radius: 10px;
        border-left: 5px solid #667eea;
    }
    
    /* Dataframe */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
    }
    </style>
    """, unsafe_allow_html=True)

# ═════════════════════════════════════════════
# CORE CONFIGURATION
# ═════════════════════════════════════════════

def get_llm(api_key: str, temperature: float = 0):
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        google_api_key=api_key,
        temperature=temperature
    )

# ═════════════════════════════════════════════
# TOOL DEFINITIONS FOR AGENTS
# ═════════════════════════════════════════════

def create_eda_tools(df: pd.DataFrame):
    @tool
    def get_data_summary() -> str:
        """Returns comprehensive information about the dataset including columns, types, missing values, and basic statistics."""
        summary = {
            "columns": df.columns.tolist(),
            "types": df.dtypes.astype(str).to_dict(),
            "null_counts": df.isnull().sum().to_dict(),
            "shape": df.shape,
            "numeric_summary": df.describe().to_dict() if len(df.select_dtypes(include='number').columns) > 0 else {}
        }
        return json.dumps(summary, indent=2)

    @tool
    def generate_visualization(plot_type: str, column: str) -> str:
        """
        Creates a visualization. 
        plot_type options: 'histogram', 'boxplot', 'countplot', 'scatterplot' (requires column in format 'col1,col2')
        Returns success message if plot is rendered.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        try:
            if plot_type.lower() == "histogram":
                sns.histplot(df[column], kde=True, ax=ax, color='#667eea')
                ax.set_title(f'Distribution of {column}', fontsize=14, fontweight='bold')
            elif plot_type.lower() == "boxplot":
                sns.boxplot(x=df[column], ax=ax, color='#764ba2')
                ax.set_title(f'Boxplot of {column}', fontsize=14, fontweight='bold')
            elif plot_type.lower() == "countplot":
                sns.countplot(y=df[column], ax=ax, palette='viridis')
                ax.set_title(f'Count of {column}', fontsize=14, fontweight='bold')
            elif plot_type.lower() == "scatterplot":
                cols = column.split(',')
                if len(cols) == 2:
                    sns.scatterplot(data=df, x=cols[0].strip(), y=cols[1].strip(), ax=ax, color='#667eea')
                    ax.set_title(f'{cols[0]} vs {cols[1]}', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            return f"✅ {plot_type} for {column} displayed successfully!"
        except Exception as e:
            return f"❌ Error creating plot: {str(e)}"

    @tool
    def get_correlation_matrix() -> str:
        """Returns correlation matrix for numeric columns."""
        numeric_df = df.select_dtypes(include='number')
        if len(numeric_df.columns) > 1:
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(numeric_df.corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
            ax.set_title('Correlation Matrix', fontsize=14, fontweight='bold')
            st.pyplot(fig)
            plt.close(fig)
            return "✅ Correlation matrix displayed!"
        return "❌ Need at least 2 numeric columns for correlation."

    return [get_data_summary, generate_visualization, get_correlation_matrix]

def create_code_generation_tools():
    @tool
    def generate_python_code(description: str) -> str:
        """Generates Python code based on the description provided."""
        return f"I'll generate code for: {description}"
    
    @tool
    def explain_code(code_snippet: str) -> str:
        """Explains what a code snippet does."""
        return f"Analyzing code snippet..."
    
    return [generate_python_code, explain_code]

def create_web_search_tools():
    @tool
    def search_information(query: str) -> str:
        """Searches for information on the web (simulated)."""
        return f"🔍 Searching for: {query}\n\nNote: Web search is simulated in this demo. Integrate with actual search API for real functionality."
    
    return [search_information]

def create_sql_tools():
    @tool
    def generate_sql_query(description: str) -> str:
        """Generates SQL query based on natural language description."""
        return f"SQL query will be generated for: {description}"
    
    @tool
    def explain_sql_query(query: str) -> str:
        """Explains what an SQL query does."""
        return f"Explaining SQL query..."
    
    return [generate_sql_query, explain_sql_query]

# ═════════════════════════════════════════════
# STREAMLIT UI SETUP
# ═════════════════════════════════════════════

st.set_page_config(
    page_title="🤖 Gemini AI Nexus",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🤖"
)

load_custom_css()

# ═════════════════════════════════════════════
# SIDEBAR CONFIGURATION
# ═════════════════════════════════════════════

with st.sidebar:
    st.markdown("# 🤖 AI Nexus Control")
    st.markdown("---")
    
    # API Key Input
    api_key = st.text_input(
        "🔑 Google API Key",
        type="password",
        value=os.getenv("GOOGLE_API_KEY", ""),
        help="Enter your Google Gemini API key"
    )
    
    if not api_key:
        st.warning("⚠️ Please enter an API Key to continue")
        st.info("💡 Get your API key from Google AI Studio")
        st.stop()
    
    st.success("✅ API Key Validated")
    
    st.markdown("---")
    
    # Agent Selection
    st.markdown("### 🎯 Select Agent")
    chat_mode = st.selectbox(
        "Choose your AI assistant",
        [
            "💬 General Chat",
            "📊 Data Analyst",
            "📄 Document RAG",
            "💻 Code Generator",
            "🔍 Web Research",
            "🗄️ SQL Assistant",
            "🎨 Creative Writer"
        ],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Model Settings
    with st.expander("⚙️ Advanced Settings"):
        temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)
        st.info("Higher temperature = more creative responses")
    
    st.markdown("---")
    
    # Agent Info
    agent_info = {
        "💬 General Chat": "General purpose conversational AI assistant",
        "📊 Data Analyst": "Analyze CSV files with visualizations",
        "📄 Document RAG": "Question-answering from PDF documents",
        "💻 Code Generator": "Generate and explain code snippets",
        "🔍 Web Research": "Research assistant with web search",
        "🗄️ SQL Assistant": "Generate and explain SQL queries",
        "🎨 Creative Writer": "Creative content generation"
    }
    
    st.markdown(f"**Current Agent:**\n{agent_info.get(chat_mode, '')}")
    
    # Stats
    st.markdown("---")
    st.markdown("### 📊 Session Stats")
    if 'message_count' not in st.session_state:
        st.session_state.message_count = 0
    st.metric("Messages", st.session_state.message_count)

    llm = get_llm(api_key, temperature)

# ═════════════════════════════════════════════
# MAIN HEADER
# ═════════════════════════════════════════════

col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("# 🤖 Gemini AI Nexus")
    st.markdown(f"*{chat_mode} - Powered by Google Gemini*")
with col2:
    st.markdown(f"### {datetime.now().strftime('%I:%M %p')}")
    st.markdown(f"{datetime.now().strftime('%B %d, %Y')}")

st.markdown("---")

# ═════════════════════════════════════════════
# AGENT IMPLEMENTATIONS
# ═════════════════════════════════════════════

# --- DATA ANALYST AGENT ---
if chat_mode == "📊 Data Analyst":
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📊 Exploratory Data Analysis Agent")
        st.markdown("Upload a CSV file and ask questions about your data!")
    
    with col2:
        uploaded_file = st.file_uploader("📁 Upload CSV File", type="csv", label_visibility="collapsed")
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        
        # Display dataset info
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📝 Rows", df.shape[0])
        with col2:
            st.metric("📊 Columns", df.shape[1])
        with col3:
            st.metric("🔢 Numeric", len(df.select_dtypes(include='number').columns))
        with col4:
            st.metric("⚠️ Missing", df.isnull().sum().sum())
        
        st.markdown("---")
        
        # Data Preview
        with st.expander("👀 Data Preview", expanded=False):
            st.dataframe(df.head(10), use_container_width=True)
        
        # Setup Chat History
        msgs = StreamlitChatMessageHistory(key="data_chat_history")
        if len(msgs.messages) == 0:
            msgs.add_ai_message("👋 Dataset loaded successfully! I can help you with:\n- Data summary and statistics\n- Visualizations (histograms, boxplots, scatter plots)\n- Correlation analysis\n\nWhat would you like to explore?")
        
        # Setup Agent
        tools = create_eda_tools(df)
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert Data Scientist with a friendly personality. 
            Use the available tools to analyze data and create visualizations.
            ALWAYS check the data summary before making assumptions about column names.
            Provide insights and recommendations based on the data.
            Be conversational and helpful."""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        agent = create_tool_calling_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
        
        # Chat Interface
        st.markdown("### 💬 Chat with Your Data")
        
        for msg in msgs.messages:
            with st.chat_message(msg.type):
                st.markdown(msg.content)
        
        if user_input := st.chat_input("Ask about your data..."):
            st.session_state.message_count += 1
            
            with st.chat_message("human"):
                st.markdown(user_input)
            
            with st.chat_message("ai"):
                with st.spinner("🤔 Analyzing..."):
                    response = agent_executor.invoke({
                        "input": user_input,
                        "chat_history": msgs.messages
                    })
                    output = response["output"]
                    st.markdown(output)
                    msgs.add_user_message(user_input)
                    msgs.add_ai_message(output)

# --- GENERAL CHAT AGENT ---
elif chat_mode == "💬 General Chat":
    st.markdown("### 💬 General AI Assistant")
    st.markdown("*Ask me anything! I'm here to help with information, advice, and conversation.*")
    st.markdown("---")
    
    msgs = StreamlitChatMessageHistory(key="gen_chat_history")
    
    if len(msgs.messages) == 0:
        msgs.add_ai_message("👋 Hello! I'm your AI assistant. How can I help you today?")
    
    for msg in msgs.messages:
        with st.chat_message(msg.type):
            st.markdown(msg.content)
    
    if prompt := st.chat_input("Type your message..."):
        st.session_state.message_count += 1
        
        with st.chat_message("human"):
            st.markdown(prompt)
        
        with st.chat_message("ai"):
            with st.spinner("💭 Thinking..."):
                res = llm.invoke(prompt)
                st.markdown(res.content)
                msgs.add_user_message(prompt)
                msgs.add_ai_message(res.content)

# --- DOCUMENT RAG AGENT ---
elif chat_mode == "📄 Document RAG":
    st.markdown("### 📄 Knowledge Base Agent")
    st.markdown("*Upload a PDF document and ask questions about its content!*")
    st.markdown("---")
    
    msgs = StreamlitChatMessageHistory(key="rag_chat_history")
    
    pdf_file = st.file_uploader("📎 Upload PDF Document", type="pdf")
    
    if pdf_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            tmp.write(pdf_file.getvalue())
            tmp_path = tmp.name
        
        with st.spinner("📖 Processing document..."):
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=150
            )
            splits = text_splitter.split_documents(docs)
            
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            vectorstore = FAISS.from_documents(splits, embeddings)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        
        st.success(f"✅ Document processed! {len(docs)} pages indexed.")
        
        if len(msgs.messages) == 0:
            msgs.add_ai_message("📚 Document loaded! Ask me anything about its content.")
        
        for msg in msgs.messages:
            with st.chat_message(msg.type):
                st.markdown(msg.content)
        
        if query := st.chat_input("Ask a question about the document..."):
            st.session_state.message_count += 1
            
            with st.chat_message("human"):
                st.markdown(query)
            
            msgs.add_user_message(query)
            
            with st.chat_message("ai"):
                with st.spinner("🔍 Searching document..."):
                    context_docs = retriever.invoke(query)
                    context_text = "\n\n".join([d.page_content for d in context_docs])
                    
                    rag_prompt = f"""You are a helpful assistant. Use the following context from a document to answer the user's question.
                    If the answer isn't in the context, say you don't know based on the document.
                    Provide detailed and accurate answers.
                    
                    Context:
                    {context_text}
                    
                    Question: {query}
                    
                    Answer:"""
                    
                    response = llm.invoke(rag_prompt)
                    st.markdown(response.content)
                    msgs.add_ai_message(response.content)
        
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# --- CODE GENERATOR AGENT ---
elif chat_mode == "💻 Code Generator":
    st.markdown("### 💻 Code Generation Assistant")
    st.markdown("*Generate code snippets, explain code, and get programming help!*")
    st.markdown("---")
    
    msgs = StreamlitChatMessageHistory(key="code_chat_history")
    tools = create_code_generation_tools()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert programmer proficient in multiple languages.
        Help users generate clean, well-documented code.
        Explain code clearly and provide best practices.
        Format code with proper syntax highlighting using markdown code blocks."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    if len(msgs.messages) == 0:
        msgs.add_ai_message("👨‍💻 Hi! I can help you with:\n- Generating code in Python, JavaScript, Java, etc.\n- Explaining code snippets\n- Debugging and optimization\n\nWhat would you like to code today?")
    
    for msg in msgs.messages:
        with st.chat_message(msg.type):
            st.markdown(msg.content)
    
    if user_input := st.chat_input("Describe what code you need..."):
        st.session_state.message_count += 1
        
        with st.chat_message("human"):
            st.markdown(user_input)
        
        with st.chat_message("ai"):
            with st.spinner("⚡ Generating code..."):
                response = agent_executor.invoke({
                    "input": user_input,
                    "chat_history": msgs.messages
                })
                st.markdown(response["output"])
                msgs.add_user_message(user_input)
                msgs.add_ai_message(response["output"])

# --- WEB RESEARCH AGENT ---
elif chat_mode == "🔍 Web Research":
    st.markdown("### 🔍 Web Research Assistant")
    st.markdown("*Get help with research and information gathering!*")
    st.markdown("---")
    
    msgs = StreamlitChatMessageHistory(key="research_chat_history")
    tools = create_web_search_tools()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a research assistant that helps users find and synthesize information.
        Provide comprehensive, well-sourced answers.
        When you don't have current information, acknowledge it."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    if len(msgs.messages) == 0:
        msgs.add_ai_message("🔬 Hello! I'm your research assistant. I can help you:\n- Find information on various topics\n- Summarize research findings\n- Compare different sources\n\nWhat would you like to research?")
    
    for msg in msgs.messages:
        with st.chat_message(msg.type):
            st.markdown(msg.content)
    
    if user_input := st.chat_input("What would you like to research?"):
        st.session_state.message_count += 1
        
        with st.chat_message("human"):
            st.markdown(user_input)
        
        with st.chat_message("ai"):
            with st.spinner("🔍 Researching..."):
                response = agent_executor.invoke({
                    "input": user_input,
                    "chat_history": msgs.messages
                })
                st.markdown(response["output"])
                msgs.add_user_message(user_input)
                msgs.add_ai_message(response["output"])

# --- SQL ASSISTANT AGENT ---
elif chat_mode == "🗄️ SQL Assistant":
    st.markdown("### 🗄️ SQL Query Assistant")
    st.markdown("*Generate SQL queries from natural language and get query explanations!*")
    st.markdown("---")
    
    msgs = StreamlitChatMessageHistory(key="sql_chat_history")
    tools = create_sql_tools()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert SQL database assistant.
        Help users write efficient SQL queries and explain database concepts.
        Provide queries with proper formatting and best practices.
        Format SQL queries in code blocks."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    if len(msgs.messages) == 0:
        msgs.add_ai_message("🗄️ Hi! I'm your SQL assistant. I can help you:\n- Generate SQL queries from descriptions\n- Explain complex queries\n- Optimize query performance\n\nWhat SQL task can I help with?")
    
    for msg in msgs.messages:
        with st.chat_message(msg.type):
            st.markdown(msg.content)
    
    if user_input := st.chat_input("Describe your SQL query need..."):
        st.session_state.message_count += 1
        
        with st.chat_message("human"):
            st.markdown(user_input)
        
        with st.chat_message("ai"):
            with st.spinner("💾 Processing..."):
                response = agent_executor.invoke({
                    "input": user_input,
                    "chat_history": msgs.messages
                })
                st.markdown(response["output"])
                msgs.add_user_message(user_input)
                msgs.add_ai_message(response["output"])

# --- CREATIVE WRITER AGENT ---
elif chat_mode == "🎨 Creative Writer":
    st.markdown("### 🎨 Creative Writing Assistant")
    st.markdown("*Generate creative content including stories, poems, articles, and more!*")
    st.markdown("---")
    
    msgs = StreamlitChatMessageHistory(key="creative_chat_history")
    
    creative_llm = get_llm(api_key, temperature=0.7)  # Higher temperature for creativity
    
    if len(msgs.messages) == 0:
        msgs.add_ai_message("✨ Hello creative soul! I can help you with:\n- Writing stories and narratives\n- Crafting poems and lyrics\n- Creating blog posts and articles\n- Brainstorming creative ideas\n\nWhat shall we create today?")
    
    for msg in msgs.messages:
        with st.chat_message(msg.type):
            st.markdown(msg.content)
    
    if user_input := st.chat_input("What would you like to create?"):
        st.session_state.message_count += 1
        
        with st.chat_message("human"):
            st.markdown(user_input)
        
        with st.chat_message("ai"):
            with st.spinner("✍️ Creating..."):
                response = creative_llm.invoke(user_input)
                st.markdown(response.content)
                msgs.add_user_message(user_input)
                msgs.add_ai_message(response.content)

# ═════════════════════════════════════════════
# FOOTER
# ═════════════════════════════════════════════

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: white; padding: 20px;'>
    <p>🤖 Powered by Google Gemini AI | Built with Streamlit & LangChain</p>
    <p>Made with ❤️ for AI Enthusiasts</p>
</div>
""", unsafe_allow_html=True)
