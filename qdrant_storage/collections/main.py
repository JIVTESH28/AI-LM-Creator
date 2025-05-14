import streamlit as st
import os
import base64
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.pydantic_v1 import BaseModel, Field
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.utils.function_calling import convert_to_openai_function
from typing import List, Dict, Optional
import time
# Vector search imports
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder

# PDF generation imports
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.units import inch
import requests
from io import BytesIO
import tempfile
from dotenv import load_dotenv
import json
import textwrap

# Load environment variables
load_dotenv()

# Model schema classes
class TopicList(BaseModel):
    """List of educational topics for lectures."""
    topics: List[str] = Field(description="List of educational topics suitable for lectures")

class LectureContent(BaseModel):
    """Detailed lecture content for a specific topic."""
    title: str = Field(description="Title of the lecture")
    content: str = Field(description="Markdown formatted lecture content with headings, subheadings, examples, and explanations")

class ImageInfo(BaseModel):
    """Information about educational images related to a topic."""
    image_urls: List[str] = Field(description="URLs of relevant educational images")
    descriptions: List[str] = Field(description="Descriptions of each image")

# Vector database class for RAG
class VectorDatabase:
    def __init__(self, collection_name="lectureVectorDb", url="http://localhost:6333"):
        self.client = QdrantClient(url=url, prefer_grpc=False)
        self.collection_name = collection_name
        
        # Initialize embeddings model
        try:
            device = 'mps' if hasattr(st, 'device_selector') and st.device_selector() == 'mps' else 'cpu'
        except:
            device = 'cpu'
            
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={'device': device}, 
            encode_kwargs={'normalize_embeddings': False}
        )
        
        # Try to initialize the database, create if not exists
        try:
            self.database = Qdrant(
                client=self.client,
                collection_name=collection_name,
                embeddings=self.embeddings
            )
        except Exception as e:
            # Collection might not exist yet
            st.warning(f"Vector database collection not initialized yet. Please ingest documents first.")
            self.database = None
    
    def retrieve_chunks(self, query, k=5):
        if not self.database:
            return []
            
        docs = self.database.similarity_search_with_score(query=query, k=k)
        # Extract source and content information
        results = []
        for doc, score in docs:
            # Extract source document info from metadata
            source = doc.metadata.get('source', 'Unknown source')
            results.append((doc.page_content, score, source))
        
        return results

# Cyclic RAG class
class CyclicRAG:
    def __init__(self, vector_db, llm):
        self.vector_db = vector_db
        self.llm = llm
        self.memory = []  # Stores past queries
        
        # Try to initialize reranker
        try:
            self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        except Exception as e:
            st.warning(f"Could not initialize reranker: {e}. Using basic similarity search.")
            self.reranker = None
            
        self.retrieval_details = []  # Store retrieval details for UI display

    def rerank_chunks(self, query, retrieved_chunks):
        if not self.reranker or not retrieved_chunks:
            return [chunk for chunk, _, _ in retrieved_chunks[:5]]
            
        # Create pairs for reranking
        pairs = [(query, chunk) for chunk, _, _ in retrieved_chunks]
        scores = self.reranker.predict(pairs)
        
        # Rerank with scores
        ranked = sorted(zip(retrieved_chunks, scores), key=lambda x: x[1], reverse=True)
        ranked_chunks = [(chunk, source) for (chunk, _, source), _ in ranked[:5]]
        
        # Store retrieval details for UI
        self.retrieval_details = []
        for (content, _, source), score in ranked[:5]:
            self.retrieval_details.append({
                'content': content,
                'source': source,
                'score': score
            })
        
        # Return just the content for the summary generation
        return [content for content, _ in ranked_chunks]
    
    def summarize_queries(self):
        if len(self.memory) < 2:
            return self.memory[-1] if self.memory else ""
            
        summary_prompt = "Summarize the following user queries into a concise query that captures the essential information needs:\n" + "\n".join(self.memory)
        
        response = self.llm.invoke([
            SystemMessage(content="You are a query summarizer. Your task is to distill multiple queries into a single, focused query."),
            HumanMessage(content=summary_prompt)
        ])
        
        return response.content
    
    def generate_context_prompt(self, system_prompt, query, retrieved_chunks):
        context = "\n\n".join([f"DOCUMENT CHUNK:\n{chunk}" for chunk in retrieved_chunks])
        full_prompt = f"{system_prompt}\n\nUser Query: {query}\n\nRetrieved Information:\n{context}\n\nAnswer:"
        return full_prompt
    
    def process_query(self, query):
        self.memory.append(query)
        refined_query = self.summarize_queries()
        
        system_prompt = """You are an educational content creator specializing in creating comprehensive lecture materials.
        Use the retrieved information to create detailed, well-structured lecture notes on the given topic.
        Include definitions, explanations, examples, and practical applications.
        Structure the content with clear Markdown headings, subheadings, and bullet points for easy comprehension.
        Cite sources where appropriate."""
        
        retrieved_chunks = self.vector_db.retrieve_chunks(refined_query)
        
        if not retrieved_chunks:
            return "No relevant information found in the knowledge base. Please try another query or ingest relevant documents."
            
        ranked_chunks = self.rerank_chunks(refined_query, retrieved_chunks)
        prompt = self.generate_context_prompt(system_prompt, refined_query, ranked_chunks)
        
        response = self.llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Create comprehensive lecture material on '{query}' using these retrieved documents:\n\n{' '.join(ranked_chunks)}")
        ])
        
        return response.content

# Main lecture material creator class
class LectureMaterialCreator:
    def __init__(self, model_name="gpt-4o"):
        """
        Initialize the lecture material creator with the specified model.
        
        Args:
            model_name (str): The model to use (gpt-4o, etc.)
        """
        self.model_name = model_name
        self.personas = self._load_personas()
        
        # Check for necessary API keys
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")
        if not self.tavily_api_key:
            st.warning("TAVILY_API_KEY environment variable not found. Search functionality will be limited.")
            self.search_tool = None
        else:
            # Initialize search tool
            self.search_tool = TavilySearchResults(k=5)
        
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            st.error("OPENAI_API_KEY environment variable not found. Please set it in your .env file.")
            raise ValueError("OPENAI_API_KEY environment variable not found.")
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=self.model_name,
            temperature=0.7,
            api_key=self.openai_api_key
        )
        
    def _load_personas(self):
        """
        Load personas with system prompts.
        
        Returns:
            dict: A dictionary of personas with system prompts
        """
        # Hardcoded personas similar to the original code
        return {
            "researcher": {
                "system_prompt": "You are a specialized educational researcher. Your task is to find relevant educational topics for a given subject. Focus on gathering comprehensive information from reliable educational websites. When searching, prioritize finding structured learning paths, course outlines, or syllabi."
            },
            "topic_finder": {
                "system_prompt": "You are a topic finder specialized in educational content. Given a subject area, extract a structured list of topics that would be appropriate for a comprehensive lecture. Format the output as a clean, numbered list with main topics and subtopics where appropriate."
            },
            "content_creator": {
                "system_prompt": "You are an educational content creator specializing in creating comprehensive lecture materials. Your task is to create detailed, well-structured lecture notes on the given topic. Include definitions, explanations, examples, and practical applications. Structure the content with clear headings, subheadings, and bullet points for easy comprehension."
            },
            "rag_content_creator": {
                "system_prompt": "You are an educational content creator specializing in creating comprehensive lecture materials based on provided document sources. Your task is to create detailed, well-structured lecture notes on the given topic using only the information provided. Include definitions, explanations, examples, and practical applications from the source material. Structure the content with clear headings, subheadings, and bullet points for easy comprehension. Cite sources appropriately."
            },
            "editor": {
                "system_prompt": "You are an editor for educational materials. Your task is to review and refine lecture content for clarity, accuracy, and comprehensiveness. Ensure the content flows logically, uses consistent terminology, and is pitched at the appropriate educational level. Suggest additions or modifications to improve the overall quality of the material."
            }
        }
    
    def create_topic_finder_agent(self):
        """
        Create an agent that finds educational topics for a given subject.
        
        Returns:
            AgentExecutor: A topic finder agent
        """
        # Define the system message with the researcher prompt
        system_message = self.personas["researcher"]["system_prompt"] + "\n" + self.personas["topic_finder"]["system_prompt"]
        
        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad")
        ])
        
        # Define tools
        tools = []
        if self.search_tool:
            tools.append(Tool(
                name="search",
                func=self.search_tool.invoke,
                description="Search for information on educational topics. Input should be a search query."
            ))
        
        # Create agent
        agent = create_openai_tools_agent(self.llm, tools, prompt)
        
        # Create agent executor
        return AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            return_intermediate_steps=True
        )
    
    def create_content_creator_agent(self):
        """
        Create an agent that generates lecture content.
        
        Returns:
            AgentExecutor: A content creator agent
        """
        # Define the system message with the content creator prompt
        system_message = self.personas["content_creator"]["system_prompt"]
        
        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad")
        ])
        
        # Define tools
        tools = []
        if self.search_tool:
            tools.append(Tool(
                name="search",
                func=self.search_tool.invoke,
                description="Search for information on the lecture topic. Input should be a search query."
            ))
        
        # Create agent
        agent = create_openai_tools_agent(self.llm, tools, prompt)
        
        # Create agent executor
        return AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            return_intermediate_steps=True
        )
    
    def create_editor_agent(self):
        """
        Create an agent that edits and refines lecture content.
        
        Returns:
            AgentExecutor: An editor agent
        """
        # Define the system message with the editor prompt
        system_message = self.personas["editor"]["system_prompt"]
        
        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad")
        ])
        
        # Create agent
        agent = create_openai_tools_agent(self.llm, tools=[], prompt=prompt)
        
        # Create agent executor
        return AgentExecutor(
            agent=agent,
            tools=[],
            verbose=True,
            return_intermediate_steps=True
        )

# Document ingestion function
def ingest_documents(data_folder="data", collection_name="lectureVectorDb", url="http://localhost:6333"):
    """
    Ingest all PDF documents from a folder into Qdrant
    """
    # Check if the data folder exists
    if not os.path.exists(data_folder):
        raise FileNotFoundError(f"Data folder '{data_folder}' not found.")
    
    # Use directory loader to load all PDFs
    loader = DirectoryLoader(
        data_folder,
        glob="**/*.pdf",  # Load all PDFs including those in subfolders
        loader_cls=PyPDFLoader,
        show_progress=True
    )
    
    documents = loader.load()
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    
    texts = text_splitter.split_documents(documents)
    
    # Ensure each chunk has source information
    for chunk in texts:
        # If source is not in metadata, add filename
        if 'source' not in chunk.metadata:
            chunk.metadata['source'] = os.path.basename(chunk.metadata.get('source', 'Unknown'))
    
    # Initialize embeddings
    try:
        device = 'mps' if hasattr(st, 'device_selector') and st.device_selector() == 'mps' else 'cpu'
    except:
        device = 'cpu'
        
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {'device': device}
    encode_kwargs = {'normalize_embeddings': False}
    
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    
    # Initialize and populate Qdrant
    qdrant = Qdrant.from_documents(
        texts,
        embeddings,
        url=url,
        prefer_grpc=False,
        collection_name=collection_name
    )
    
    return len(texts)

# Function to find topics using traditional search or RAG
def find_topics_for_subject(creator, subject, vector_db=None):
    """
    Find topics for the given subject using the topic finder agent or RAG.
    
    Args:
        creator (LectureMaterialCreator): The lecture material creator
        subject (str): The subject to find topics for
        vector_db (VectorDatabase, optional): Vector database for RAG
        
    Returns:
        list: A list of topics for the subject
    """
    # Create the topic finder agent
    topic_finder = creator.create_topic_finder_agent()
    
    # If we have a vector database, use it to supplement the search
    context = ""
    if vector_db and vector_db.database:
        retrieved_chunks = vector_db.retrieve_chunks(f"key educational topics in {subject}", k=3)
        if retrieved_chunks:
            context = "\n\n".join([chunk for chunk, _, _ in retrieved_chunks])
            context = f"\nHere's some relevant information from our knowledge base:\n{context}"
    
    # Run the agent
    result = topic_finder.invoke({
        "input": f"Research and find 10-15 key educational topics for {subject} that would make good lectures. Format the output as a numbered list.{context}",
        "chat_history": []
    })
    
    # Extract topics from the result
    topics = []
    for line in result["output"].split('\n'):
        line = line.strip()
        if line and (line[0].isdigit() or line.startswith("- ")):
            # Remove any numbering or bullet points
            cleaned_line = line.split(".", 1)[-1].strip() if "." in line else line
            cleaned_line = cleaned_line[2:].strip() if cleaned_line.startswith("- ") else cleaned_line
            topics.append(cleaned_line)
    
    return topics

# Function to create lecture material using traditional search or RAG
def create_lecture_material(creator, topic, vector_db=None):
    """
    Create lecture material for the given topic.
    
    Args:
        creator (LectureMaterialCreator): The lecture material creator
        topic (str): The topic to create material for
        vector_db (VectorDatabase, optional): Vector database for RAG
        
    Returns:
        str: The lecture content
    """
    if vector_db and vector_db.database:
        # If we have a RAG database, use it to create lecture material
        cyclic_rag = CyclicRAG(vector_db, creator.llm)
        content = cyclic_rag.process_query(f"Create comprehensive lecture material on '{topic}'")
        
        # Get the retrieval details for debugging/transparency
        retrieval_details = cyclic_rag.retrieval_details
        
        # Edit content
        editor = creator.create_editor_agent()
        edited_result = editor.invoke({
            "input": f"Review and refine the following lecture material on '{topic}' for clarity, accuracy, and comprehensiveness:\n\n{content}",
            "chat_history": []
        })
        
        return edited_result["output"], retrieval_details
    else:
        # Create the content creator agent
        content_creator = creator.create_content_creator_agent()
        
        # Create the editor agent
        editor = creator.create_editor_agent()
        
        # Generate lecture content
        content_result = content_creator.invoke({
            "input": f"Create comprehensive lecture material on '{topic}'. Include definitions, explanations, examples, and practical applications. Structure with clear headings and subheadings using Markdown formatting.",
            "chat_history": []
        })
        
        # Edit content
        edited_result = editor.invoke({
            "input": f"Review and refine the following lecture material on '{topic}' for clarity, accuracy, and comprehensiveness:\n\n{content_result['output']}",
            "chat_history": []
        })
        
        return edited_result["output"], []

# Function to create PDF
def create_pdf(content, topic, image_urls=None, image_descriptions=None):
    """
    Create a PDF from the lecture material and images.
    
    Args:
        content (str): The lecture material content
        topic (str): The topic of the lecture material
        image_urls (list, optional): List of image URLs to include
        image_descriptions (list, optional): List of image descriptions
        
    Returns:
        bytes: The PDF file as bytes
    """
    # Initialize empty lists if not provided
    if image_urls is None:
        image_urls = []
    if image_descriptions is None:
        image_descriptions = []
    
    # Create a temporary file for the PDF
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        temp_filename = tmp.name
    
    # Create the PDF
    doc = SimpleDocTemplate(temp_filename, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Add title
    title_style = styles['Heading1']
    story.append(Paragraph(f"Lecture Material: {topic}", title_style))
    story.append(Spacer(1, 12))
    
    # Add content
    content_lines = content.split('\n')
    i = 0
    while i < len(content_lines):
        line = content_lines[i].strip()
        if not line:
            i += 1
            continue
            
        if line.startswith('# '):
            # Heading 1
            style = styles['Heading1']
            text = line[2:].strip()
        elif line.startswith('## '):
            # Heading 2
            style = styles['Heading2']
            text = line[3:].strip()
        elif line.startswith('### '):
            # Heading 3
            style = styles['Heading3']
            text = line[4:].strip()
        elif line.startswith('```'):
            # Code block
            code_lines = []
            i += 1
            while i < len(content_lines) and not content_lines[i].startswith('```'):
                code_lines.append(content_lines[i])
                i += 1
            code_text = '<font face="Courier" size="9">' + '<br/>'.join(code_lines) + '</font>'
            style = styles['Normal']
            text = code_text
        else:
            # Normal text
            style = styles['Normal']
            text = line
        
        story.append(Paragraph(text, style))
        story.append(Spacer(1, 6))
        i += 1
    
    # Add images
    for i, (url, desc) in enumerate(zip(image_urls, image_descriptions)):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                img_io = BytesIO(response.content)
                img = Image(img_io, width=6*inch, height=4*inch)
                story.append(img)
                story.append(Spacer(1, 12))
                story.append(Paragraph(f"Figure {i+1}: {desc}", styles['Caption']))
                story.append(Spacer(1, 24))
        except Exception as e:
            print(f"Error adding image {url}: {e}")
    
    # Build the PDF
    doc.build(story)
    
    # Read the PDF file
    with open(temp_filename, "rb") as f:
        pdf_bytes = f.read()
    
    # Clean up the temporary file
    os.unlink(temp_filename)
    
    return pdf_bytes

def main():
    st.set_page_config(
        page_title="AI Lecture Material Creator",
        page_icon="📚",
        layout="wide"
    )
    
    # Initialize session state for navigation
    if 'page' not in st.session_state:
        st.session_state.page = 'creator'  # Default to creator page
    
    # Sidebar navigation
    st.sidebar.title("AI Lecture Creator 📚")
    page = st.sidebar.radio("Navigation", ["Create Lecture Material", "Ingest Documents"])
    
    if page == "Create Lecture Material":
        st.session_state.page = 'creator'
    else:
        st.session_state.page = 'ingest'
    
    # Common configuration
    st.sidebar.header("Configuration")
    collection_name = st.sidebar.text_input("Qdrant Collection Name", "lectureVectorDb")
    qdrant_url = st.sidebar.text_input("Qdrant URL", "http://localhost:6333")
    
    # Initialize vector database
    try:
        vector_db = VectorDatabase(collection_name=collection_name, url=qdrant_url)
    except Exception as e:
        st.sidebar.warning(f"Could not connect to vector database: {e}")
        vector_db = None
    
    # Display appropriate page based on navigation
    if st.session_state.page == 'creator':
        display_creator_page(collection_name, qdrant_url, vector_db)
    else:
        display_ingest_page(collection_name, qdrant_url)

def display_creator_page(collection_name, qdrant_url, vector_db=None):
    st.title("AI Lecture Material Creator")
    st.write("Create comprehensive lecture materials for any subject using AI and RAG.")
    
    # Check if documents are available in the vector database
    use_rag = False
    if vector_db and vector_db.database:
        # Try a simple query to check if documents exist
        test_chunks = vector_db.retrieve_chunks("test query", k=1)
        use_rag = len(test_chunks) > 0
    
    if use_rag:
        st.success("📚 Documents detected in the vector database! RAG mode is active.")
    else:
        st.info("No documents detected in the vector database. Using standard search mode.")
    
    # Input fields
    subject = st.text_input("Enter the subject (e.g., Python Programming, Machine Learning):")
    model_options = ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]
    model_name = st.selectbox("Select AI model:", model_options)
    
    if st.button("Generate Topics"):
        if subject:
            with st.spinner(f"Finding topics for {subject}..."):
                try:
                    creator = LectureMaterialCreator(model_name=model_name)
                    topics = find_topics_for_subject(creator, subject, vector_db)
                    
                    if topics:
                        st.session_state.topics = topics
                        st.session_state.creator = creator
                    else:
                        st.error("No topics were found. Please try a different subject or model.")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            st.error("Please enter a subject.")
    
    # Display topics if they exist in session state
    if "topics" in st.session_state:
        st.subheader("Select a Topic to Create Lecture Material")
        selected_topic = st.selectbox("Topics:", st.session_state.topics)
        
        if st.button("Create Lecture Material"):
            with st.spinner(f"Creating lecture material for {selected_topic}..."):
                try:
                    content, retrieval_details = create_lecture_material(st.session_state.creator, selected_topic, vector_db)
                    
                    st.session_state.content = content
                    st.session_state.selected_topic = selected_topic
                    st.session_state.retrieval_details = retrieval_details
                    
                    # Create PDF
                    pdf_bytes = create_pdf(content, selected_topic)
                    st.session_state.pdf_bytes = pdf_bytes
                except Exception as e:
                    st.error(f"Error creating lecture material: {str(e)}")
    
    # Display content if it exists in session state
    if "content" in st.session_state:
        st.subheader(f"Lecture Material: {st.session_state.selected_topic}")
        
        # Display content
        st.markdown(st.session_state.content)
        
        # Display retrieval details if RAG was used
        if st.session_state.retrieval_details:
            with st.expander("View Sources Used (RAG)", expanded=False):
                st.subheader("Retrieved Document Chunks")
                for i, detail in enumerate(st.session_state.retrieval_details):
                    st.markdown(f"**Source {i+1}:** {detail['source']}")
                    st.text(textwrap.fill(detail['content'][:300] + "..." if len(detail['content']) > 300 else detail['content'], width=80))
                    st.markdown(f"Relevance Score: {detail['score']:.4f}")
                    st.divider()
        
        # Download button for PDF
        if "pdf_bytes" in st.session_state:
            st.download_button(
                label="Download Lecture Material as PDF",
                data=st.session_state.pdf_bytes,
                file_name=f"lecture_{st.session_state.selected_topic.replace(' ', '_')}.pdf",
                mime="application/pdf"
            )
def display_ingest_page(collection_name, qdrant_url):
    st.title("Document Ingestion for RAG")
    st.subheader("Add PDF documents to enhance lecture creation")
    
    # Input fields
    data_folder = st.text_input("Data Folder Path", "data")
    
    # Display current documents if folder exists
    if os.path.exists(data_folder):
        files = [f for f in os.listdir(data_folder) if f.endswith('.pdf')]
        if files:
            st.write(f"Found {len(files)} PDF files in '{data_folder}':")
            for file in files:
                st.write(f"- {file}")
        else:
            st.warning(f"No PDF files found in '{data_folder}'. Please add some PDF files to this folder.")
    else:
        st.warning(f"Folder '{data_folder}' not found. Please create this folder and add PDF files.")
        
        # Add button to create folder
        if st.button("Create data folder"):
            try:
                os.makedirs(data_folder)
                st.success(f"Created folder '{data_folder}'. Please add PDF files to this folder.")
            except Exception as e:
                st.error(f"Error creating folder: {e}")
    
    # File uploader for direct PDF uploads
    uploaded_files = st.file_uploader("Or upload PDF files directly", type="pdf", accept_multiple_files=True)
    
    if uploaded_files:
        if not os.path.exists(data_folder):
            try:
                os.makedirs(data_folder)
                st.success(f"Created folder '{data_folder}' for your uploaded files.")
            except Exception as e:
                st.error(f"Error creating folder: {e}")
                return
        
        # Save uploaded files to the data folder
        for uploaded_file in uploaded_files:
            file_path = os.path.join(data_folder, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"Saved {uploaded_file.name} to {data_folder}")
    
    if st.button("Start Ingestion"):
        try:
            if not os.path.exists(data_folder):
                st.error(f"Folder '{data_folder}' does not exist. Please create it first.")
                return
                
            files = [f for f in os.listdir(data_folder) if f.endswith('.pdf')]
            if not files:
                st.error(f"No PDF files found in '{data_folder}'. Please add some PDF files first.")
                return
                
            with st.spinner("Ingesting documents... This may take a while depending on the number of PDFs."):
                num_chunks = ingest_documents(data_folder, collection_name, qdrant_url)
                st.success(f"Ingestion complete! {num_chunks} chunks added to the vector database.")
                
                # Display file details
                st.write(f"Processed {len(files)} PDF files:")
                for file in files:
                    st.write(f"- {file}")
        except Exception as e:
            st.error(f"Error during document ingestion: {str(e)}")
            st.error("Make sure Qdrant server is running at the specified URL.")


def display_creator_page(collection_name, qdrant_url, vector_db=None):
    # Check if documents are available in the vector database
    use_rag = False
    if vector_db and vector_db.database:
        # Try a simple query to check if documents exist
        test_chunks = vector_db.retrieve_chunks("test query", k=1)
        use_rag = len(test_chunks) > 0
    
    # RAG status indicator
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("Create comprehensive lecture materials")
    
    with col2:
        if use_rag:
            st.success("📚 RAG Active")
        else:
            st.info("🔍 Search Only")
    
    # Input fields with better styling
    st.markdown("### Subject Selection")
    subject = st.text_input("Enter the subject (e.g., Python Programming, Machine Learning):")
    
    # Add example subjects for quick selection
    example_subjects = ["Machine Learning", "Web Development", "Physics", "Data Science", 
                     "History", "Mathematics", "Psychology", "Computer Science"]
    
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown("**Or select:**")
    with col2:
        selected_example = st.selectbox("Example subjects", [""] + example_subjects)
        if selected_example and not subject:
            subject = selected_example
    
    # Generate topics button with progress display
    if st.button("📋 Generate Topics", key="gen_topics_btn", help="Generate lecture topics for the selected subject"):
        if subject:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("Initializing topic generation...")
            progress_bar.progress(10)
            
            try:
                status_text.text("Creating AI model...")
                progress_bar.progress(30)
                
                # Get model name from session state or use default
                model_name = st.session_state.get('model_name', "gpt-4o")
                creator = LectureMaterialCreator(model_name=model_name)
                
                status_text.text(f"Finding topics for {subject}...")
                progress_bar.progress(50)
                
                topics = find_topics_for_subject(creator, subject, vector_db)
                
                progress_bar.progress(100)
                status_text.text("Topics generated successfully!")
                
                if topics:
                    st.session_state.topics = topics
                    st.session_state.creator = creator
                    st.session_state.current_subject = subject
                else:
                    st.error("No topics were found. Please try a different subject or model.")
            except Exception as e:
                st.error(f"Error: {str(e)}")
            finally:
                # Clean up progress indicators
                progress_bar.empty()
                status_text.empty()
        else:
            st.error("Please enter a subject.")
    
    # Display topics if they exist in session state
    if "topics" in st.session_state:
        st.markdown(f"### Topics for '{st.session_state.current_subject}'")
        
        # Calculate 2-3 columns based on number of topics
        num_topics = len(st.session_state.topics)
        num_cols = 2 if num_topics <= 10 else 3
        topics_per_col = (num_topics + num_cols - 1) // num_cols
        
        # Container with light background for topic display
        with st.container():
            st.markdown("""
            <style>
            .topic-container {
                background-color: #f8f9fa;
                padding: 15px;
                border-radius: 5px;
                margin-bottom: 20px;
            }
            </style>
            <div class="topic-container">
            """, unsafe_allow_html=True)
            
            # Create columns
            cols = st.columns(num_cols)
            
            # Distribute topics among columns
            for i, topic in enumerate(st.session_state.topics):
                col_index = i // topics_per_col
                cols[col_index].markdown(f"- {topic}")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Topic selection for lecture creation
        st.subheader("Create Lecture Material")
        selected_topic = st.selectbox("Select a topic:", st.session_state.topics)
        
        # Custom RAG option
        use_custom_rag = False
        if use_rag:
            use_custom_rag = st.checkbox("Use RAG for content generation", value=True)
        
        # Create lecture material
        if st.button("🔍 Create Lecture Material", key="create_lecture_btn"):
            if selected_topic:
                # Create progress indicators
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    status_text.text("Starting lecture material creation...")
                    progress_bar.progress(10)
                    
                    status_text.text("Researching content...")
                    progress_bar.progress(30)
                    
                    # Use RAG if available and selected
                    rag_vector_db = vector_db if use_rag and use_custom_rag else None
                    
                    status_text.text("Generating comprehensive content...")
                    progress_bar.progress(60)
                    
                    content, retrieval_details = create_lecture_material(
                        st.session_state.creator, 
                        selected_topic, 
                        rag_vector_db
                    )
                    
                    status_text.text("Creating PDF document...")
                    progress_bar.progress(85)
                    
                    # Create PDF
                    pdf_bytes = create_pdf(content, selected_topic)
                    
                    # Store in session state
                    st.session_state.content = content
                    st.session_state.selected_topic = selected_topic
                    st.session_state.retrieval_details = retrieval_details
                    st.session_state.pdf_bytes = pdf_bytes
                    
                    progress_bar.progress(100)
                    status_text.text("Lecture material created successfully!")
                    
                except Exception as e:
                    st.error(f"Error creating lecture material: {str(e)}")
                finally:
                    # Clean up progress indicators after short delay
                    time.sleep(1)
                    progress_bar.empty()
                    status_text.empty()
    
    # Display content if it exists in session state
    if "content" in st.session_state:
        st.markdown("---")
        st.subheader(f"Lecture Material: {st.session_state.selected_topic}")
        
        # Tabs for different views
        tab1, tab2 = st.tabs(["Content", "Sources"])
        
        # Content tab
        with tab1:
            st.markdown(st.session_state.content)
            
            # Download button for PDF
            if "pdf_bytes" in st.session_state:
                st.download_button(
                    label="📥 Download as PDF",
                    data=st.session_state.pdf_bytes,
                    file_name=f"lecture_{st.session_state.selected_topic.replace(' ', '_')}.pdf",
                    mime="application/pdf",
                    key="download_pdf_btn"
                )
                
                # Copy to clipboard button
                st.button("📋 Copy to Clipboard", 
                          key="copy_content_btn",
                          help="Copy lecture content to clipboard",
                          on_click=lambda: st.write('<script>navigator.clipboard.writeText(`' + 
                                                   st.session_state.content.replace('`', '\\`') + 
                                                   '`);</script>', unsafe_allow_html=True))
        
        # Sources tab
        with tab2:
            if st.session_state.retrieval_details:
                st.subheader("Retrieved Document Chunks")
                for i, detail in enumerate(st.session_state.retrieval_details):
                    with st.expander(f"Source {i+1}: {detail['source']}"):
                        st.text(textwrap.fill(detail['content'][:300] + "..." 
                                            if len(detail['content']) > 300 
                                            else detail['content'], width=80))
                        st.markdown(f"Relevance Score: {detail['score']:.4f}")
            else:
                st.info("This content was generated using general knowledge and web search, not from specific document sources.")

def display_about_page():
    st.header("About AI Lecture Material Creator")
    
    st.markdown("""
    This application uses AI and Retrieval Augmented Generation (RAG) to create comprehensive lecture materials on any subject.
    
    ### Key Features:
    - Generate lecture topics for any subject
    - Create detailed lecture materials with proper structure
    - Download lecture materials as PDF
    - Use your own documents to enhance content creation (RAG)
    - Support for multiple AI models
    
    ### How to Use:
    1. **Create Lecture Material**: Enter a subject, generate topics, and create lecture content
    2. **Ingest Documents**: Upload PDF documents to enhance the knowledge base
    
    ### Technology Stack:
    - **Frontend**: Streamlit
    - **AI**: OpenAI GPT models
    - **RAG**: Qdrant Vector Database, HuggingFace Embeddings
    - **PDF Generation**: ReportLab
    
    ### About RAG:
    Retrieval Augmented Generation enhances AI-generated content by retrieving relevant information from your document library before generating responses, leading to more accurate and informative content.
    """)
    
    # FAQ section
    with st.expander("Frequently Asked Questions"):
        st.markdown("""
        **Q: Do I need to set up Qdrant?**
        
        A: Yes, for the RAG functionality to work, you need to have a Qdrant server running. The default URL is http://localhost:6333.
        
        **Q: What types of documents can I ingest?**
        
        A: Currently, the system supports PDF documents only.
        
        **Q: How do I get the best results?**
        
        A: For best results:
        - Use specific subject descriptions
        - Ingest high-quality reference materials
        - Use GPT-4o for more comprehensive content
        - Try different topics if you're not satisfied with the initial results
        """)
    
    # Display version information
    st.sidebar.markdown("---")
    st.sidebar.info("Version 2.0 - RAG Enhanced")
    st.sidebar.markdown("Made with ❤️ using Streamlit and AI")

if __name__ == "__main__":
    main()