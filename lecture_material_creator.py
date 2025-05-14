import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.pydantic_v1 import BaseModel, Field
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.utils.function_calling import convert_to_openai_function
from typing import List, Dict, Optional
import yaml
import os
import base64
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

# Load environment variables
load_dotenv()

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

class LectureMaterialCreator:
    def __init__(self, model_name="gpt-4o"):
        """
        Initialize the lecture material creator with the specified model.
        
        Args:
            model_name (str): The model to use (gpt-4o, claude-3-opus, etc.)
        """
        self.model_name = model_name
        self.personas = self._load_personas()
        
        # Check for necessary API keys
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")
        if not self.tavily_api_key:
            raise ValueError("TAVILY_API_KEY environment variable not found. Please set it in your .env file.")
        
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        
        # Initialize search tool
        self.search_tool = TavilySearchResults(k=5)
        
        # Initialize LLM based on model name
        self.llm = self._initialize_llm()
        
    def _initialize_llm(self):
        """Initialize the appropriate LLM based on model name."""
        if self.model_name.startswith("gpt"):
            if not self.openai_api_key:
                raise ValueError("OPENAI_API_KEY environment variable not found. Please set it in your .env file.")
            return ChatOpenAI(
                model=self.model_name,
                temperature=0.7,
                api_key=self.openai_api_key
            )
        elif self.model_name.startswith("claude"):
            if not self.anthropic_api_key:
                raise ValueError("ANTHROPIC_API_KEY environment variable not found. Please set it in your .env file.")
            return ChatAnthropic(
                model=self.model_name,
                temperature=0.7,
                anthropic_api_key=self.anthropic_api_key
            )
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

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
            "image_finder": {
                "system_prompt": "You are an image curator for educational materials. Find relevant, high-quality images that illustrate the key concepts for the given topic. Focus on diagrams, charts, and visual representations that enhance understanding. For each image, provide a brief description and relevance to the topic."
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
        tools = [Tool(
            name="search",
            func=self.search_tool.invoke,
            description="Search for information on educational topics. Input should be a search query."
        )]
        
        # Create function to parse topics
        topic_list_function = convert_to_openai_function(TopicList)
        
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
        tools = [Tool(
            name="search",
            func=self.search_tool.invoke,
            description="Search for information on the lecture topic. Input should be a search query."
        )]
        
        # Create agent
        agent = create_openai_tools_agent(self.llm, tools, prompt)
        
        # Create agent executor
        return AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            return_intermediate_steps=True
        )
    
    def create_image_finder_agent(self):
        """
        Create an agent that finds relevant images for the lecture topic.
        
        Returns:
            AgentExecutor: An image finder agent
        """
        # Define the system message with the image finder prompt
        system_message = self.personas["image_finder"]["system_prompt"]
        
        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad")
        ])
        
        # Define tools
        tools = [Tool(
            name="search",
            func=self.search_tool.invoke,
            description="Search for educational images and diagrams. Input should be a search query."
        )]
        
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


def find_topics_for_subject(creator, subject):
    """
    Find topics for the given subject using the topic finder agent.
    
    Args:
        creator (LectureMaterialCreator): The lecture material creator
        subject (str): The subject to find topics for
        
    Returns:
        list: A list of topics for the subject
    """
    # Create the topic finder agent
    topic_finder = creator.create_topic_finder_agent()
    
    # Run the agent
    result = topic_finder.invoke({
        "input": f"Research and find 10-15 key educational topics for {subject} that would make good lectures. Format the output as a numbered list.",
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


def create_lecture_material(creator, topic):
    """
    Create lecture material for the given topic.
    
    Args:
        creator (LectureMaterialCreator): The lecture material creator
        topic (str): The topic to create material for
        
    Returns:
        tuple: (lecture_content, image_urls)
    """
    # Create the content creator agent
    content_creator = creator.create_content_creator_agent()
    
    # Create the image finder agent
    image_finder = creator.create_image_finder_agent()
    
    # Create the editor agent
    editor = creator.create_editor_agent()
    
    # Generate lecture content
    content_result = content_creator.invoke({
        "input": f"Create comprehensive lecture material on '{topic}'. Include definitions, explanations, examples, and practical applications. Structure with clear headings and subheadings using Markdown formatting.",
        "chat_history": []
    })
    
    # Find images
    image_result = image_finder.invoke({
        "input": f"Find 2-3 relevant educational diagrams or images for the topic: {topic}. For each image, provide the URL and a brief description.",
        "chat_history": []
    })
    
    # Edit content
    edited_result = editor.invoke({
        "input": f"Review and refine the following lecture material on '{topic}' for clarity, accuracy, and comprehensiveness:\n\n{content_result['output']}",
        "chat_history": []
    })
    
    # Extract image URLs from the image finding result
    image_urls = []
    image_descriptions = []
    
    for line in image_result["output"].split('\n'):
        line = line.strip()
        if "http" in line and ("png" in line.lower() or "jpg" in line.lower() or "jpeg" in line.lower() or "svg" in line.lower()):
            # Extract the URL from the line
            url_start = line.find("http")
            url_end = line.find(" ", url_start) if line.find(" ", url_start) > 0 else len(line)
            url = line[url_start:url_end].rstrip(',.;:')
            image_urls.append(url)
            
            # Try to extract description
            if ":" in line[url_end:]:
                desc = line[url_end:].split(":", 1)[1].strip()
                image_descriptions.append(desc)
            else:
                image_descriptions.append(f"Image related to {topic}")
    
    return edited_result["output"], image_urls, image_descriptions


def create_pdf(content, image_urls, image_descriptions, topic):
    """
    Create a PDF from the lecture material and images.
    
    Args:
        content (str): The lecture material content
        image_urls (list): List of image URLs to include
        image_descriptions (list): List of image descriptions
        topic (str): The topic of the lecture material
        
    Returns:
        bytes: The PDF file as bytes
    """
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
    st.title("Lecture Material Creator")
    st.write("Create comprehensive lecture materials for any subject using LangChain and AI.")
    
    # Input fields
    subject = st.text_input("Enter the subject (e.g., Python Programming, Machine Learning):")
    model_options = ["gpt-4o", "gpt-3.5-turbo", "claude-3-opus", "claude-3-sonnet"]
    model_name = st.selectbox("Select AI model:", model_options)
    
    if st.button("Generate Topics"):
        if subject:
            with st.spinner(f"Finding topics for {subject}..."):
                try:
                    creator = LectureMaterialCreator(model_name=model_name)
                    topics = find_topics_for_subject(creator, subject)
                    
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
                    content, image_urls, image_descriptions = create_lecture_material(st.session_state.creator, selected_topic)
                    
                    st.session_state.content = content
                    st.session_state.image_urls = image_urls
                    st.session_state.image_descriptions = image_descriptions
                    st.session_state.selected_topic = selected_topic
                    
                    # Create PDF
                    pdf_bytes = create_pdf(content, image_urls, image_descriptions, selected_topic)
                    st.session_state.pdf_bytes = pdf_bytes
                except Exception as e:
                    st.error(f"Error creating lecture material: {str(e)}")
    
    # Display content if it exists in session state
    if "content" in st.session_state:
        st.subheader(f"Lecture Material: {st.session_state.selected_topic}")
        
        # Display content
        st.markdown(st.session_state.content)
        
        # Display images
        if st.session_state.image_urls:
            st.subheader("Related Images")
            for i, (url, desc) in enumerate(zip(st.session_state.image_urls, st.session_state.image_descriptions)):
                try:
                    st.image(url, caption=f"Figure {i+1}: {desc}")
                except Exception as e:
                    st.error(f"Error displaying image: {e}")
        
        # Download button for PDF
        if "pdf_bytes" in st.session_state:
            st.download_button(
                label="Download Lecture Material as PDF",
                data=st.session_state.pdf_bytes,
                file_name="lecture_material.pdf",
                mime="application/pdf"
            )


if __name__ == "__main__":
    main()