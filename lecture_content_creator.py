"""
Core logic for creating lecture materials using AI agents.

This module defines the `LectureMaterialCreator` class, which orchestrates
AI agents (built with LangChain) to find educational topics, generate
lecture content, and find relevant images. It also includes helper
functions for these tasks and for creating PDF output of the materials.

Key Components:
- Pydantic models (`TopicList`, `LectureContent`, `ImageInfo`): Define data structures.
- `LectureMaterialCreator`: Manages API keys, LLM initialization, agent creation,
  and persona loading.
- Agent creation methods: Set up agents for specific tasks like topic finding,
  content creation, image searching, and editing.
- Helper functions:
    - `find_topics_for_subject`: Uses an agent to find topics.
    - `create_lecture_material`: Uses agents to generate content and find images.
    - `create_pdf`: Converts generated markdown and images into a PDF.
"""
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

import os
import base64
import requests
from io import BytesIO
import tempfile
from dotenv import load_dotenv
import json
import yaml
import markdown2
from weasyprint import HTML

# Load environment variables
load_dotenv()

# Pydantic models for structuring data, especially for agent outputs.
class TopicList(BaseModel):
    """
    A Pydantic model representing a list of educational topics.
    Used for structured output from the topic finding agent.
    """
    topics: List[str] = Field(description="List of educational topics suitable for lectures")

class LectureContent(BaseModel):
    """
    A Pydantic model for detailed lecture content.
    Used for structured output from the content creation agent.
    """
    title: str = Field(description="Title of the lecture")
    content: str = Field(description="Markdown formatted lecture content with headings, subheadings, examples, and explanations")

class ImageInfo(BaseModel):
    """
    A Pydantic model for information about educational images.
    Used for structured output from the image finding agent.
    """
    image_urls: List[str] = Field(description="URLs of relevant educational images")
    descriptions: List[str] = Field(description="Descriptions of each image")

class LectureMaterialCreator:
    """
    Orchestrates AI agents to create lecture materials.

    This class handles:
    - Initialization of the language model (LLM) based on the specified model name.
    - Loading of predefined agent personas from `prompts.yaml`.
    - Checking for necessary API keys (Tavily, OpenAI, Anthropic).
    - Creation of specialized agents for:
        - Finding educational topics.
        - Generating lecture content.
        - Finding relevant images.
        - Editing and refining content.
    """
    def __init__(self, model_name: str = "gpt-4o"):
        """
        Initializes the LectureMaterialCreator.

        Args:
            model_name (str): The name of the language model to use (e.g., "gpt-4o",
                              "claude-3-opus-20240229"). Defaults to "gpt-4o".

        Raises:
            ValueError: If required API keys (TAVILY_API_KEY) are not found in environment variables.
        """
        self.model_name = model_name
        self.personas = self._load_personas() # Load agent personalities and system prompts
        
        # Check for necessary API keys
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")
        if not self.tavily_api_key:
            raise ValueError("TAVILY_API_KEY environment variable not found. Please set it in your .env file.")
        
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        
        # Initialize search tool
        self.search_tool = TavilySearchResults(k=5)
        
        # Initialize LLM based on model name
        self.llm = self._initialize_llm() # Initialize the language model
        
    def _initialize_llm(self) -> ChatOpenAI | ChatAnthropic:
        """
        Initializes and returns the appropriate LangChain LLM instance based on `self.model_name`.

        Supports OpenAI (gpt-*) and Anthropic (claude-*) models.
        Checks for necessary API keys (OPENAI_API_KEY or ANTHROPIC_API_KEY) and
        raises a ValueError if a key is missing for the selected model type.

        Returns:
            Union[ChatOpenAI, ChatAnthropic]: An instance of the LangChain chat model.

        Raises:
            ValueError: If the model name is unsupported or if required API keys are missing.
        """
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

    def _load_personas(self) -> Dict[str, Dict[str, str]]:
        """
        Loads agent personas from the `prompts.yaml` file.

        Each persona typically includes a system prompt defining its role and behavior.
        Handles `FileNotFoundError` if `prompts.yaml` is missing and `yaml.YAMLError`
        for parsing issues, displaying errors via `st.error`.

        Returns:
            Dict[str, Dict[str, str]]: A dictionary where keys are persona names (e.g., "researcher")
                                       and values are dictionaries containing persona details
                                       (e.g., {"system_prompt": "prompt text"}).
                                       Returns an empty dictionary if loading fails.
        """
        # Load personas from prompts.yaml
        try:
            # Ensure prompts.yaml is in the same directory or an accessible path.
            with open("prompts.yaml", "r") as f:
                prompts_data = yaml.safe_load(f)
            return prompts_data.get("personas", {})
        except FileNotFoundError:
            st.error("prompts.yaml not found. Please ensure the file exists.")
            return {}
        except yaml.YAMLError as e:
            st.error(f"Error parsing prompts.yaml: {e}")
            return {}
    
    def create_topic_finder_agent(self) -> AgentExecutor:
        """
        Creates and returns an agent specialized in finding educational topics.

        This agent uses a combination of "researcher" and "topic_finder" personas
        and is equipped with a search tool. It's designed to take a subject
        and return a list of relevant lecture topics.

        Returns:
            AgentExecutor: An initialized LangChain agent executor for topic finding.
        """
        # Combines researcher and topic_finder personas for a comprehensive system prompt.
        system_message = self.personas["researcher"]["system_prompt"] + "\n" + self.personas["topic_finder"]["system_prompt"]
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad")
        ])
        
        tools = [Tool(
            name="search",
            func=self.search_tool.invoke, # Uses TavilySearchResults
            description="Search for information on educational topics. Input should be a search query."
        )]
        
        # The TopicList Pydantic model can be used for function calling if the LLM supports it,
        # to get structured output.
        # topic_list_function = convert_to_openai_function(TopicList)
        
        agent = create_openai_tools_agent(self.llm, tools, prompt)
        
        return AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True, # Logs agent actions and thoughts, useful for debugging.
            return_intermediate_steps=True # Returns intermediate steps, can be useful for debugging or advanced logic.
        )
    
    def create_content_creator_agent(self) -> AgentExecutor:
        """
        Creates and returns an agent specialized in generating lecture content.

        This agent uses the "content_creator" persona and a search tool.
        It takes a topic and generates detailed, structured lecture notes.

        Returns:
            AgentExecutor: An initialized LangChain agent executor for content creation.
        """
        system_message = self.personas["content_creator"]["system_prompt"]
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad")
        ])
        
        tools = [Tool(
            name="search",
            func=self.search_tool.invoke,
            description="Search for information on the lecture topic. Input should be a search query."
        )]
        
        agent = create_openai_tools_agent(self.llm, tools, prompt)
        
        return AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            return_intermediate_steps=True
        )
    
    def create_image_finder_agent(self) -> AgentExecutor:
        """
        Creates and returns an agent specialized in finding relevant images for lecture content.

        This agent uses the "image_finder" persona and a search tool.
        It takes a topic and returns URLs and descriptions for relevant images.

        Returns:
            AgentExecutor: An initialized LangChain agent executor for image finding.
        """
        system_message = self.personas["image_finder"]["system_prompt"]
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad")
        ])
        
        tools = [Tool(
            name="search",
            func=self.search_tool.invoke,
            description="Search for educational images and diagrams. Input should be a search query."
        )]
        
        agent = create_openai_tools_agent(self.llm, tools, prompt)
        
        return AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            return_intermediate_steps=True
        )
    
    def create_editor_agent(self) -> AgentExecutor:
        """
        Creates and returns an agent specialized in editing and refining lecture content.

        This agent uses the "editor" persona. It typically does not require external tools
        as its primary function is to improve text provided to it.

        Returns:
            AgentExecutor: An initialized LangChain agent executor for content editing.
        """
        system_message = self.personas["editor"]["system_prompt"]
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad")
        ])
        
        # Editor agent might not need external tools, focuses on refining provided text.
        agent = create_openai_tools_agent(self.llm, tools=[], prompt=prompt)
        
        return AgentExecutor(
            agent=agent,
            tools=[],
            verbose=True,
            return_intermediate_steps=True
        )

# Standalone functions that use the LectureMaterialCreator and its agents.

def find_topics_for_subject(creator: LectureMaterialCreator, subject: str) -> List[str]:
    """
    Finds and lists educational topics for a given subject using the creator's topic_finder_agent.

    Args:
        creator (LectureMaterialCreator): An instance of LectureMaterialCreator.
        subject (str): The subject for which to find topics.

    Returns:
        List[str]: A list of topic strings. Returns an empty list if no topics are found
                   or if the agent output cannot be parsed.
    """
    topic_finder_agent = creator.create_topic_finder_agent()
    
    # Constructing the input prompt for the agent.
    agent_input = f"Research and find 10-15 key educational topics for {subject} that would make good lectures. Format the output as a numbered list."
    
    try:
        result = topic_finder_agent.invoke({
            "input": agent_input,
            "chat_history": [] # Assuming no prior chat history for this interaction.
        })
    except Exception as e:
        st.error(f"Error invoking topic finder agent: {e}")
        return []

    # Parsing the agent's output to extract topics.
    # The agent is prompted to return a numbered list, but parsing should be robust.
    topics = []
    agent_output = result.get("output", "")
    if not agent_output: # Handle cases where output might be None or empty
        st.warning(f"Topic finder agent returned no output for subject: {subject}")
        return []

    for line in agent_output.split('\n'):
        line = line.strip()
        # Check if the line starts with a digit (e.g., "1.") or a common list marker (e.g., "- ").
        if line and (line[0].isdigit() or line.startswith("- ")):
            # Remove numbering/bullet points (e.g., "1. ", "- ")
            cleaned_line = line.split(".", 1)[-1].strip() if "." in line and line[0].isdigit() else line
            cleaned_line = cleaned_line[2:].strip() if cleaned_line.startswith("- ") else cleaned_line
            if cleaned_line: # Ensure non-empty topic after cleaning
                topics.append(cleaned_line)
    
    if not topics: # Log if no topics were parsed from a non-empty output
        st.info(f"No specific topics parsed from agent output for subject: {subject}. Agent output was: '{agent_output}'")

    return topics


def create_lecture_material(creator: LectureMaterialCreator, topic: str) -> tuple[str, List[str], List[str]]:
    """
    Generates lecture material for a specific topic using the creator's agents.

    This involves:
    1. Generating initial content using the content_creator_agent.
    2. Finding relevant images using the image_finder_agent.
    3. Refining the content using the editor_agent.
    4. Parsing image URLs and descriptions from the image_finder_agent's output.

    Args:
        creator (LectureMaterialCreator): An instance of LectureMaterialCreator.
        topic (str): The topic for which to create lecture material.

    Returns:
        tuple[str, List[str], List[str]]: A tuple containing:
            - lecture_content (str): The edited and refined lecture content in Markdown format.
            - image_urls (List[str]): A list of URLs for relevant images.
            - image_descriptions (List[str]): A list of descriptions corresponding to the image_urls.
    """
    content_creator_agent = creator.create_content_creator_agent()
    image_finder_agent = creator.create_image_finder_agent()
    editor_agent = creator.create_editor_agent()
    
    # 1. Generate initial lecture content
    content_prompt = f"Create comprehensive lecture material on '{topic}'. Include definitions, explanations, examples, and practical applications. Structure with clear headings and subheadings using Markdown formatting."
    try:
        content_result = content_creator_agent.invoke({"input": content_prompt, "chat_history": []})
        raw_content = content_result.get("output", "")
    except Exception as e:
        st.error(f"Error invoking content creator agent for topic '{topic}': {e}")
        raw_content = f"Error: Could not generate content for the topic '{topic}'." # Provide a fallback message

    # 2. Find relevant images
    image_prompt = f"Find 2-3 relevant educational diagrams or images for the topic: {topic}. For each image, provide the URL and a brief description."
    try:
        image_result = image_finder_agent.invoke({"input": image_prompt, "chat_history": []})
        image_agent_output = image_result.get("output", "")
    except Exception as e:
        st.error(f"Error invoking image finder agent for topic '{topic}': {e}")
        image_agent_output = "" # No images if agent fails

    # 3. Edit and refine the content
    # Ensure raw_content is not empty before passing to editor to avoid issues.
    if raw_content:
        editor_prompt = f"Review and refine the following lecture material on '{topic}' for clarity, accuracy, and comprehensiveness:\n\n{raw_content}"
        try:
            edited_result = editor_agent.invoke({"input": editor_prompt, "chat_history": []})
            lecture_content = edited_result.get("output", raw_content) # Fallback to raw_content if editor fails or returns empty
        except Exception as e:
            st.error(f"Error invoking editor agent for topic '{topic}': {e}")
            lecture_content = raw_content # Fallback to raw_content
    else:
        lecture_content = raw_content # If raw_content was empty, lecture_content remains so.

    # 4. Parse image URLs and descriptions from image_agent_output
    image_urls = []
    image_descriptions = []
    if image_agent_output: # Proceed only if there's output from the image agent
        for line in image_agent_output.split('\n'):
            line = line.strip()
            # Heuristic to find lines containing image URLs. This can be made more robust.
            # Looks for http/https and common image extensions.
            if "http" in line and any(ext in line.lower() for ext in [".png", ".jpg", ".jpeg", ".svg", ".gif"]):
                url_start_index = line.find("http")
                # Determine end of URL (e.g., by space, comma, or end of line)
                url_end_candidates = [line.find(char, url_start_index) for char in [' ', ','] if line.find(char, url_start_index) != -1]
                url_end_index = min(url_end_candidates) if url_end_candidates else len(line)

                url = line[url_start_index:url_end_index].rstrip(',.;:') # Clean common trailing punctuation from URL

                if url: # Ensure URL is not empty after parsing
                    image_urls.append(url)

                    # Attempt to extract a description if present after the URL.
                    # This part is heuristic and might need refinement based on agent's typical output format.
                    description_part = line[url_end_index:].strip()
                    # Remove common leading characters for descriptions like ':', '-', or just space.
                    if description_part.startswith((":", "-", " ")):
                        description = description_part[1:].strip()
                    else:
                        description = description_part

                    # Use a default description if parsed one is empty or too generic.
                    image_descriptions.append(description if description else f"Image related to {topic}")
    
    return lecture_content, image_urls, image_descriptions


def create_pdf(content: str, image_urls: List[str], image_descriptions: List[str], topic: str) -> bytes:
    """
    Creates a PDF document from Markdown content and a list of image URLs.

    The Markdown content is converted to HTML, and images are embedded.
    WeasyPrint is used for HTML to PDF conversion. Includes basic CSS for styling.

    Args:
        content (str): The lecture content in Markdown format.
        image_urls (List[str]): A list of URLs for images to be included in the PDF.
        image_descriptions (List[str]): A list of descriptions for the images.
        topic (str): The topic of the lecture, used for the PDF title.

    Returns:
        bytes: The generated PDF content as a byte string. Returns empty bytes if an error occurs.
    """
    # Basic CSS for styling the PDF. This can be expanded for more sophisticated styling.
    html_style = """
    <style>
        /* General body styling */
        body { font-family: sans-serif; line-height: 1.6; margin: 20px; }
        /* Headings */
        h1, h2, h3, h4, h5, h6 { page-break-after: avoid; color: #2c3e50; }
        h1 { font-size: 24pt; margin-bottom: 0.5em; border-bottom: 2px solid #3498db; padding-bottom: 0.2em;}
        h2 { font-size: 18pt; margin-bottom: 0.4em; border-bottom: 1px solid #bdc3c7; padding-bottom: 0.1em;}
        h3 { font-size: 14pt; margin-bottom: 0.3em; }
        /* Paragraphs and lists */
        p { margin-bottom: 1em; }
        ul, ol { margin-bottom: 1em; padding-left: 1.8em; }
        li { margin-bottom: 0.3em; }
        /* Code blocks */
        code {
            font-family: "Courier New", Courier, monospace;
            background-color: #ecf0f1;
            padding: 2px 5px;
            border-radius: 4px;
            font-size: 0.9em;
            color: #2c3e50;
        }
        pre {
            background-color: #ecf0f1;
            padding: 12px;
            border-radius: 4px;
            overflow-x: auto; /* Allow horizontal scrolling for wide code blocks */
            font-size: 0.9em;
            border: 1px solid #bdc3c7;
        }
        pre code { padding: 0; background-color: transparent; border-radius: 0; border: none; }
        /* Images and captions */
        img {
            max-width: 90%; /* Limit image width to fit page */
            height: auto; /* Maintain aspect ratio */
            display: block; /* Center images */
            margin-left: auto;
            margin-right: auto;
            margin-top: 15px;
            margin-bottom: 8px;
            border: 1px solid #bdc3c7; /* Optional border for images */
            padding: 3px;
        }
        .caption {
            text-align: center;
            font-style: italic;
            font-size: 0.9em;
            color: #7f8c8d;
            margin-bottom: 20px;
        }
        /* Utility for page breaks if needed, though markdown2 doesn't typically generate this class */
        .page-break { page-break-before: always; }
    </style>
    """

    # Convert Markdown content to HTML using markdown2.
    # Extras like "fenced-code-blocks" (for ```code```) and "tables" enhance compatibility.
    html_content = markdown2.markdown(
        content,
        extras=["fenced-code-blocks", "tables", "header-ids", "smarty-pants", "code-friendly"]
    )
    
    # Prepare the document title, to be placed at the beginning of the body.
    html_title = f"<h1>Lecture Material: {topic}</h1>"
    
    # Construct HTML for images. Each image is followed by its caption.
    images_html_parts = []
    for i, (url, desc) in enumerate(zip(image_urls, image_descriptions)):
        # Basic validation for image URLs to ensure they are web links.
        safe_desc = desc.replace('"', '&quot;') # Sanitize description for HTML attributes
        if url and url.startswith(("http://", "https://")):
            images_html_parts.append(f'<img src="{url}" alt="{safe_desc}">')
            images_html_parts.append(f'<p class="caption">Figure {i+1}: {safe_desc}</p>')
        else:
            # Placeholder for invalid or missing image URLs.
            images_html_parts.append(f'<p class="caption">Figure {i+1}: (Image URL not valid or missing) {safe_desc}</p>')
    images_html = "\n".join(images_html_parts)

    # Combine all parts into a full HTML document.
    # The CSS is embedded in the <head> for simplicity.
    full_html = f"<html><head><meta charset='UTF-8'>{html_style}</head><body>{html_title}{html_content}{images_html}</body></html>"
    
    # Generate PDF from the complete HTML string using WeasyPrint.
    try:
        pdf_bytes = HTML(string=full_html).write_pdf()
        return pdf_bytes
    except Exception as e:
        # If PDF generation fails, log the error (if a logger was configured)
        # and display an error in the Streamlit app.
        st.error(f"Error generating PDF with WeasyPrint: {e}")
        # Return empty bytes to indicate failure.
        return b""


def main():
    """
    A simple main function for testing or standalone execution of parts of this module.
    This is not used when `lecture_content_creator.py` is imported as a module by `app.py`.
    """
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

