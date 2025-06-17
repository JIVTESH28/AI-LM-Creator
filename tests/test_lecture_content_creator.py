import pytest
from unittest.mock import MagicMock, patch, call
import lecture_content_creator
from lecture_content_creator import (
    LectureMaterialCreator,
    find_topics_for_subject,
    create_lecture_material,
    create_pdf
)
import streamlit as st # Required for mocking st.error

# Mock os.getenv for API keys to avoid actual environment dependency
@pytest.fixture(autouse=True)
def mock_env_vars(mocker):
    mocker.patch.dict(lecture_content_creator.os.environ, {
        "TAVILY_API_KEY": "fake_tavily_key",
        "OPENAI_API_KEY": "fake_openai_key",
        "ANTHROPIC_API_KEY": "fake_anthropic_key"
    })

@pytest.fixture
def mock_creator(mocker):
    creator = MagicMock(spec=LectureMaterialCreator)
    # Mock the _load_personas method to return a valid dictionary
    creator._load_personas = MagicMock(return_value={
        "researcher": {"system_prompt": "researcher_prompt"},
        "topic_finder": {"system_prompt": "topic_finder_prompt"},
        "content_creator": {"system_prompt": "content_creator_prompt"},
        "image_finder": {"system_prompt": "image_finder_prompt"},
        "editor": {"system_prompt": "editor_prompt"}
    })
    # Mock LLM initialization as it requires API keys if not done carefully
    creator._initialize_llm = MagicMock(return_value=MagicMock())
    # Mock search tool
    creator.search_tool = MagicMock()
    return creator

class TestFindTopics:
    def test_find_topics_parses_numbered_list(self, mock_creator, mocker):
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {
            "output": "1. Topic One\n2. Topic Two\n3. Another Topic"
        }
        mock_creator.create_topic_finder_agent = MagicMock(return_value=mock_agent)

        topics = find_topics_for_subject(mock_creator, "Some Subject")

        assert topics == ["Topic One", "Topic Two", "Another Topic"]
        mock_agent.invoke.assert_called_once_with({
            "input": "Research and find 10-15 key educational topics for Some Subject that would make good lectures. Format the output as a numbered list.",
            "chat_history": []
        })

    def test_find_topics_parses_bulleted_list(self, mock_creator, mocker):
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {
            "output": "- Topic A\n- Topic B\n- Topic C with spaces"
        }
        mock_creator.create_topic_finder_agent = MagicMock(return_value=mock_agent)

        topics = find_topics_for_subject(mock_creator, "Another Subject")

        assert topics == ["Topic A", "Topic B", "Topic C with spaces"]

    def test_find_topics_handles_mixed_formatting_and_empty_lines(self, mock_creator, mocker):
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {
            "output": "1. Topic 1\n\n- Topic 2\n3. Topic 3 with extra.period."
        }
        mock_creator.create_topic_finder_agent = MagicMock(return_value=mock_agent)

        topics = find_topics_for_subject(mock_creator, "Mixed Subject")

        assert topics == ["Topic 1", "Topic 2", "Topic 3 with extra.period."]

    def test_find_topics_handles_no_topics_found(self, mock_creator, mocker):
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {"output": "No topics found for this subject."}
        mock_creator.create_topic_finder_agent = MagicMock(return_value=mock_agent)

        topics = find_topics_for_subject(mock_creator, "Obscure Subject")

        assert topics == []

class TestCreateLectureMaterial:
    @pytest.fixture
    def setup_mocks(self, mock_creator, mocker):
        self.mock_content_agent = MagicMock()
        self.mock_image_agent = MagicMock()
        self.mock_editor_agent = MagicMock()

        mock_creator.create_content_creator_agent = MagicMock(return_value=self.mock_content_agent)
        mock_creator.create_image_finder_agent = MagicMock(return_value=self.mock_image_agent)
        mock_creator.create_editor_agent = MagicMock(return_value=self.mock_editor_agent)

        return mock_creator, self.mock_content_agent, self.mock_image_agent, self.mock_editor_agent

    def test_create_material_happy_path(self, setup_mocks):
        mock_creator, mock_content_agent, mock_image_agent, mock_editor_agent = setup_mocks

        mock_content_agent.invoke.return_value = {"output": "Raw lecture content."}
        mock_image_agent.invoke.return_value = {
            "output": "Image 1: http://example.com/image1.png - Description 1\nImage 2: https://example.com/image2.jpg, Description 2"
        }
        mock_editor_agent.invoke.return_value = {"output": "Edited lecture content."}

        content, urls, descs = create_lecture_material(mock_creator, "Test Topic")

        assert content == "Edited lecture content."
        assert urls == ["http://example.com/image1.png", "https://example.com/image2.jpg"]
        assert descs == ["Description 1", "Description 2"]

        mock_content_agent.invoke.assert_called_once_with({
            "input": "Create comprehensive lecture material on 'Test Topic'. Include definitions, explanations, examples, and practical applications. Structure with clear headings and subheadings using Markdown formatting.",
            "chat_history": []
        })
        mock_image_agent.invoke.assert_called_once_with({
            "input": "Find 2-3 relevant educational diagrams or images for the topic: Test Topic. For each image, provide the URL and a brief description.",
            "chat_history": []
        })
        mock_editor_agent.invoke.assert_called_once_with({
            "input": "Review and refine the following lecture material on 'Test Topic' for clarity, accuracy, and comprehensiveness:\n\nRaw lecture content.",
            "chat_history": []
        })

    def test_image_parsing_various_formats(self, setup_mocks):
        mock_creator, _, mock_image_agent, _ = setup_mocks
        mock_image_agent.invoke.return_value = {
            "output": (
                "1. URL: http://site.com/img.png (Caption for img1)\n"
                "- https://another.site/pic.jpeg - Caption for pic2\n"
                "Just a line with http://nolongeravalidformat.com/img.svg\n" # Should be picked up if it has image extension
                "No URL here.\n"
                "Image: http://image.com/graphic.svg: A nice graphic." # Valid
            )
        }
        # Need to mock content and editor agents as well, even if not the focus here
        mock_creator.create_content_creator_agent.return_value.invoke.return_value = {"output": "content"}
        mock_creator.create_editor_agent.return_value.invoke.return_value = {"output": "edited_content"}


        _, urls, descs = create_lecture_material(mock_creator, "Image Test Topic")

        assert "http://site.com/img.png" in urls
        assert "https://another.site/pic.jpeg" in urls
        assert "http://nolongeravalidformat.com/img.svg" in urls #This will be picked up
        assert "http://image.com/graphic.svg" in urls

        assert "Caption for img1" in descs
        assert "Caption for pic2" in descs
        # For the one without explicit caption but with image extension
        assert f"Image related to Image Test Topic" in descs # Default description
        assert "A nice graphic." in descs


    def test_image_parsing_no_images_found(self, setup_mocks):
        mock_creator, _, mock_image_agent, _ = setup_mocks
        mock_image_agent.invoke.return_value = {"output": "No images found for this topic."}

        # Mock other agents
        mock_creator.create_content_creator_agent.return_value.invoke.return_value = {"output": "content"}
        mock_creator.create_editor_agent.return_value.invoke.return_value = {"output": "edited_content"}

        _, urls, descs = create_lecture_material(mock_creator, "No Image Topic")

        assert urls == []
        assert descs == []

class TestCreatePdf:
    @patch('lecture_content_creator.markdown2.markdown')
    @patch('lecture_content_creator.HTML') # from weasyprint import HTML
    def test_create_pdf_happy_path(self, mock_weasy_html, mock_markdown2, mocker):
        mock_st_error = mocker.patch('streamlit.error') # Mock streamlit.error

        test_content = "## Test Content\n- Point 1"
        test_image_urls = ["http://example.com/img1.png"]
        test_image_descs = ["Description for img1"]
        test_topic = "PDF Test Topic"

        mock_markdown2.return_value = "<p>HTML Content</p>"

        mock_pdf_instance = MagicMock()
        mock_pdf_instance.write_pdf.return_value = b"pdf_bytes_content"
        mock_weasy_html.return_value = mock_pdf_instance

        pdf_bytes = create_pdf(test_content, test_image_urls, test_image_descs, test_topic)

        assert pdf_bytes == b"pdf_bytes_content"
        mock_markdown2.assert_called_once_with(test_content, extras=["fenced-code-blocks", "tables", "header-ids", "smarty-pants"])

        # Check that HTML was called with a string containing the title, content, and image
        args, _ = mock_weasy_html.call_args
        html_passed_to_weasyprint = args[0]
        assert f"<h1>Lecture Material: {test_topic}</h1>" in html_passed_to_weasyprint
        assert "<p>HTML Content</p>" in html_passed_to_weasyprint
        assert '<img src="http://example.com/img1.png" alt="Description for img1">' in html_passed_to_weasyprint
        assert '<p class="caption">Figure 1: Description for img1</p>' in html_passed_to_weasyprint

        mock_pdf_instance.write_pdf.assert_called_once()
        mock_st_error.assert_not_called()

    @patch('lecture_content_creator.markdown2.markdown')
    @patch('lecture_content_creator.HTML')
    def test_create_pdf_weasyprint_error(self, mock_weasy_html, mock_markdown2, mocker):
        mock_st_error = mocker.patch('streamlit.error')

        mock_markdown2.return_value = "<p>Some HTML</p>"
        mock_weasy_html_instance = MagicMock()
        mock_weasy_html_instance.write_pdf.side_effect = Exception("WeasyPrintError")
        mock_weasy_html.return_value = mock_weasy_html_instance

        pdf_bytes = create_pdf("content", [], [], "Error Topic")

        assert pdf_bytes == b""
        mock_st_error.assert_called_once()
        assert "Error generating PDF with WeasyPrint: WeasyPrintError" in mock_st_error.call_args[0][0]

    @patch('lecture_content_creator.markdown2.markdown')
    @patch('lecture_content_creator.HTML')
    def test_create_pdf_invalid_image_url(self, mock_weasy_html, mock_markdown2, mocker):
        mock_st_error = mocker.patch('streamlit.error')
        test_content = "Some content"
        test_image_urls = ["invalid_url_format"] # Invalid URL
        test_image_descs = ["Invalid image"]
        test_topic = "Invalid Image URL Topic"

        mock_markdown2.return_value = "<p>HTML for invalid image</p>"
        mock_pdf_instance = MagicMock()
        mock_pdf_instance.write_pdf.return_value = b"pdf_for_invalid_image"
        mock_weasy_html.return_value = mock_pdf_instance

        pdf_bytes = create_pdf(test_content, test_image_urls, test_image_descs, test_topic)

        assert pdf_bytes == b"pdf_for_invalid_image"

        args, _ = mock_weasy_html.call_args
        html_passed_to_weasyprint = args[0]
        assert '<p class="caption">Figure 1: (Invalid URL) Invalid image</p>' in html_passed_to_weasyprint
        assert '<img src="invalid_url_format" alt="Invalid image">' not in html_passed_to_weasyprint # Should not generate img tag for invalid url
        mock_st_error.assert_not_called()

    @patch('lecture_content_creator.markdown2.markdown')
    @patch('lecture_content_creator.HTML')
    def test_create_pdf_no_images(self, mock_weasy_html, mock_markdown2, mocker):
        mock_st_error = mocker.patch('streamlit.error')
        test_content = "Content without images"
        test_topic = "No Image PDF Topic"

        mock_markdown2.return_value = "<p>HTML no images</p>"
        mock_pdf_instance = MagicMock()
        mock_pdf_instance.write_pdf.return_value = b"pdf_no_images"
        mock_weasy_html.return_value = mock_pdf_instance

        pdf_bytes = create_pdf(test_content, [], [], test_topic)

        assert pdf_bytes == b"pdf_no_images"
        args, _ = mock_weasy_html.call_args
        html_passed_to_weasyprint = args[0]
        assert "<img" not in html_passed_to_weasyprint # No image tags
        mock_st_error.assert_not_called()

# It's good practice to also test the LectureMaterialCreator class itself,
# especially the _initialize_llm and _load_personas methods if they were more complex.
# For _load_personas, since it now reads from a file, you might mock open.
@patch('builtins.open', new_callable=pytest.mock.mock_open, read_data="personas:\n  researcher:\n    system_prompt: \"Test prompt from YAML\"")
@patch('lecture_content_creator.yaml.safe_load')
def test_load_personas_success(mock_safe_load, mock_file_open, mocker):
    # Temporarily unpatch os.getenv for this specific test if it interferes,
    # or ensure LectureMaterialCreator doesn't rely on all keys for just this method.
    # Here, we assume constructor doesn't fail with missing keys if we only call _load_personas.

    # To properly test LectureMaterialCreator, we need to instantiate it.
    # Mocking os.getenv for the duration of this test or ensuring keys are set.
    mocker.patch.dict(lecture_content_creator.os.environ, {
        "TAVILY_API_KEY": "fake_tavily_key",
        "OPENAI_API_KEY": "fake_openai_key", # Needed if not mocking _initialize_llm
        "ANTHROPIC_API_KEY": "fake_anthropic_key" # Needed if not mocking _initialize_llm
    })

    mock_safe_load.return_value = {"personas": {"researcher": {"system_prompt": "Test prompt from YAML"}}}

    # We need an instance of LectureMaterialCreator to call _load_personas
    # We can mock _initialize_llm to prevent actual LLM setup
    with patch.object(LectureMaterialCreator, '_initialize_llm', MagicMock()):
        creator_instance = LectureMaterialCreator() # Now calls the real _load_personas

    personas = creator_instance.personas # Access the loaded personas

    mock_file_open.assert_called_once_with("prompts.yaml", "r")
    mock_safe_load.assert_called_once()
    assert "researcher" in personas
    assert personas["researcher"]["system_prompt"] == "Test prompt from YAML"

@patch('builtins.open', side_effect=FileNotFoundError)
@patch('lecture_content_creator.st.error') # Mock streamlit.error
def test_load_personas_file_not_found(mock_st_error, mock_file_open, mocker):
    mocker.patch.dict(lecture_content_creator.os.environ, {
        "TAVILY_API_KEY": "fake_tavily_key",
        "OPENAI_API_KEY": "fake_openai_key",
        "ANTHROPIC_API_KEY": "fake_anthropic_key"
    })
    with patch.object(LectureMaterialCreator, '_initialize_llm', MagicMock()):
        creator_instance = LectureMaterialCreator()

    assert creator_instance.personas == {}
    mock_st_error.assert_called_once_with("prompts.yaml not found. Please ensure the file exists.")

@patch('builtins.open', new_callable=pytest.mock.mock_open, read_data="invalid_yaml_content: [")
@patch('lecture_content_creator.yaml.safe_load', side_effect=lecture_content_creator.yaml.YAMLError("YAML parse error"))
@patch('lecture_content_creator.st.error') # Mock streamlit.error
def test_load_personas_yaml_error(mock_st_error, mock_safe_load, mock_file_open, mocker):
    mocker.patch.dict(lecture_content_creator.os.environ, {
        "TAVILY_API_KEY": "fake_tavily_key",
        "OPENAI_API_KEY": "fake_openai_key",
        "ANTHROPIC_API_KEY": "fake_anthropic_key"
    })
    with patch.object(LectureMaterialCreator, '_initialize_llm', MagicMock()):
        creator_instance = LectureMaterialCreator()

    assert creator_instance.personas == {}
    mock_st_error.assert_called_once_with("Error parsing prompts.yaml: YAML parse error")

# Test LLM initialization
def test_initialize_llm_gpt(mocker):
    mocker.patch.dict(lecture_content_creator.os.environ, {
        "OPENAI_API_KEY": "fake_openai_key",
        "TAVILY_API_KEY": "fake_tavily_key" # Tavily key also checked in constructor
    })
    mock_chat_openai = mocker.patch('lecture_content_creator.ChatOpenAI')
    creator = LectureMaterialCreator(model_name="gpt-4o")
    creator._initialize_llm() # Call it directly to test
    mock_chat_openai.assert_called_once_with(model="gpt-4o", temperature=0.7, api_key="fake_openai_key")

def test_initialize_llm_claude(mocker):
    mocker.patch.dict(lecture_content_creator.os.environ, {
        "ANTHROPIC_API_KEY": "fake_anthropic_key",
        "TAVILY_API_KEY": "fake_tavily_key" # Tavily key also checked in constructor
    })
    mock_chat_anthropic = mocker.patch('lecture_content_creator.ChatAnthropic')
    creator = LectureMaterialCreator(model_name="claude-3-opus")
    creator._initialize_llm() # Call it directly to test
    mock_chat_anthropic.assert_called_once_with(model="claude-3-opus", temperature=0.7, anthropic_api_key="fake_anthropic_key")

def test_initialize_llm_unsupported(mocker):
    mocker.patch.dict(lecture_content_creator.os.environ, {
        "TAVILY_API_KEY": "fake_tavily_key"
    })
    with pytest.raises(ValueError, match="Unsupported model: llama-3"):
        creator = LectureMaterialCreator(model_name="llama-3")
        creator._initialize_llm() # This will be called by constructor

def test_initialize_llm_missing_openai_key(mocker):
    mocker.patch.dict(lecture_content_creator.os.environ, {
        "TAVILY_API_KEY": "fake_tavily_key",
        # OPENAI_API_KEY is missing
    })
    mocker.patch.dict(lecture_content_creator.os.environ, {"OPENAI_API_KEY": ""}) # Ensure it's empty
    with pytest.raises(ValueError, match="OPENAI_API_KEY environment variable not found"):
        creator = LectureMaterialCreator(model_name="gpt-4")
        creator._initialize_llm()

def test_initialize_llm_missing_anthropic_key(mocker):
    mocker.patch.dict(lecture_content_creator.os.environ, {
        "TAVILY_API_KEY": "fake_tavily_key",
        # ANTHROPIC_API_KEY is missing
    })
    mocker.patch.dict(lecture_content_creator.os.environ, {"ANTHROPIC_API_KEY": ""}) # Ensure it's empty
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY environment variable not found"):
        creator = LectureMaterialCreator(model_name="claude-3")
        creator._initialize_llm()

# Test constructor regarding TAVILY_API_KEY
def test_creator_constructor_missing_tavily_key(mocker):
    mocker.patch.dict(lecture_content_creator.os.environ, {"TAVILY_API_KEY": ""}) # Ensure it's empty
    with pytest.raises(ValueError, match="TAVILY_API_KEY environment variable not found"):
        LectureMaterialCreator()

# Minimal test for agent creation methods to ensure they use personas
# These tests primarily check if the prompts from loaded personas are used.
def test_create_topic_finder_agent_uses_personas(mock_creator, mocker):
    mock_chat_prompt_template = mocker.patch('lecture_content_creator.ChatPromptTemplate.from_messages')
    mocker.patch('lecture_content_creator.create_openai_tools_agent') # Mock deeper call
    mocker.patch('lecture_content_creator.AgentExecutor')

    mock_creator.create_topic_finder_agent() # Call the real method on the mock

    # Check that from_messages was called and the system prompt was constructed correctly
    # from the mocked personas
    args, _ = mock_chat_prompt_template.call_args
    messages_arg = args[0]
    system_message_tuple = messages_arg[0] # Should be ('system', system_prompt_string)
    assert system_message_tuple[0] == "system"
    assert "researcher_prompt" in system_message_tuple[1]
    assert "topic_finder_prompt" in system_message_tuple[1]

def test_create_content_creator_agent_uses_personas(mock_creator, mocker):
    mock_chat_prompt_template = mocker.patch('lecture_content_creator.ChatPromptTemplate.from_messages')
    mocker.patch('lecture_content_creator.create_openai_tools_agent')
    mocker.patch('lecture_content_creator.AgentExecutor')

    mock_creator.create_content_creator_agent()
    args, _ = mock_chat_prompt_template.call_args
    messages_arg = args[0]
    system_message_tuple = messages_arg[0]
    assert system_message_tuple[0] == "system"
    assert "content_creator_prompt" in system_message_tuple[1]

def test_create_image_finder_agent_uses_personas(mock_creator, mocker):
    mock_chat_prompt_template = mocker.patch('lecture_content_creator.ChatPromptTemplate.from_messages')
    mocker.patch('lecture_content_creator.create_openai_tools_agent')
    mocker.patch('lecture_content_creator.AgentExecutor')

    mock_creator.create_image_finder_agent()
    args, _ = mock_chat_prompt_template.call_args
    messages_arg = args[0]
    system_message_tuple = messages_arg[0]
    assert system_message_tuple[0] == "system"
    assert "image_finder_prompt" in system_message_tuple[1]

def test_create_editor_agent_uses_personas(mock_creator, mocker):
    mock_chat_prompt_template = mocker.patch('lecture_content_creator.ChatPromptTemplate.from_messages')
    mocker.patch('lecture_content_creator.create_openai_tools_agent')
    mocker.patch('lecture_content_creator.AgentExecutor')

    mock_creator.create_editor_agent()
    args, _ = mock_chat_prompt_template.call_args
    messages_arg = args[0]
    system_message_tuple = messages_arg[0]
    assert system_message_tuple[0] == "system"
    assert "editor_prompt" in system_message_tuple[1]

# Example of how you might run tests with pytest:
# Ensure you have pytest and pytest-mock installed:
# pip install pytest pytest-mock
# Then run from the root of your repository:
# pytest
#
# To get coverage:
# pip install pytest-cov
# pytest --cov=lecture_content_creator --cov-report=html
# (and view htmlcov/index.html)
