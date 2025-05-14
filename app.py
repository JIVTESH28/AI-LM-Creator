import streamlit as st
import os
from dotenv import load_dotenv
from lecture_content_creator import LectureMaterialCreator, find_topics_for_subject, create_lecture_material, create_pdf

# Load environment variables from .env file
load_dotenv()

def main():
    st.set_page_config(page_title="Lecture Material Creator", page_icon="📚", layout="wide")
    
    st.title("📚 AI-Powered Lecture Material Creator")
    st.write("Create comprehensive lecture materials with intelligent AI agents")
    
    # Check if OpenAI API key is set
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        st.error("OpenAI API key not found. Please set it as an environment variable: OPENAI_API_KEY")
        return
    
    # Initialize session state
    if 'topics' not in st.session_state:
        st.session_state.topics = []
    if 'subject' not in st.session_state:
        st.session_state.subject = ""
    if 'selected_topic' not in st.session_state:
        st.session_state.selected_topic = ""
    if 'lecture_material' not in st.session_state:
        st.session_state.lecture_material = ""
    if 'pdf_bytes' not in st.session_state:
        st.session_state.pdf_bytes = None
    
    # Sidebar for settings
    with st.sidebar:
        st.header("Settings")
        model_name = st.selectbox(
            "Select AI Model",
            ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
            index=0
        )
        
        st.divider()
        st.write("### About")
        st.write("""
        This application uses AI agents to create comprehensive lecture materials.
        It searches educational websites for topics, generates content, and includes
        relevant images to create a complete learning resource.
        """)
    
    # Main content area with tabs
    tab1, tab2, tab3 = st.tabs(["Create Material", "Preview Material", "Download"])
    
    # Tab 1: Create Material
    with tab1:
        # Step 1: Enter subject
        with st.container():
            st.header("Step 1: Enter Subject")
            subject = st.text_input("Enter the subject (e.g., Python Programming, Machine Learning)", 
                                  value=st.session_state.subject)
            
            col1, col2 = st.columns([1, 5])
            with col1:
                find_topics_button = st.button("Find Topics", use_container_width=True)
            
            if find_topics_button and subject:
                st.session_state.subject = subject
                
                with st.spinner(f"Finding topics for {subject}..."):
                    try:
                        creator = LectureMaterialCreator(model_name=model_name)
                        topics = find_topics_for_subject(creator, subject)
                        st.session_state.topics = topics
                        st.success(f"Found {len(topics)} topics for {subject}")
                    except Exception as e:
                        st.error(f"Error finding topics: {str(e)}")
        
        # Step 2: Select topics
        if st.session_state.topics:
            with st.container():
                st.header("Step 2: Select Topic")
                
                # Display topics as radio buttons
                selected_topic = st.radio("Choose a topic for your lecture material:", 
                                         st.session_state.topics)
                
                col1, col2 = st.columns([1, 5])
                with col1:
                    generate_button = st.button("Generate Material", use_container_width=True)
                
                if generate_button and selected_topic:
                    st.session_state.selected_topic = selected_topic
                    
                    with st.spinner(f"Creating lecture material for {selected_topic}..."):
                        try:
                            creator = LectureMaterialCreator(model_name=model_name)
                            # Fix here: capture all three return values
                            lecture_content, image_urls, image_descriptions = create_lecture_material(creator, selected_topic)
                            st.session_state.lecture_material = lecture_content
                            
                            # Create PDF
                            pdf_bytes = create_pdf(lecture_content, image_urls, image_descriptions, selected_topic)
                            st.session_state.pdf_bytes = pdf_bytes
                            
                            st.success("Lecture material created successfully!")
                            # Switch to the Preview tab
                            tab2.active = True
                        except Exception as e:
                            st.error(f"Error generating lecture material: {str(e)}")
    
    # Tab 2: Preview Material
    with tab2:
        if st.session_state.lecture_material:
            st.header(f"Lecture Material: {st.session_state.selected_topic}")
            st.markdown(st.session_state.lecture_material)
        else:
            st.info("No lecture material generated yet. Go to the 'Create Material' tab to generate content.")
    
    # Tab 3: Download
    with tab3:
        if st.session_state.pdf_bytes:
            st.header("Download Lecture Material")
            st.write("Your lecture material is ready for download as a PDF.")
            
            col1, col2 = st.columns([1, 3])
            with col1:
                st.download_button(
                    label="Download PDF",
                    data=st.session_state.pdf_bytes,
                    file_name=f"{st.session_state.selected_topic.replace(' ', '_')}_lecture.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            
            with col2:
                st.write("This PDF includes the complete lecture material with relevant images.")
            
            # Preview of the first page as an image (if possible)
            st.write("### How to use this material")
            st.write("""
            1. **Review the material**: Read through the content to get familiar with the topic
            2. **Add your own notes**: Customize the material with your own insights
            3. **Use for teaching**: Present the material in your lecture or class
            4. **Share with students**: Distribute the PDF to your students as a learning resource
            """)
        else:
            st.info("No PDF generated yet. Go to the 'Create Material' tab to generate content.")

if __name__ == "__main__":
    main()