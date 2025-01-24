import streamlit as st
import asyncio
import dspy
from research_agent import AdvancedResearchAgent, format_markdown_report, save_report
from pathlib import Path
import json
from datetime import datetime
import pandas as pd
from streamlit_pills import pills
import time
from config import config
import nest_asyncio

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Configure DSPy with our config module
dspy.configure(lm=config.get_dspy_lm())

# Configure Streamlit page
st.set_page_config(
    page_title="Advanced Research Agent",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
        .stProgress .st-bo {
            background-color: #f0f2f6;
        }
        .stProgress .st-bp {
            background-color: #00a0dc;
        }
        .research-status {
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
            background-color: #f7f7f7;
        }
        .feedback-box {
            border: 1px solid #ddd;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
        }
        .section-header {
            background-color: #f0f2f6;
            padding: 0.5rem;
            border-radius: 0.3rem;
            margin: 1rem 0;
        }
    </style>
""", unsafe_allow_html=True)


class StreamlitResearchUI:
    def __init__(self):
        self.agent = AdvancedResearchAgent()
        self.progress = 0
        self.status_text = ""

        # Initialize session state
        if 'research_complete' not in st.session_state:
            st.session_state.research_complete = False
        if 'report_data' not in st.session_state:
            st.session_state.report_data = None
        if 'status_message' not in st.session_state:
            st.session_state.status_message = ""
        if 'feedback' not in st.session_state:
            st.session_state.feedback = {}
        if 'selected_section' not in st.session_state:
            st.session_state.selected_section = None

    def update_progress(self, message: str, progress: float):
        """Update progress bar and status message."""
        self.status_text = message
        self.progress = progress
        if hasattr(self, 'progress_bar'):
            self.progress_bar.progress(progress)
        if hasattr(self, 'status_placeholder'):
            self.status_placeholder.markdown(f"**Status:** {message}")
        st.session_state.status_message = message

    async def process_research(self, query: str):
        """Process research query with progress updates."""
        try:
            # Reset progress
            self.progress = 0
            st.session_state.status_message = "Initializing research..."
            self.status_placeholder.markdown(
                "**Status:** Initializing research...")
            self.progress_bar.progress(0.1)

            # Define progress callback
            def progress_callback(status):
                try:
                    if isinstance(status, dict) and 'section' in status:
                        section_name = status['section']
                        progress = 0.2
                        if st.session_state.report_data and 'sections' in st.session_state.report_data:
                            progress += 0.6 * \
                                (len(st.session_state.report_data['sections']))

                        # Update status directly
                        message = f"Processing section: {section_name}"
                        st.session_state.status_message = message
                        self.status_placeholder.markdown(
                            f"**Status:** {message}")
                        self.progress_bar.progress(min(progress, 0.9))
                except Exception as e:
                    st.error(f"Error in progress callback: {str(e)}")

            # Run research with status updates
            with st.spinner("Starting research process..."):
                self.status_placeholder.markdown(
                    "**Status:** Gathering initial research data...")
                self.progress_bar.progress(0.2)
                report_data = await self.agent.generate_report(query, progress_callback)

                if report_data:
                    st.session_state.report_data = report_data
                    self.status_placeholder.markdown(
                        "**Status:** Formatting research report...")
                    self.progress_bar.progress(0.9)
                    markdown_report = format_markdown_report(report_data)
                    file_path = save_report(
                        markdown_report, query, report_data)

                    self.status_placeholder.markdown(
                        "**Status:** Research complete!")
                    self.progress_bar.progress(1.0)
                    st.session_state.research_complete = True
                    return report_data, file_path
                else:
                    st.error("Failed to generate research report.")
                    return None, None

        except Exception as e:
            st.error(f"Error during research: {str(e)}")
            return None, None

    def display_table(self, table_data):
        """Display table data."""
        try:
            if isinstance(table_data, str):
                st.markdown(table_data)
            else:
                st.table(pd.DataFrame(table_data))
        except Exception as e:
            st.warning(f"Could not display table: {str(e)}")

    def display_report(self, report_data):
        """Display the research report in an organized layout."""
        if not report_data:
            return

        # Sidebar navigation
        st.sidebar.title("Navigation")
        sections = [section['heading'] for section in report_data['sections']]

        # Use session state to maintain selected section
        selected_section = st.sidebar.selectbox(
            "Jump to Section",
            sections,
            key='section_selector',
            index=sections.index(
                st.session_state.selected_section) if st.session_state.selected_section in sections else 0
        )
        st.session_state.selected_section = selected_section

        # Add download options in sidebar
        st.sidebar.title("Download Report")

        # Markdown download
        markdown_content = format_markdown_report(report_data)
        st.sidebar.download_button(
            "📝 Download as Markdown",
            markdown_content,
            file_name=f"research_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown"
        )

        # Word document download
        if st.sidebar.button("📄 Download as Word Document"):
            try:
                doc_path = self.agent.export_to_docx(
                    report_data,
                    Path(st.session_state.file_path).parent / "report"
                )
                if doc_path and doc_path.exists():
                    with open(doc_path, 'rb') as f:
                        st.sidebar.download_button(
                            "📄 Download Word Document",
                            f,
                            file_name=f"research_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx",
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                        )
                else:
                    st.sidebar.error("Failed to generate Word document")
            except Exception as e:
                st.sidebar.error(f"Error generating Word document: {str(e)}")

        # PDF download
        if st.sidebar.button("📑 Download as PDF"):
            try:
                with st.spinner("Generating PDF..."):
                    pdf_path = self.agent.export_to_pdf(
                        report_data,
                        Path(st.session_state.file_path).parent / "report"
                    )
                    if pdf_path and pdf_path.exists():
                        with open(pdf_path, 'rb') as f:
                            st.sidebar.download_button(
                                "📑 Download PDF",
                                f,
                                file_name=f"research_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                mime="application/pdf"
                            )
                    else:
                        st.sidebar.error("Failed to generate PDF")
            except Exception as e:
                error_msg = str(e)
                if "wkhtmltopdf" in error_msg.lower():
                    st.sidebar.error(error_msg)
                    st.sidebar.info(
                        "After installing wkhtmltopdf, please restart the application.")
                else:
                    st.sidebar.error(f"Error generating PDF: {error_msg}")
                    st.sidebar.info(
                        "Please check the console for more details.")

        # Main content area
        st.title(report_data['title'])
        st.markdown(
            f"*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")

        # Research methodology
        with st.expander("Research Methodology", expanded=True):
            for i, step in enumerate(report_data['research_plan'], 1):
                st.markdown(f"{i}. {step}")

        # Display selected section
        for section in report_data['sections']:
            if section['heading'] == selected_section:
                st.markdown(f"## {section['heading']}")

                # Content with feedback option
                with st.container():
                    st.markdown(section['content'])

                    # Feedback interface
                    with st.expander("Provide Feedback"):
                        feedback_key = f"feedback_{section['heading']}"
                        feedback = st.text_area(
                            "How can we improve this section?",
                            value=st.session_state.feedback.get(
                                section['heading'], ''),
                            key=feedback_key
                        )
                        if st.button("Submit Feedback", key=f"submit_{section['heading']}"):
                            st.session_state.feedback[section['heading']] = feedback
                            st.success("Feedback submitted successfully!")

                # Tables
                if section.get('tables'):
                    st.markdown("### Tables")
                    for table in section['tables']:
                        self.display_table(table)

                # Citations
                if section.get('citations'):
                    with st.expander("Sources"):
                        for citation in section['citations']:
                            st.markdown(
                                f"- [{citation['id']}] {citation['text']} - {citation['source'].get('url', 'N/A')}")

        # Follow-up questions
        if report_data.get('follow_up_questions'):
            with st.expander("Follow-up Questions"):
                for question in report_data['follow_up_questions']:
                    st.markdown(f"- {question}")


def main():
    ui = StreamlitResearchUI()

    # Title and description
    st.title("🔍 Advanced Research Agent")
    st.markdown("""
        Welcome to the Advanced Research Agent! This tool helps you generate comprehensive
        research reports by combining multiple sources and organizing information effectively.
    """)

    # Query input
    query = st.text_area("Enter your research query:", height=100)

    if st.button("Start Research"):
        if not query:
            st.warning("Please enter a research query.")
            return

        # Initialize progress tracking
        ui.progress_bar = st.progress(0)
        ui.status_placeholder = st.empty()

        # Process research
        report_data, file_path = asyncio.run(ui.process_research(query))

        if report_data:
            st.session_state.report_data = report_data  # Store report data in session state
            st.session_state.file_path = file_path
            st.session_state.research_complete = True
            st.rerun()  # Rerun to show the report
        else:
            st.error("Failed to generate research report. Please try again.")

    # Display report if it exists in session state
    if st.session_state.get('research_complete') and st.session_state.get('report_data'):
        ui.display_report(st.session_state.report_data)
        if st.session_state.get('file_path'):
            st.success(f"Report saved to: {st.session_state.file_path}")


if __name__ == "__main__":
    main()
