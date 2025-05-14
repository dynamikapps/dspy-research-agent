# Advanced Research Agent

This advanced research agent combines DSPy, ReAct, Tavily Search, and Crawl4AI to generate comprehensive, well-structured research reports. It features a multi-stage pipeline with context-aware content generation and source attribution.

## Environment Setup

1. Create and activate a conda environment:

```bash
# Create new environment
conda create -n research_agent python=3.10
conda activate research_agent

# Install dependencies
pip install -r requirements.txt
```

2. Install wkhtmltopdf (required for PDF export):

   - **macOS**: `brew install wkhtmltopdf`
   - **Linux**: `sudo apt-get install wkhtmltopdf`
   - **Windows**: Download and install from [wkhtmltopdf.org](https://wkhtmltopdf.org/downloads.html)

3. Create a `.env` file with your API keys:

```
OPENAI_API_KEY=your_openai_api_key
TAVILY_API_KEY=your_tavily_api_key
```

## Features

1. **ReAct-Based Research Planning**

   - Dynamic research strategy generation
   - Context-aware planning
   - Automated follow-up question generation
   - Tool integration for adaptive research

2. **Advanced Content Generation**

   - Context-aware section generation
   - Automatic source attribution
   - Wikipedia-style citations
   - Content review and improvement
   - Section-specific context mapping

3. **Source Management**

   - Automated citation tracking
   - Hyperlinked references
   - Source content extraction
   - URL crawling and content processing

4. **Report Organization**

   - Structured table of contents
   - Research methodology documentation
   - Section-based organization
   - Follow-up questions section
   - Properly formatted citations

5. **Export Options**
   - Markdown format with citations and tables
   - PDF export with professional formatting
   - Word document (.docx) with proper styling
   - Interactive web interface

## Available Tools

### Research Tools

- **Tavily Search**: Advanced web search with content synthesis
- **Crawl4AI**: Deep web content extraction and processing
- **ReAct Agent**: Dynamic research action planning and execution

### Content Processing

- **Context Analysis**: Section-specific content relevance mapping
- **Citation Management**: Automatic source tracking and linking
- **Content Review**: Automated content improvement suggestions

## Usage

1. Start the Streamlit interface:

```bash
streamlit run streamlit_app.py
```

2. Enter your research query in the text area
3. Click "Start Research" to begin the research process
4. Navigate through sections using the sidebar
5. Download the report in your preferred format:
   - Markdown (.md)
   - PDF (requires wkhtmltopdf)
   - Word Document (.docx)

### Report Structure

Reports are saved in the `reports/content/` directory with the following structure:

```
reports/
└── content/
    └── query_name_timestamp/
        ├── report.md
        └── metadata.json
```

### Report Sections

- Title and Timestamp
- Table of Contents
- Research Methodology
- Research Outline
- Main Content (with citations)
- Further Research Questions
- References

## Output Format

- Properly formatted markdown
- Wikipedia-style citations
- Internal navigation links
- Source references with quotes
- Tables in markdown format

## Error Handling

The agent includes comprehensive error handling for:

- API failures
- Content processing issues
- File operations
- Citation processing
- Export format generation

## Customization

You can customize:

- Search depth and result count
- Report formatting
- Content processing parameters
- Crawling configurations

## Requirements

### System Requirements

- Python 3.8+
- wkhtmltopdf (for PDF export)

### API Requirements

- OpenAI API access
- Tavily API access

### Python Packages

See `requirements.txt` for the complete list of dependencies.

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.
