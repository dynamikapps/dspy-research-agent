# Advanced Research Agent

This advanced research agent combines DSPy, ReAct, Tavily Search, and Crawl4AI to generate comprehensive, well-structured research reports. It features a multi-stage pipeline with context-aware content generation and source attribution.

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

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Create a `.env` file with your API keys:

```
OPENAI_API_KEY=your_openai_api_key
TAVILY_API_KEY=your_tavily_api_key
```

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

Run the advanced research agent:

```bash
python research_agent_advance_example.py
```

The agent will:

1. Accept your research query
2. Generate a research plan using ReAct
3. Gather and analyze relevant content
4. Generate a structured outline
5. Create content with proper citations
6. Save a comprehensive markdown report

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

## Error Handling

The agent includes comprehensive error handling for:

- API failures
- Content processing issues
- File operations
- Citation processing

## Customization

You can customize:

- Search depth and result count
- Report formatting
- Content processing parameters
- Crawling configurations

## Requirements

- Python 3.8+
- DSPy
- Tavily API access
- OpenAI API access
- Required Python packages (see requirements.txt)
