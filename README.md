# AEGIS Project - Enhanced Multi-Agent Compliance System

## 🛡️ Project Overview

AEGIS (Advanced European Governance Intelligence System) is a sophisticated multi-agent compliance analysis platform designed specifically for European businesses navigating complex regulatory landscapes. The system combines artificial intelligence with domain expertise to provide comprehensive compliance guidance across GDPR, Polish data protection law, and internal security procedures.

### What Makes AEGIS Special?

Think of AEGIS as having three expert consultants working together seamlessly: a GDPR specialist who knows European data protection regulations inside and out, a Polish law expert who understands local implementation nuances, and a security professional who ensures your internal procedures align with regulatory requirements. These "agents" collaborate to provide you with actionable compliance guidance that considers all relevant legal and procedural dimensions.

## 🏗️ System Architecture

### The Multi-Agent Approach

AEGIS employs a **multi-agent architecture** where specialized AI agents collaborate to solve complex compliance problems. This approach mirrors how real-world compliance teams work: different experts contribute their specialized knowledge, and a coordinator synthesizes their inputs into comprehensive guidance.

**Core Agents:**
- **GDPR Agent**: Analyzes European data protection requirements using the complete GDPR regulation
- **Polish Law Agent**: Examines Polish data protection implementation specifics and local legal requirements  
- **Internal Security Agent**: Evaluates internal security procedures and operational requirements
- **Summarization Agent**: Integrates findings from all agents into actionable guidance with professional citations

**Core RAG:**
- **GDPR processing**: Retrieves the complete GDPR regulation in jsoin format, transforms it to chroma compatible structure and embeds it into vector store
- **Polish Law Agent**: Performs the same job but for Polish Law
- **Internal Security Agent**: Performs the same job but for Internal security procedures, especially that the json structure is significantly different for this document

### Technical Foundation

The system demonstrates advanced software engineering principles including:

**Backend Architecture (Python/FastAPI):**
- **Modular Design**: Each component has a single, well-defined responsibility
- **Dependency Injection**: Components receive their dependencies rather than creating them, making the system more testable and maintainable
- **Factory Pattern**: Clean instantiation of complex objects with proper configuration
- **Vector Database Integration**: Uses Chroma for sophisticated document retrieval and semantic search
- **Real-time Communication**: Server-Sent Events (SSE) for live progress updates during analysis

**Frontend Architecture (Streamlit):**
- **Progressive Disclosure**: Users can start with guided examples before creating custom queries
- **Sample Query System**: Curated business scenarios that teach effective query patterns
- **Enhanced Citation Display**: Professional formatting of legal and regulatory references
- **Graceful Error Handling**: Comprehensive error recovery and user guidance

**MCP Server (EXPERIMENTAL, MAY BREAK):**
- **Saving summary report to root/mcp_server/summarization_reports**: MAY BREAK so it was introduced as feat/mcp_server(breaking) branch and NOT MERGED but pushed. Most likely relative paths to directories are incorrect


## 📁 Project Structure

Understanding the project architecture helps you navigate the codebase effectively and appreciate how different components interact following modern software engineering principles.

```
AEGIS_PROJECT/
├── 🔧 backend/                    # AI Processing & API Services
│   ├── 🤖 agent/                  # Multi-Agent System Components
│   ├── ⚙️ processing/             # Document Processing Pipeline
│   ├── 💾 data/                   # Vector Databases & Knowledge Base (to recreate on your own)
│   ├── 🌐 api.py                  # FastAPI Server Entry Point
│   ├── 🎯 main.py                 # Multi-Agent Orchestrator
│   └── 📋 test_queries.py         # EU Business Test Scenarios
│
└── frontend/
    ├── 📱 app.py                          # Main Streamlit Application
    ├── 🧩 ui_components.py               # Enhanced UI Component Library
    ├── 📚 sample_queries.py              # Curated Business Scenarios
    ├── 🔌 backend_client.py              # API Communication Layer
    ├── 🔄 enhanced_response_handler.py   # Response Format Normalization
    ├── 📊 progress_tracking.py           # Real-Time Progress Updates
    ├── 📋 citation_parser.py             # Professional Citation Formatting
    └── ⚙️ config.py                      # Configuration Management
```

```
AEGIS_PROJECT/
├── backend/                          # Python/FastAPI backend system
│   ├── agent/                        # Multi-agent system components
│   │   ├── gdpr/                     # GDPR compliance analysis agent
│   │   ├── polish_law/               # Polish law analysis agent
│   │   ├── internal_security/        # Security procedure agent
│   │   └── summarization/            # Integration and synthesis agent
│   ├── processing/                   # Document processing pipeline
│   │   ├── gdpr/                     # GDPR document processing modules
│   │   ├── polish_law/               # Polish law processing modules
│   │   └── internal_security/        # Security procedure processing
│   ├── data/                         # Vector databases and processed documents
│   ├── main.py                       # Multi-agent orchestrator
│   ├── api.py                        # FastAPI backend with progress tracking
│   ├── test_queries.py               # EU-focused test scenarios
│   └── requirements.txt              # Python dependencies
└── frontend/                         # Streamlit user interface
    ├── app.py                        # Main application with sample integration
    ├── ui_components.py              # Enhanced UI components
    ├── backend_client.py             # Backend communication with response normalization
    ├── sample_queries.py             # Curated business scenarios
    ├── progress_tracking.py          # Real-time progress display
    ├── enhanced_response_handler.py  # Multi-format response handling
    ├── citation_parser.py            # Professional citation formatting
    ├── config.py                     # Configuration management
    └── requirements.txt              # Frontend dependencies
```

## 🔄 Data Flow Architecture

```
graph TD
    A[User Query] --> B[Sample Integration]
    B --> C[Local Validation] 
    C --> D[Backend API]
    
    D --> E[Multi-Agent Processing]
    E --> F[GDPR Agent]
    E --> G[Polish Law Agent]
    E --> H[Security Agent]
    E --> I[Summarization Agent]
    
    F --> J[Vector Search & Citation Building]
    G --> K[Document Analysis & Legal References]
    H --> L[Procedure Matching & Implementation Guidance]
    I --> M[Integration & Professional Formatting]
    
    J --> N[Response Normalization]
    K --> N
    L --> N
    M --> N
    
    N --> O[Citation Formatting]
    O --> P[Progressive Display]
```

## ✨ Key Features

### 1. Sophisticated Multi-Domain Analysis

Rather than treating compliance as a single-dimensional problem, AEGIS recognizes that real-world scenarios involve multiple overlapping regulatory domains. The system analyzes your business scenario across:

- **European Data Protection (GDPR)**: Understanding of all 99 articles, recitals, and implementation guidelines
- **Polish Legal Framework**: Local implementation specifics, national derogations, and cultural considerations
- **Internal Security Procedures**: Operational requirements that ensure regulatory compliance translates into practical implementation

### 2. Advanced Document Processing Pipeline

The system includes a sophisticated document processing infrastructure that:

**Metadata Flattening Architecture**: Converts complex hierarchical legal documents into vector database-compatible formats while preserving all structural information needed for precise citations.

**Enhanced Vector Retrieval**: Uses semantic search to find relevant legal provisions based on the meaning and context of your business scenario, not just keyword matching.

**Citation Precision**: Maintains exact legal references with article numbers, paragraph citations, and regulatory context for professional documentation.

### 3. Real-Time Progress Tracking

Unlike traditional AI systems that leave users waiting with no feedback, AEGIS provides detailed real-time updates about what's happening during analysis:

- **Agent-by-Agent Progress**: See which expert agent is currently analyzing your scenario
- **Processing Stage Details**: Understand what specific analysis step is being performed
- **Time Estimation**: Get realistic expectations about processing duration
- **Graceful Error Recovery**: If something goes wrong, you get actionable guidance rather than cryptic error messages

### 4. Progressive User Experience

The interface is designed to support both newcomers and experienced users:

**Sample Query System**: Instead of confronting users with a blank text box, AEGIS provides curated business scenarios organized by category (HR/Employment, Customer Data, IT Security, etc.). These samples teach effective query patterns and demonstrate system capabilities.

**Educational Scaffolding**: Users learn by example, starting with realistic scenarios they can modify rather than having to formulate queries from scratch.

**Enhanced Citation Formatting**: Legal and regulatory references are formatted according to professional standards, suitable for compliance documentation and legal review.

## 📚 Screenshots

![image](https://github.com/user-attachments/assets/d489aa62-f599-4328-96b8-d112a6e2cd87)

![image](https://github.com/user-attachments/assets/34bff81c-7ac8-455f-a039-36b676a3e43c)

![image](https://github.com/user-attachments/assets/65a34b1f-fd45-4150-9f1b-57d185f63317)


## 🚀 Getting Started

### Prerequisites

Before you begin, make sure you have the following installed on your system:

- **Python 3.13 or higher**: The system uses modern Python features for better performance and maintainability
- **OpenAI API Key**: Required for the AI agents to perform analysis (set as environment variable `OPENAI_API_KEY`)
- **LangSmith API Key**: Optional for monitoring the system (if you have LangSmith account follow the documentation at https://docs.smith.langchain.com/observability or directly set environment variable for `LANGSMITH_API_KEY`, `LANGSMITH_TRACING`, `LANGSMITH_PROJECT` and `LANGSMITH_PROJECT`)
- **Git**: For cloning the repository and version control

### Step-by-Step Installation

The AEGIS project can be set up using several different Python environment management approaches. Each has its own advantages, so choose the one that best fits your development workflow and preferences.

**Understanding Your Options:**

- **Traditional pip + venv**: The standard Python approach that works everywhere and is well-documented. Choose this if you're new to Python or want maximum compatibility.
- **uv**: A modern, extremely fast package installer and environment manager written in Rust. Choose this if you want faster installations and more efficient dependency resolution.]

**1. Clone and Prepare the Repository**

First, let's get the code and navigate to the project directory. This step is the same regardless of which environment management approach you choose:

```bash
# Clone the repository to your local machine
git clone https://github.com/imasharc/aegis_project.git
cd aegis-project
```

**2. Create Your Python Environment**

Open two terminals and in one navigate to backend:
```bash
cd backend
```

 in the other terminal open frontend:
 ```bash
cd backend
```

Now choose one of the following approaches based on your preference and repeat these steps for both backend and frontend:

**Option A: Traditional pip + venv (Recommended for beginners)**

This uses Python's built-in virtual environment capabilities and is the most widely supported approach:

```bash
# Create a Python virtual environment to isolate dependencies
python -m venv .venv

# Activate the virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Upgrade pip to ensure you have the latest package installer
python -m pip install --upgrade pip
```

**Option B: Using uv (Fastest installation)**

uv is a modern Python package installer that's significantly faster than pip and provides better dependency resolution:

```bash
# Install uv if you haven't already (you only need to do this once on your system)
# On macOS/Linux:
curl -LsSf https://astral.sh/uv/install.sh | sh
# On Windows (PowerShell):
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
# OR if you have pip but want to install uv
pip install uv

# Create and activate a virtual environment with uv
uv venv

# Activate the environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

**3. Set Up the Backend System**

The backend handles the sophisticated multi-agent analysis and document processing. These steps are the same regardless of which environment approach you chose above:

```bash
# You should be in /backend directory already in one terminal

# Install Python dependencies using your chosen approach
# If using traditional pip or uv:
pip install -r requirements.txt

# Alternative: If you're using uv, you can install dependencies even faster:
uv pip install -r requirements.txt

# Set up your OpenAI API key by creating a .env file
# This file stores your API key securely and keeps it out of your code
# On macOS/Linux:
echo "OPENAI_API_KEY=your_actual_api_key_here" > .env
# On Windows (Command Prompt):
echo OPENAI_API_KEY=your_actual_api_key_here > .env
# On Windows (PowerShell):
'OPENAI_API_KEY=your_actual_api_key_here' | Out-File -FilePath .env -Encoding utf8

# Process the legal documents to create the vector databases
# Note: You'll need the GDPR, Polish law, and security procedure documents
# in the data/ directory for this step to work completely
# These scripts transform complex legal documents into AI-searchable formats
uv run ./processing/gdpr/process_gdpr.py
uv run ./processing/polish_law/process_process_polish_law.py
uv run ./processing/internal_security/process_internal_security.py

# Start the backend server (this command launches the localhost sever; to run agents in the command-line go uv run main.py)
uv run api.py
```

The backend server will start on `http://localhost:8000` and provide the multi-agent analysis capabilities. You should see output indicating that all agents have connected successfully to their respective vector databases. Wait until you see
```
bash
INFO:     Application startup complete.
```
in the terminal.
Then proceed with running the frontend

**4. Set Up the Frontend Interface**

The frontend provides the user-friendly interface with sample queries and real-time progress tracking. Open a new terminal window (keeping the backend running) and follow these steps:

```bash
# You should be in /frontend directory already in the other terminal

# Activate your environment again in this new terminal
# For traditional venv:
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

# For uv: same activation commands as above


# Install frontend dependencies
# For traditional pip or conda environments:
pip install -r requirements.txt

# For uv (faster installation):
uv pip install -r requirements.txt

# Start the Streamlit application
streamlit run app.py
```

The frontend application will automatically open in your browser at `http://localhost:8501`. You should see the AEGIS interface with sample queries and system status indicators.

### Verification

To verify everything is working correctly:

1. **Check Backend Health**: Visit `http://localhost:8000/health` - you should see a JSON response indicating system status
2. **Test Frontend Connection**: The Streamlit interface should show "✅ Backend System Operational" in the sidebar
3. **Try a Sample Query**: Select one of the sample business scenarios and click "Use This Query" to test the complete workflow

## 📋 Usage Guide

### Understanding Query Effectiveness

The quality of compliance guidance you receive depends largely on how you formulate your query. AEGIS works best when you provide:

**Business Context**: Describe what you're trying to accomplish, not just the technical details. For example, "implementing employee productivity monitoring" rather than just "installing software."

**Geographic Specificity**: Mention specific locations, especially when dealing with cross-border scenarios. The system understands EU member state differences and can provide targeted guidance.

**Data Types and Processing Activities**: Be specific about what data you're collecting, processing, or transferring. The system can provide much more precise guidance when it understands the data protection implications.

**Technology Details**: Include information about cloud providers, software platforms, and third-party services. This helps the system assess cross-border data flow requirements and vendor management obligations.

### Using Sample Queries Effectively

The sample query system is designed to teach you effective query patterns:

**Browse by Business Category**: Start with the category that most closely matches your situation (Employee Monitoring, Customer Data, IT Security, etc.).

**Study the Pattern**: Notice how effective sample queries are structured - they include context, specifics, and clear questions.

**Modify Rather Than Start From Scratch**: Even experienced users often get better results by starting with a relevant sample and modifying it rather than writing entirely custom queries.

**Learn From Examples**: The samples demonstrate the level of detail and specificity that produces the most comprehensive compliance guidance.

### Interpreting Results

AEGIS provides results in several components:

**Action Plan**: Numbered, actionable steps you can follow to achieve compliance. These are designed to be implementable by your team without requiring deep legal expertise.

**Authoritative Citations**: Professional legal references that support each recommendation. These citations are formatted according to legal standards and can be included in compliance documentation.

**System Insights**: Information about the analysis process, including which agents contributed to specific recommendations and confidence levels.

**Cross-Domain Integration**: Guidance on how different regulatory domains interact in your specific scenario.

## 🛠️ Development and Customization

### Architecture Principles

The system demonstrates several important software engineering principles that make it maintainable and extensible:

**Single Responsibility Principle**: Each component has one clear job. The GDPR agent focuses solely on GDPR analysis, the citation builder handles only citation formatting, etc.

**Dependency Injection**: Components receive their dependencies rather than creating them internally. This makes the system much easier to test and modify.

**Factory Pattern**: Complex objects are created through factory functions that handle all the configuration details, making the code cleaner and more maintainable.

**Graceful Degradation**: When something goes wrong, the system provides useful fallback behavior rather than simply failing.

### Adding New Compliance Domains

To add a new compliance domain (such as CCPA for California privacy law):

1. **Create the Agent**: Follow the pattern established by `gdpr_agent.py` - create a new agent that specializes in your compliance domain
2. **Build Processing Pipeline**: Create document processing modules following the pattern in `processing/gdpr/`
3. **Integrate with Orchestrator**: Add your agent to the workflow in `main.py`
4. **Update UI Components**: Modify the frontend to display results from your new domain

### Customizing for Different Industries

The system can be adapted for specific industry requirements:

**Healthcare**: Add HIPAA compliance agents and medical data protection procedures
**Financial Services**: Integrate PCI DSS and financial regulatory requirements  
**Manufacturing**: Include industry-specific data protection and operational security requirements

### Performance Optimization

For high-volume usage, consider:

**Caching**: Implement response caching for frequently asked questions
**Load Balancing**: Deploy multiple backend instances with a load balancer
**Database Optimization**: Tune vector database settings for your query patterns
**Agent Parallelization**: Modify the workflow to run agents in parallel rather than sequentially

## 🔍 Advanced Features

### Real-Time Progress Tracking

The system implements Server-Sent Events (SSE) for real-time communication between backend and frontend. This allows users to see detailed progress during analysis rather than waiting for a final result.

**Implementation Benefits**:
- Users remain engaged during long-running analyses
- Transparent feedback about what's happening behind the scenes
- Ability to identify bottlenecks in the analysis process
- Professional user experience comparable to enterprise software

### Enhanced Response Handling

The system includes sophisticated response normalization that allows the frontend to work with different backend response formats. This demonstrates important principles for building resilient systems that can evolve over time.

**Multi-Format Support**: The frontend can handle both legacy response formats and new structured formats automatically.

**Graceful Migration**: As the system evolves, new features can be added without breaking existing functionality.

**Format Validation**: All responses are validated to ensure they meet frontend expectations before being displayed to users.

### Professional Citation Management

The citation system goes beyond simple reference lists to provide professional-grade legal citations:

**Structured Metadata**: Each citation includes complete source information, legal context, and relevance analysis.

**Cross-Reference Capability**: Citations are numbered and can be cross-referenced within action plans.

**Format Compliance**: Citations follow legal documentation standards suitable for compliance reports and legal review.

## 🤝 Contributing

### Code Quality Standards

The project maintains high code quality through:

**Comprehensive Documentation**: Every function includes detailed docstrings explaining purpose, parameters, and return values.

**Modular Architecture**: Code is organized into focused modules with clear interfaces and minimal coupling.

**Error Handling**: Comprehensive error handling with useful error messages and recovery suggestions.

**Testing Patterns**: Although not included in this version, the architecture is designed to be easily testable through dependency injection and modular design.

### Contribution Guidelines

When contributing to the project:

1. **Follow Existing Patterns**: Study how current agents and components are structured before adding new functionality
2. **Maintain Documentation**: Update docstrings and comments when modifying functionality  
3. **Test Thoroughly**: Verify that your changes work across different scenarios and edge cases
4. **Consider Backward Compatibility**: Ensure that existing functionality continues to work when adding new features

## 📊 System Performance

### Expected Performance Characteristics

**Query Processing Time**: Typical analyses complete in 15-45 seconds, depending on query complexity and system load.

**Citation Accuracy**: The system maintains high precision in legal citations through sophisticated metadata preservation during document processing.

**Memory Usage**: Vector databases require significant RAM for optimal performance (recommend 8GB+ for production use).

**Scaling Characteristics**: The system can handle multiple concurrent users, though performance may degrade with very high concurrent load due to OpenAI API rate limits.

### Optimization Recommendations

For production deployment:

**Infrastructure**: Use SSD storage for vector databases, ensure adequate RAM for document embeddings, and consider GPU acceleration for large-scale deployments.

**API Management**: Implement rate limiting and request queuing to manage OpenAI API usage efficiently.

**Monitoring**: Set up logging and monitoring to track system performance and identify optimization opportunities.

## 🔒 Security and Privacy

### Data Handling

The system is designed with privacy and security in mind:

**Local Processing**: All document analysis happens locally - your queries and results are not sent to external services except for OpenAI's analysis API.

**No Data Persistence**: User queries are not permanently stored (though they're held in session memory for context).

**Secure Communication**: All API communication uses standard HTTPS protocols.

### Deployment Security

For production deployment:

**Environment Variables**: Store sensitive configuration (API keys, database credentials) in environment variables, not in code.

**Access Control**: Implement appropriate authentication and authorization for production use.

**Network Security**: Deploy behind appropriate firewalls and network security controls.

## 📄 License and Legal

This project is provided for educational and development purposes. When deploying in production environments, ensure that:

- You have appropriate licenses for all legal and regulatory documents used in the vector databases
- Your use complies with OpenAI's terms of service
- You understand the limitations of AI-generated compliance guidance and involve appropriate legal review for critical decisions

## 🆘 Support and Troubleshooting

### Common Issues

**"Backend System Unavailable"**: Ensure the FastAPI backend is running on port 8000 and that you've set your OpenAI API key correctly.

**"No Citations Generated"**: This usually indicates that the document processing pipeline hasn't been run or that the vector databases are empty.

**"Analysis Timeout"**: Very complex queries may timeout. Try breaking your query into smaller, more focused questions.

**Performance Issues**: Ensure adequate system resources (RAM, storage) and check OpenAI API rate limit status.

### Getting Help

For technical issues:
1. Check the application logs for detailed error information
2. Verify that all prerequisites are properly installed
3. Ensure that document processing has been completed successfully
4. Review the sample queries to understand effective query patterns

The system includes comprehensive logging and error reporting to help diagnose and resolve issues quickly.

---

## 🎯 Conclusion

AEGIS represents a sophisticated approach to AI-powered compliance analysis that goes beyond simple question-answering to provide comprehensive, multi-domain guidance backed by authoritative legal sources. The system demonstrates advanced software engineering principles while remaining accessible to users who need practical compliance guidance.

Whether you're a compliance professional seeking efficient analysis tools or a developer interested in multi-agent AI systems, AEGIS provides a robust foundation for understanding and implementing sophisticated AI-powered business solutions.

The project showcases how thoughtful architecture, user-centered design, and comprehensive error handling can create AI systems that are both powerful and reliable for real-world business applications.
