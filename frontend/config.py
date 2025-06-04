"""
Configuration module for the Enhanced Multi-Agent Compliance System Frontend

This module centralizes all configuration settings, making them easy to modify
and maintain. By separating configuration from logic, we follow the principle
of "configuration as data" which makes the application more flexible and
easier to deploy in different environments.

Why separate configuration?
- Easy environment management (dev, staging, production)
- Single source of truth for settings
- No magic numbers scattered throughout the code
- Easier to modify behavior without touching business logic
"""

# Backend API Configuration
BACKEND_URL = "http://localhost:8000"
BACKEND_TIMEOUT = 120  # seconds - allow time for sophisticated analysis
HEALTH_CHECK_TIMEOUT = 10  # seconds - quick health check

# Streamlit Page Configuration
PAGE_CONFIG = {
    "page_title": "Enhanced Compliance Analysis",
    "page_icon": "🛡️",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# UI Text and Messages
UI_MESSAGES = {
    "title": "🛡️ Enhanced Compliance Analysis System",
    "description": """
    **Sophisticated multi-agent analysis** combining GDPR expertise, Polish law knowledge, 
    and internal security procedures to provide comprehensive compliance guidance.
    
    *Now featuring advanced citation formatting for professional documentation.*
    """,
    "processing_steps": """**Processing Steps:**
1. 🇪🇺 GDPR Agent: Analyzing European data protection requirements
2. 🇵🇱 Polish Law Agent: Reviewing Polish implementation specifics
3. 🔒 Security Agent: Evaluating internal procedure requirements
4. 📊 Integration Agent: Creating comprehensive action plan
5. 🔗 Citation Agent: Formatting authoritative references""",
    "query_placeholder": """Example: "We're implementing employee monitoring software in our Warsaw office that tracks productivity metrics and integrates with our German cloud service provider. What specific GDPR compliance steps do we need, and how do our internal security procedures apply to this cross-border data processing scenario?\"""",
    "query_help": "Describe your business scenario with specific details about location, technology, and data types for the most accurate analysis with properly formatted citations."
}

# Validation Rules
VALIDATION = {
    "min_query_length": 10,
    "max_history_display": 3  # number of previous queries to show
}

# Citation Processing Configuration
CITATION_CONFIG = {
    "citation_style": "numbered",
    "citation_pattern": r'AUTHORITATIVE SOURCE CITATIONS:\s*\n\n(.*?)(?=\n\n[A-Z]|\Z)',
    "source_pattern": r'([^:]+:)\s*\[(\d+)\s+with[^]]*\]\s*(.*?)(?=\n\n[^:]+:|\Z)',
    "individual_pattern": r'\[(\d+)\]\s*([^[]+?)(?=\[|\Z)'
}