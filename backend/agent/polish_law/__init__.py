"""
Enhanced Polish Law Agent Module

This module provides sophisticated Polish law analysis capabilities through a clean,
modular architecture that mirrors the excellence of the GDPR agent refactoring while
adapting to Polish legal document patterns and citation requirements.

The module demonstrates how architectural patterns can be consistently applied
across different legal domains while respecting their unique characteristics.

Key Components:
- PolishLawAgent: Main orchestrator with sophisticated component coordination
- PolishLawVectorStoreConnector: Specialized vector database interaction for Polish law
- PolishLawMetadataProcessor: Intelligent metadata reconstruction with Polish law patterns
- PolishLawContentAnalyzer: Guided content structure analysis for Polish legal documents
- PolishLawCitationBuilder: Precise Polish legal citation creation with proper formatting
- PolishLawResponseParser: Enhanced LLM response processing for Polish law contexts

Polish Law Specific Features:
- Section-aware document organization (unique to Polish law structure)
- Gazette reference integration for legal authenticity
- Polish legal numbering pattern recognition
- Parliament session and amendment tracking
- Polish legal citation formatting standards

Usage:
    from backend.agent.polish_law import create_enhanced_polish_law_agent
    
    agent = create_enhanced_polish_law_agent(db_path, logger)
    result = agent.process(state)
"""

from .polish_law_agent import PolishLawAgent, create_enhanced_polish_law_agent
from .polish_law_vector_store_connector import PolishLawVectorStoreConnector, create_polish_law_vector_store_connector
from .polish_law_metadata_processor import PolishLawMetadataProcessor, create_polish_law_metadata_processor
from .polish_law_content_analyzer import PolishLawContentAnalyzer, create_polish_law_content_analyzer
from .polish_law_citation_builder import PolishLawCitationBuilder, create_polish_law_citation_builder
from .polish_law_response_parser import PolishLawResponseParser, create_polish_law_response_parser

# Main interface - most users only need this
__all__ = [
    'create_enhanced_polish_law_agent',
    'PolishLawAgent'
]

# Advanced interface - for users who need component-level access
__advanced__ = [
    'PolishLawVectorStoreConnector', 'create_polish_law_vector_store_connector',
    'PolishLawMetadataProcessor', 'create_polish_law_metadata_processor', 
    'PolishLawContentAnalyzer', 'create_polish_law_content_analyzer',
    'PolishLawCitationBuilder', 'create_polish_law_citation_builder',
    'PolishLawResponseParser', 'create_polish_law_response_parser'
]

# Version info
__version__ = '2.0.0'
__author__ = 'Enhanced Architecture Team'
__description__ = 'Sophisticated Polish law analysis with modular component architecture'