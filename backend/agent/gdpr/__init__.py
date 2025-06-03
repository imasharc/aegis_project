"""
Enhanced GDPR Agent Module

This module provides sophisticated GDPR analysis capabilities through a clean,
modular architecture that mirrors the excellence of your processing pipeline refactor.

The module demonstrates how architectural patterns can be consistently applied
across different domains to create maintainable, reliable, and powerful systems.

Key Components:
- GDPRAgent: Main orchestrator with sophisticated component coordination
- GDPRVectorStoreConnector: Specialized vector database interaction
- GDPRMetadataProcessor: Intelligent metadata reconstruction  
- GDPRContentAnalyzer: Guided content structure analysis
- GDPRCitationBuilder: Precise legal citation creation
- GDPRResponseParser: Enhanced LLM response processing

Usage:
    from backend.agent.gdpr import create_enhanced_gdpr_agent
    
    agent = create_enhanced_gdpr_agent(db_path, logger)
    result = agent.process(state)
"""

from .gdpr_agent import GDPRAgent, create_enhanced_gdpr_agent
from .gdpr_vector_store_connector import GDPRVectorStoreConnector, create_gdpr_vector_store_connector
from .gdpr_metadata_processor import GDPRMetadataProcessor, create_gdpr_metadata_processor
from .gdpr_content_analyzer import GDPRContentAnalyzer, create_gdpr_content_analyzer
from .gdpr_citation_builder import GDPRCitationBuilder, create_gdpr_citation_builder
from .gdpr_response_parser import GDPRResponseParser, create_gdpr_response_parser

# Main interface - most users only need this
__all__ = [
    'create_enhanced_gdpr_agent',
    'GDPRAgent'
]

# Advanced interface - for users who need component-level access
__advanced__ = [
    'GDPRVectorStoreConnector', 'create_gdpr_vector_store_connector',
    'GDPRMetadataProcessor', 'create_gdpr_metadata_processor', 
    'GDPRContentAnalyzer', 'create_gdpr_content_analyzer',
    'GDPRCitationBuilder', 'create_gdpr_citation_builder',
    'GDPRResponseParser', 'create_gdpr_response_parser'
]

# Version info
__version__ = '2.0.0'
__author__ = 'Enhanced Architecture Team'
__description__ = 'Sophisticated GDPR analysis with modular component architecture'