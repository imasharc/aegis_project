"""
Enhanced Internal Security Agent Module

This module provides sophisticated security procedure analysis capabilities through a clean,
modular architecture that mirrors the excellence of your GDPR agent refactor but adapted
for security procedure implementation workflows and organizational patterns.

The module demonstrates how architectural patterns can be consistently applied
across different domains to create maintainable, reliable, and powerful systems.

Key Components:
- InternalSecurityAgent: Main orchestrator with sophisticated component coordination
- InternalSecurityVectorStoreConnector: Specialized vector database interaction
- InternalSecurityMetadataProcessor: Intelligent procedural metadata reconstruction  
- InternalSecurityContentAnalyzer: Guided content structure analysis
- InternalSecurityCitationBuilder: Precise procedure citation creation
- InternalSecurityResponseParser: Enhanced LLM response processing

Usage:
    from backend.agent.internal_security import create_enhanced_internal_security_agent
    
    agent = create_enhanced_internal_security_agent(db_path, logger)
    result = agent.process(state)
"""

from .internal_security_agent import InternalSecurityAgent, create_enhanced_internal_security_agent
from .internal_security_vector_store_connector import InternalSecurityVectorStoreConnector, create_internal_security_vector_store_connector
from .internal_security_metadata_processor import InternalSecurityMetadataProcessor, create_internal_security_metadata_processor
from .internal_security_content_analyzer import InternalSecurityContentAnalyzer, create_internal_security_content_analyzer
from .internal_security_citation_builder import InternalSecurityCitationBuilder, create_internal_security_citation_builder
from .internal_security_response_parser import InternalSecurityResponseParser, create_internal_security_response_parser

# Main interface - most users only need this
__all__ = [
    'create_enhanced_internal_security_agent',
    'InternalSecurityAgent'
]

# Advanced interface - for users who need component-level access
__advanced__ = [
    'InternalSecurityVectorStoreConnector', 'create_internal_security_vector_store_connector',
    'InternalSecurityMetadataProcessor', 'create_internal_security_metadata_processor', 
    'InternalSecurityContentAnalyzer', 'create_internal_security_content_analyzer',
    'InternalSecurityCitationBuilder', 'create_internal_security_citation_builder',
    'InternalSecurityResponseParser', 'create_internal_security_response_parser'
]

# Version info
__version__ = '2.0.0'
__author__ = 'Enhanced Architecture Team'
__description__ = 'Sophisticated security procedure analysis with modular component architecture'