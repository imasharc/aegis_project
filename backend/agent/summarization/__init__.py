"""
Enhanced Summarization Agent Module

This module provides sophisticated multi-domain citation integration and response generation
through a clean, modular architecture that demonstrates how architectural patterns can be
consistently applied across different types of agent functionality.

The module showcases how to handle the complex challenge of unifying outputs from multiple
specialized agents while preserving the precision that each agent worked hard to achieve.

Key Components:
- SummarizationAgent: Main orchestrator with sophisticated component coordination
- SummarizationCitationManager: Multi-domain citation unification and numbering
- SummarizationPrecisionAnalyzer: Cross-domain precision analysis and reporting
- SummarizationFormatter: Professional response formatting and presentation
- SummarizationStatisticsCollector: Comprehensive system performance analytics
- SummarizationResponseBuilder: LLM integration and response construction

Usage:
    from backend.agent.summarization import create_enhanced_summarization_agent
    
    agent = create_enhanced_summarization_agent(logger)
    result = agent.process(state)
"""

from .summarization_agent import SummarizationAgent, create_enhanced_summarization_agent
from .summarization_citation_manager import SummarizationCitationManager, create_summarization_citation_manager
from .summarization_precision_analyzer import SummarizationPrecisionAnalyzer, create_summarization_precision_analyzer
from .summarization_formatter import SummarizationFormatter, create_summarization_formatter
from .summarization_statistics_collector import SummarizationStatisticsCollector, create_summarization_statistics_collector
from .summarization_response_builder import SummarizationResponseBuilder, create_summarization_response_builder

# Main interface - most users only need this
__all__ = [
    'create_enhanced_summarization_agent',
    'SummarizationAgent'
]

# Advanced interface - for users who need component-level access
__advanced__ = [
    'SummarizationCitationManager', 'create_summarization_citation_manager',
    'SummarizationPrecisionAnalyzer', 'create_summarization_precision_analyzer', 
    'SummarizationFormatter', 'create_summarization_formatter',
    'SummarizationStatisticsCollector', 'create_summarization_statistics_collector',
    'SummarizationResponseBuilder', 'create_summarization_response_builder'
]

# Version info
__version__ = '2.0.0'
__author__ = 'Enhanced Architecture Team'
__description__ = 'Sophisticated multi-domain citation integration with modular component architecture'