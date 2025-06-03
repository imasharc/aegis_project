# summarization_agent.py (root directory - compatibility wrapper)
"""
Backward Compatibility Wrapper for Enhanced Summarization Agent

This wrapper maintains the existing interface while delegating to the 
enhanced modular architecture. This allows existing code to work unchanged
while providing access to enhanced capabilities for multi-domain integration.
"""

import os
import logging
from datetime import datetime
from typing import Dict, Any

# Import the enhanced agent from the new location
from backend.agent.summarization import create_enhanced_summarization_agent

class SummarizationAgent:
    """
    Backward compatibility wrapper for the enhanced summarization agent.
    
    This class maintains the exact same interface as the original monolithic
    agent while delegating all operations to the sophisticated component
    architecture. This demonstrates how good architectural design enables
    gradual migration without system disruption for multi-domain workflows.
    """
    
    def __init__(self):
        """Initialize the compatibility wrapper with automatic configuration."""
        # Set up logging to match existing patterns
        self.logger = self._setup_compatibility_logging()
        
        # Create the enhanced agent using the new architecture
        self.enhanced_agent = create_enhanced_summarization_agent(self.logger)
        
        self.logger.info("Summarization Agent compatibility wrapper initialized with enhanced backend")
    
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process method maintaining exact same interface as original agent.
        
        This method delegates to the enhanced agent while maintaining complete
        backward compatibility. Existing code continues to work unchanged.
        """
        # Delegate to enhanced agent - same interface, enhanced capabilities
        return self.enhanced_agent.process(state)
    
    def _setup_compatibility_logging(self) -> logging.Logger:
        """Set up logging compatible with existing system patterns."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger = logging.getLogger(f"SummarizationAgentCompatibility_{timestamp}")
        
        # Use existing logging configuration if available
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        
        return logger
    
    # Add any other methods that existing code might be calling
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics from the enhanced agent."""
        if hasattr(self.enhanced_agent, 'get_agent_statistics'):
            return self.enhanced_agent.get_agent_statistics()
        return {}
    
    def create_numbered_citations(self, gdpr_citations, polish_law_citations, internal_policy_citations):
        """Backward compatibility method for citation creation."""
        # This method exists in the original for compatibility
        if hasattr(self.enhanced_agent, 'citation_manager'):
            return self.enhanced_agent.citation_manager.create_unified_citation_system(
                gdpr_citations, polish_law_citations, internal_policy_citations
            )
        return [], ""
    
    def detect_enhanced_citation_precision(self, citation):
        """Backward compatibility method for precision detection."""
        # This method exists in the original for compatibility  
        if hasattr(self.enhanced_agent, 'precision_analyzer'):
            # Use the enhanced precision analyzer capabilities
            reference = citation.get("reference", "").lower()
            
            # Legal precision indicators
            legal_precision = any(indicator in reference for indicator in [
                "paragraph", "sub-paragraph", "(a)", "(b)", "(c)", "(d)", "(e)", "(f)",
                "(1)", "(2)", "(3)", "(4)", "(5)", "chapter", "section"
            ])
            
            # Procedural precision indicators
            procedural_precision = any(indicator in reference for indicator in [
                "step", "configuration", "implementation", "phase", "procedure",
                "process", "workflow", "requirement"
            ])
            
            return legal_precision or procedural_precision
        
        # Fallback to basic detection
        return "paragraph" in citation.get("reference", "").lower()