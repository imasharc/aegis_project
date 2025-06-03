# polish_law_agent.py (root directory - compatibility wrapper)
"""
Backward Compatibility Wrapper for Enhanced Polish Law Agent

This wrapper maintains the existing interface while delegating to the 
enhanced modular architecture. This allows existing code to work unchanged
while providing access to enhanced capabilities specifically adapted for
Polish legal document analysis.

Polish Law Specific Features:
- Section-aware document analysis and citation creation
- Polish legal terminology recognition and validation  
- Gazette reference integration for legal authenticity
- Parliament session and amendment context tracking
- Polish legal numbering pattern recognition
- Enhanced support for Polish legal document organizational patterns

Following the same proven pattern as the GDPR agent wrapper, this demonstrates
how architectural consistency enables seamless migration across different
legal domains while respecting their unique characteristics.
"""

import os
import logging
from datetime import datetime
from typing import Dict, Any

# Import the enhanced agent from the new location
from agent.polish_law import create_enhanced_polish_law_agent

class PolishLawAgent:
    """
    Backward compatibility wrapper for the enhanced Polish law agent.
    
    This class maintains the exact same interface as the original monolithic
    agent while delegating all operations to the sophisticated component
    architecture specifically adapted for Polish legal documents. This demonstrates 
    how good architectural design enables gradual migration without system disruption
    while respecting the unique requirements of different legal systems.
    """
    
    def __init__(self):
        """Initialize the compatibility wrapper with automatic configuration for Polish law."""
        # Set up logging to match existing patterns
        self.logger = self._setup_compatibility_logging()
        
        # Determine database path automatically for Polish law documents
        db_path = self._detect_database_path()
        
        # Create the enhanced agent using the new architecture
        self.enhanced_agent = create_enhanced_polish_law_agent(db_path, self.logger)
        
        # Connect and validate the enhanced components
        if not self.enhanced_agent.connect_and_validate():
            self.logger.warning("Enhanced Polish law agent validation failed - some features may be limited")
        
        self.logger.info("Polish Law Agent compatibility wrapper initialized with enhanced backend")
    
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process method maintaining exact same interface as original agent.
        
        This method delegates to the enhanced agent while maintaining complete
        backward compatibility for Polish law analysis. Existing code continues 
        to work unchanged while gaining access to sophisticated Polish law-specific
        features like section awareness and gazette reference integration.
        """
        # Delegate to enhanced agent - same interface, enhanced capabilities
        return self.enhanced_agent.process(state)
    
    def _setup_compatibility_logging(self) -> logging.Logger:
        """Set up logging compatible with existing system patterns."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger = logging.getLogger(f"PolishLawAgentCompatibility_{timestamp}")
        
        # Use existing logging configuration if available
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        
        return logger
    
    def _detect_database_path(self) -> str:
        """Automatically detect the Polish law database path from standard locations."""
        # Check common locations for the Polish law database
        possible_paths = [
            os.path.join(os.path.dirname(__file__), "data", "polish_law_db"),
            os.path.join(os.path.dirname(__file__), "backend", "data", "polish_law_db"),
            os.path.join("data", "polish_law_db")
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                self.logger.info(f"Found Polish law database at: {path}")
                return path
        
        # Default fallback
        default_path = os.path.join(os.path.dirname(__file__), "data", "polish_law_db")
        self.logger.warning(f"Polish law database not found, using default: {default_path}")
        return default_path
    
    # Add any other methods that existing code might be calling
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics from the enhanced Polish law agent."""
        if hasattr(self.enhanced_agent, 'get_agent_statistics'):
            return self.enhanced_agent.get_agent_statistics()
        return {}