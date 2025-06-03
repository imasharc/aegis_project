# gdpr_agent.py (root directory - compatibility wrapper)
"""
Backward Compatibility Wrapper for Enhanced GDPR Agent

This wrapper maintains the existing interface while delegating to the 
enhanced modular architecture. This allows existing code to work unchanged
while providing access to enhanced capabilities.
"""

import os
import logging
from datetime import datetime
from typing import Dict, Any

# Import the enhanced agent from the new location
from agent.gdpr import create_enhanced_gdpr_agent

class GDPRAgent:
    """
    Backward compatibility wrapper for the enhanced GDPR agent.
    
    This class maintains the exact same interface as the original monolithic
    agent while delegating all operations to the sophisticated component
    architecture. This demonstrates how good architectural design enables
    gradual migration without system disruption.
    """
    
    def __init__(self):
        """Initialize the compatibility wrapper with automatic configuration."""
        # Set up logging to match existing patterns
        self.logger = self._setup_compatibility_logging()
        
        # Determine database path automatically
        db_path = self._detect_database_path()
        
        # Create the enhanced agent using the new architecture
        self.enhanced_agent = create_enhanced_gdpr_agent(db_path, self.logger)
        
        # Connect and validate the enhanced components
        if not self.enhanced_agent.connect_and_validate():
            self.logger.warning("Enhanced GDPR agent validation failed - some features may be limited")
        
        self.logger.info("GDPR Agent compatibility wrapper initialized with enhanced backend")
    
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
        logger = logging.getLogger(f"GDPRAgentCompatibility_{timestamp}")
        
        # Use existing logging configuration if available
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        
        return logger
    
    def _detect_database_path(self) -> str:
        """Automatically detect the GDPR database path from standard locations."""
        # Check common locations for the database
        possible_paths = [
            os.path.join(os.path.dirname(__file__), "data", "gdpr_db"),
            os.path.join(os.path.dirname(__file__), "backend", "data", "gdpr_db"),
            os.path.join("data", "gdpr_db")
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                self.logger.info(f"Found GDPR database at: {path}")
                return path
        
        # Default fallback
        default_path = os.path.join(os.path.dirname(__file__), "data", "gdpr_db")
        self.logger.warning(f"GDPR database not found, using default: {default_path}")
        return default_path
    
    # Add any other methods that existing code might be calling
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics from the enhanced agent."""
        if hasattr(self.enhanced_agent, 'get_agent_statistics'):
            return self.enhanced_agent.get_agent_statistics()
        return {}