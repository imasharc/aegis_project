"""
Internal Security Document Loader and Validator

This module handles loading and validating enhanced internal security procedure JSON files 
with sophisticated procedural metadata. Following the same proven pattern as the GDPR and 
Polish law loaders, this module's only responsibility is to safely load and validate 
internal security procedure documents.

Key responsibilities:
- Load internal security procedure JSON files with comprehensive error handling
- Validate expected enhanced procedural metadata structure specific to security implementations
- Provide detailed logging about document structure and implementation step quality
- Return validated document data and sections for further processing

The validation logic is adapted for internal security procedure patterns while maintaining
the same robust error handling and logging approach established in the legal document systems.
This demonstrates how the same architectural pattern can be adapted to different domains
(legal vs. procedural) while maintaining consistency and reliability.
"""

import os
import json
import logging
from typing import Dict, List, Any, Tuple
from datetime import datetime


class InternalSecurityDocumentLoader:
    """
    Handles loading and validation of enhanced internal security procedure JSON files.
    
    This class encapsulates all the logic for safely loading internal security documents
    and validating that they contain the expected enhanced procedural metadata structure
    that the citation system depends on. The validation is specifically tailored for
    internal security procedure patterns and organizational structure.
    
    Unlike legal documents which focus on articles and paragraphs, this loader understands
    procedural documents that focus on implementation steps, tools, configurations, and
    security workflows. This adaptation demonstrates the flexibility of the loader pattern
    across different document types.
    """
    
    def __init__(self, logger: logging.Logger):
        """
        Initialize the internal security document loader.
        
        Args:
            logger: Configured logger instance for detailed operation tracking
        """
        self.logger = logger
        self.logger.info("Internal Security Document Loader initialized")
    
    def load_and_validate_security_json(self, file_path: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Load and validate enhanced internal security procedure JSON file with sophisticated procedural metadata.
        
        This method reads your carefully crafted internal security JSON that contains both content
        and rich procedural metadata about security implementations. We validate the structure to
        ensure it contains the enhanced metadata we expect for creating precise procedure citations
        in the internal security context.
        
        The validation process understands that security procedures have different organizational
        patterns compared to legal documents - they focus on implementation steps, required tools,
        configuration settings, and security workflows rather than legal articles and provisions.
        
        Args:
            file_path: Path to the enhanced internal security procedure JSON file
            
        Returns:
            Tuple of (document_data, sections) where:
            - document_data: Complete document with procedural metadata
            - sections: List of processed sections ready for conversion
            
        Raises:
            FileNotFoundError: If the JSON file doesn't exist
            ValueError: If the JSON structure is invalid
            Exception: For other loading/validation errors
        """
        self.logger.info(f"Loading enhanced internal security procedure JSON file from: {file_path}")
        
        # Validate file exists before attempting to load
        if not os.path.exists(file_path):
            error_msg = f"Enhanced internal security procedure JSON file not found: {file_path}"
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        try:
            # Load the JSON file with proper encoding handling
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract and validate document metadata specific to internal security procedures
            document_metadata = data.get('document_metadata', {})
            self._log_document_metadata(document_metadata)
            
            # Extract and validate sections with procedural metadata patterns
            sections = data.get('sections', [])
            self._validate_sections_structure(sections)
            
            self.logger.info("Enhanced internal security procedure JSON loading and validation completed successfully")
            return data, sections
            
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON format in internal security procedure file: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        except Exception as e:
            error_msg = f"Error loading enhanced internal security procedure JSON file: {str(e)}"
            self.logger.error(error_msg)
            raise
    
    def _log_document_metadata(self, metadata: Dict[str, Any]) -> None:
        """
        Log comprehensive information about the internal security document metadata.
        
        This helps track what kind of procedural document we're processing and validates
        that it contains the expected metadata structure for internal security procedures.
        The logging is tailored to highlight security procedure-specific attributes that
        differ from legal documents.
        """
        self.logger.info("=== INTERNAL SECURITY DOCUMENT METADATA ===")
        self.logger.info(f"Document title: {metadata.get('document_title', 'Unknown title')}")
        self.logger.info(f"Document ID: {metadata.get('document_id', 'Unknown ID')}")
        self.logger.info(f"Version: {metadata.get('version', 'Unknown version')}")
        self.logger.info(f"Last updated: {metadata.get('last_updated', 'Unknown date')}")
        self.logger.info(f"Total sections: {metadata.get('total_sections', 'Unknown')}")
        self.logger.info(f"Total procedures: {metadata.get('total_procedures', 'Unknown')}")
        self.logger.info(f"Approved by: {metadata.get('approved_by', 'Unknown')}")
        
        # Log security procedure-specific metadata fields
        if metadata.get('classification_level'):
            self.logger.info(f"Classification level: {metadata['classification_level']}")
        if metadata.get('compliance_frameworks'):
            frameworks = metadata['compliance_frameworks']
            if isinstance(frameworks, list):
                self.logger.info(f"Compliance frameworks: {', '.join(frameworks)}")
            else:
                self.logger.info(f"Compliance frameworks: {frameworks}")
        if metadata.get('review_cycle'):
            self.logger.info(f"Review cycle: {metadata['review_cycle']}")
        if metadata.get('applicable_systems'):
            systems = metadata['applicable_systems']
            if isinstance(systems, list):
                self.logger.info(f"Applicable systems: {', '.join(systems)}")
            else:
                self.logger.info(f"Applicable systems: {systems}")
        
        # Log any additional procedural metadata fields that might be present
        security_specific_fields = [
            'document_title', 'document_id', 'version', 'last_updated', 'total_sections',
            'total_procedures', 'approved_by', 'classification_level', 'compliance_frameworks',
            'review_cycle', 'applicable_systems'
        ]
        
        for key, value in metadata.items():
            if key not in security_specific_fields:
                self.logger.debug(f"Additional procedural metadata - {key}: {value}")
    
    def _validate_sections_structure(self, sections: List[Dict[str, Any]]) -> None:
        """
        Validate the structure and quality of internal security procedure sections.
        
        This method performs comprehensive validation to ensure the sections contain
        the enhanced procedural metadata that the citation system needs. The validation
        is specifically adapted for internal security procedure structure and implementation
        patterns, which focus on actionable steps rather than legal provisions.
        """
        self.logger.info(f"Validating structure of {len(sections)} internal security procedure sections...")
        
        if not sections:
            self.logger.warning("No sections found in internal security document - this may indicate a processing issue")
            return
        
        # Track validation statistics specific to internal security procedures
        validation_stats = {
            'total_sections': len(sections),
            'sections_with_enhanced_metadata': 0,
            'sections_with_content': 0,
            'section_types': {},
            'procedures_found': set(),
            'sections_found': set(),
            'implementation_steps_found': 0,
            'validation_errors': []
        }
        
        # Validate each section with security procedure-specific checks
        for i, section in enumerate(sections):
            try:
                self._validate_single_section(section, i, validation_stats)
            except Exception as e:
                error_msg = f"Validation error in internal security section {i}: {str(e)}"
                validation_stats['validation_errors'].append(error_msg)
                self.logger.warning(error_msg)
        
        # Log comprehensive validation results
        self._log_validation_results(validation_stats)
    
    def _validate_single_section(self, section: Dict[str, Any], index: int, stats: Dict[str, Any]) -> None:
        """
        Validate a single section and update validation statistics.
        
        This method checks for internal security procedure-specific structural elements
        and patterns while maintaining the same validation rigor as the legal document systems.
        The focus is on procedural implementation elements rather than legal structural elements.
        
        Args:
            section: The section to validate
            index: Index of the section for error reporting
            stats: Statistics dictionary to update
        """
        # Check for required basic structure
        content = section.get('content', '')
        metadata = section.get('metadata', {})
        
        # Validate content presence and quality
        if content and content.strip():
            stats['sections_with_content'] += 1
            
            # Check for security procedure-specific content indicators
            security_indicators = [
                'configure', 'install', 'implement', 'monitor', 'access control',
                'authentication', 'authorization', 'encryption', 'security policy',
                'incident response', 'vulnerability', 'compliance', 'audit'
            ]
            
            content_lower = content.lower()
            found_indicators = [indicator for indicator in security_indicators if indicator in content_lower]
            
            if found_indicators:
                self.logger.debug(f"Section {index} contains security procedure indicators: {found_indicators[:3]}")
        else:
            self.logger.warning(f"Internal security section {index} has empty or missing content")
        
        # Track section types with security procedure-specific categories
        section_type = metadata.get('type', 'unknown')
        stats['section_types'][section_type] = stats['section_types'].get(section_type, 0) + 1
        
        # Track internal security organizational elements
        if metadata.get('procedure_number'):
            stats['procedures_found'].add(metadata['procedure_number'])
        if metadata.get('section_number'):
            stats['sections_found'].add(metadata['section_number'])
        
        # Count implementation steps if present in metadata
        implementation_steps = metadata.get('implementation_steps', [])
        if isinstance(implementation_steps, list):
            stats['implementation_steps_found'] += len(implementation_steps)
        
        # Check for enhanced procedural metadata specific to security procedures
        if metadata and any(key in metadata for key in ['implementation_steps', 'required_tools', 'responsible_roles']):
            stats['sections_with_enhanced_metadata'] += 1
            self._validate_enhanced_procedural_metadata(metadata, index)
    
    def _validate_enhanced_procedural_metadata(self, metadata: Dict[str, Any], section_index: int) -> None:
        """
        Validate the enhanced procedural metadata structure for internal security procedures.
        
        This ensures that the sophisticated procedural metadata your system depends on
        is properly formatted and contains the expected implementation information
        specific to internal security procedure organization and workflow patterns.
        
        Unlike legal metadata which focuses on structural elements, procedural metadata
        focuses on actionable implementation elements like steps, tools, and responsibilities.
        """
        if not isinstance(metadata, dict):
            self.logger.warning(f"Internal security section {section_index}: metadata is not a dictionary")
            return
        
        # Check for expected procedural elements in internal security procedures
        expected_fields = ['implementation_steps', 'required_tools', 'responsible_roles']
        present_fields = []
        
        for field in expected_fields:
            if field in metadata:
                present_fields.append(field)
        
        # Analyze implementation steps structure if present
        if 'implementation_steps' in metadata:
            implementation_steps = metadata['implementation_steps']
            if isinstance(implementation_steps, list):
                # Look for detailed step structure
                detailed_steps = 0
                for step in implementation_steps:
                    if isinstance(step, dict) and any(key in step for key in ['description', 'required_tools', 'validation']):
                        detailed_steps += 1
                
                if detailed_steps > 0:
                    self.logger.debug(f"Internal security section {section_index}: Found {detailed_steps} detailed implementation steps")
        
        # Check for tool and responsibility information
        if 'required_tools' in metadata:
            tools = metadata['required_tools']
            if isinstance(tools, list) and tools:
                self.logger.debug(f"Internal security section {section_index}: Found {len(tools)} required tools")
        
        if 'responsible_roles' in metadata:
            roles = metadata['responsible_roles']
            if isinstance(roles, list) and roles:
                self.logger.debug(f"Internal security section {section_index}: Found {len(roles)} responsible roles")
        
        if present_fields:
            self.logger.debug(f"Internal security section {section_index}: Enhanced procedural metadata contains {present_fields}")
        else:
            self.logger.debug(f"Internal security section {section_index}: Enhanced metadata present but minimal procedural structure")
    
    def _log_validation_results(self, stats: Dict[str, Any]) -> None:
        """
        Log comprehensive validation results for internal security procedure documents.
        
        This provides a clear picture of the procedural document quality and helps identify
        any issues with the enhanced procedural metadata structure specific to internal
        security procedures and implementation workflows.
        """
        self.logger.info("=== INTERNAL SECURITY SECTIONS VALIDATION RESULTS ===")
        self.logger.info(f"Total sections: {stats['total_sections']}")
        self.logger.info(f"Sections with content: {stats['sections_with_content']}")
        self.logger.info(f"Sections with enhanced procedural metadata: {stats['sections_with_enhanced_metadata']}")
        self.logger.info(f"Unique procedures found: {len(stats['procedures_found'])}")
        self.logger.info(f"Unique sections found: {len(stats['sections_found'])}")
        self.logger.info(f"Total implementation steps found: {stats['implementation_steps_found']}")
        
        # Log section type distribution for internal security procedures
        self.logger.info("Internal security section type distribution:")
        for section_type, count in sorted(stats['section_types'].items()):
            self.logger.info(f"  - {section_type}: {count} sections")
        
        # Calculate and log enhancement rate for procedural metadata
        if stats['total_sections'] > 0:
            enhancement_rate = (stats['sections_with_enhanced_metadata'] / stats['total_sections']) * 100
            self.logger.info(f"Procedural enhancement rate: {enhancement_rate:.1f}% of sections have enhanced metadata")
        
        # Log any validation errors encountered
        if stats['validation_errors']:
            self.logger.warning(f"Validation completed with {len(stats['validation_errors'])} errors:")
            for error in stats['validation_errors'][:5]:  # Show first 5 errors
                self.logger.warning(f"  - {error}")
            if len(stats['validation_errors']) > 5:
                self.logger.warning(f"  ... and {len(stats['validation_errors']) - 5} more errors")
        else:
            self.logger.info("✅ All internal security procedure sections passed validation successfully")
    
    def find_security_procedure_file(self, processed_dir: str, raw_dir: str,
                                   filename: str = "internal_security_procedures_final_manual.json") -> str:
        """
        Find the internal security procedure JSON file in the expected locations.
        
        This method implements the file discovery logic, checking multiple possible
        locations for the enhanced internal security procedure JSON file. It follows
        the same reliable pattern established in the legal document systems while
        adapting to internal security procedure file naming conventions.
        
        Args:
            processed_dir: Primary directory to check
            raw_dir: Backup directory to check
            filename: Name of the file to find
            
        Returns:
            Path to the found file
            
        Raises:
            FileNotFoundError: If file is not found in any location
        """
        primary_path = os.path.join(processed_dir, filename)
        backup_path = os.path.join(raw_dir, filename)
        
        if os.path.exists(primary_path):
            self.logger.info(f"Found internal security procedure file at primary location: {primary_path}")
            return primary_path
        elif os.path.exists(backup_path):
            self.logger.info(f"Found internal security procedure file at backup location: {backup_path}")
            return backup_path
        else:
            error_msg = f"Enhanced internal security procedure JSON file '{filename}' not found in either location:\n" \
                       f"  Primary: {primary_path}\n" \
                       f"  Backup: {backup_path}"
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)


def create_internal_security_loader(logger: logging.Logger) -> InternalSecurityDocumentLoader:
    """
    Factory function to create a configured internal security document loader.
    
    This provides a clean interface for creating loader instances with proper
    dependency injection of the logger. The factory pattern ensures consistent
    initialization across the application and demonstrates how the same patterns
    work effectively across different document types (legal vs. procedural).
    """
    return InternalSecurityDocumentLoader(logger)