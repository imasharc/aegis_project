"""
Internal Security Document Converter with Intelligent Procedural Metadata Integration

This module handles the conversion of enhanced internal security procedure JSON sections into 
LangChain Document objects while applying the sophisticated procedural metadata flattening approach.

Following the same proven pattern as the GDPR and Polish law converters, this creates "bilingual 
documents" that work efficiently with vector databases while preserving all the information needed 
for precise procedure citations. The converter adapts the universal conversion process to internal 
security procedure-specific document patterns and organizational structures.

This demonstrates a key architectural principle: the same conversion framework can be adapted to 
different domains (legal vs. procedural) by changing the domain-specific processing logic while 
maintaining the same overall structure and error handling patterns. The procedural focus brings 
unique challenges around implementation steps, tool dependencies, and workflow complexity that 
differ significantly from legal document processing.

The converter showcases how procedural knowledge management differs from legal knowledge management
while still benefiting from the same architectural patterns and design principles.
"""

import os
import logging
from typing import Dict, List, Any
from datetime import datetime
from langchain.docstore.document import Document

from internal_security_metadata_flattener import InternalSecurityMetadataFlattener


class InternalSecurityDocumentConverter:
    """
    Converts enhanced internal security procedure JSON sections to LangChain Documents with intelligent procedural metadata flattening.
    
    This class represents the core solution to the vector database constraint challenge for internal 
    security procedure documents. We transform sophisticated nested procedural metadata into a format 
    that Chroma can store while preserving all the information the enhanced citation system needs 
    for precise procedure references and implementation guidance.
    
    The conversion process creates documents that can "speak" both the vector database language 
    (simple key-value pairs) and the sophisticated procedural analysis language (complex nested 
    implementation structures) simultaneously. This dual capability enables both efficient search 
    and precise implementation guidance for security procedures.
    
    Unlike legal document conversion which focuses on structural elements, procedural document 
    conversion must handle implementation workflows, tool dependencies, role assignments, and 
    configuration requirements that are essential for actionable security procedure citations.
    """
    
    def __init__(self, metadata_flattener: InternalSecurityMetadataFlattener, logger: logging.Logger):
        """
        Initialize the internal security document converter.
        
        The dependency injection pattern used here is crucial for maintainability and testability.
        By injecting the metadata flattener rather than creating it internally, we make the converter 
        testable and flexible. This also demonstrates the Single Responsibility Principle - this class 
        handles conversion, while the flattener handles the complex procedural metadata transformation logic.
        
        The separation is particularly important for procedural documents because the flattening logic
        for implementation steps and security workflows is significantly more complex than legal
        document flattening, requiring specialized domain knowledge about security operations.
        
        Args:
            metadata_flattener: Configured procedural metadata flattener for processing complex structures
            logger: Configured logger for tracking conversion operations
        """
        self.metadata_flattener = metadata_flattener
        self.logger = logger
        self.logger.info("Internal Security Document Converter initialized with intelligent procedural metadata flattening")
        
        # Track conversion statistics to monitor system performance for procedural documents
        # These statistics help identify patterns and optimize the conversion process for security procedures
        self.conversion_stats = {
            'total_sections': 0,
            'successful_conversions': 0,
            'enhanced_procedure_count': 0,
            'section_types': {},
            'complexity_levels': {},
            'errors': 0,
            'procedural_specific_patterns': {}  # Track security procedure-specific patterns
        }
    
    def convert_sections_to_documents(self, sections: List[Dict[str, Any]], 
                                    source_metadata: Dict[str, Any],
                                    processing_timestamp: str) -> List[Document]:
        """
        Convert enhanced internal security procedure JSON sections to LangChain Document objects with intelligent procedural metadata flattening.
        
        This function represents the core solution to the vector database constraint challenge for 
        internal security procedure documents. We transform sophisticated nested procedural metadata 
        into a format that Chroma can store while preserving all the information your enhanced 
        citation system needs for precise procedure references and implementation guidance.
        
        The process demonstrates how complex procedural transformations can be broken down into 
        manageable, well-tested steps. Each section goes through validation, procedural metadata 
        enhancement, flattening, and final document creation, with comprehensive error handling 
        at each stage. This approach ensures that implementation knowledge is preserved while 
        maintaining database compatibility.
        
        The procedural focus requires different validation and enhancement patterns compared to 
        legal documents, emphasizing actionable implementation steps rather than legal provisions.
        
        Args:
            sections: List of enhanced JSON sections from internal security procedure processing
            source_metadata: Document-level metadata for context
            processing_timestamp: Timestamp for tracking processing sessions
            
        Returns:
            List of LangChain Document objects ready for embedding
        """
        self.logger.info("Starting conversion of enhanced internal security procedure JSON sections to LangChain Document objects...")
        self.logger.info("Implementing intelligent procedural metadata flattening for vector database compatibility...")
        
        self.conversion_stats['total_sections'] = len(sections)
        docs = []
        
        # Process each section individually with comprehensive error handling
        # This approach ensures that a problem with one section doesn't break the entire conversion
        # This is particularly important for procedural documents where implementation complexity varies significantly
        for i, section in enumerate(sections):
            try:
                document = self._convert_single_section(section, source_metadata, processing_timestamp, i)
                if document:
                    docs.append(document)
                    self.conversion_stats['successful_conversions'] += 1
                    
            except Exception as e:
                self.conversion_stats['errors'] += 1
                self.logger.error(f"Error converting internal security procedure section {i}: {str(e)}")
                continue  # Continue processing other sections even if one fails
        
        # Log comprehensive conversion statistics
        self._log_conversion_results()
        
        return docs
    
    def _convert_single_section(self, section: Dict[str, Any], source_metadata: Dict[str, Any],
                              processing_timestamp: str, section_index: int) -> Document:
        """
        Convert a single section to a LangChain Document with enhanced procedural metadata.
        
        This method handles the detailed work of creating a properly formatted document with both 
        flattened metadata for database compatibility and preserved complex structure for sophisticated 
        analysis. The step-by-step approach makes the conversion process transparent and debuggable,
        which is particularly important for procedural documents where implementation details matter significantly.
        
        The method demonstrates how complex procedural processes can be broken down into clear, 
        manageable steps that each have a single, well-defined responsibility. The procedural focus 
        requires different validation and processing patterns compared to legal documents.
        """
        content = section.get('content', '')
        metadata = section.get('metadata', {})
        
        # Step 1: Validate content quality before proceeding
        # This early validation prevents creating empty or invalid documents that would affect procedure implementation
        if not self._validate_section_content(content, section_index):
            return None
        
        # Step 2: Build the enhanced but flattened metadata structure for security procedures
        # This creates the foundation metadata that includes both security procedure-specific
        # information and processing context needed for sophisticated implementation analysis
        enhanced_metadata = self._build_enhanced_procedural_metadata(metadata, source_metadata, 
                                                                   processing_timestamp, section_index)
        
        # Step 3: Apply intelligent procedural metadata flattening for complex implementation structures
        # This is where the magic happens for procedural documents - complex nested implementation 
        # structures get transformed into simple key-value pairs while preserving all the information
        self._apply_procedural_metadata_flattening(metadata, enhanced_metadata, section_index)
        
        # Step 4: Track statistics for this conversion
        # This helps us understand patterns and optimize the conversion process for security procedures
        self._update_conversion_statistics(metadata, enhanced_metadata)
        
        # Step 5: Create the final document with flattened but complete procedural metadata
        document = Document(
            page_content=content.strip(),
            metadata=enhanced_metadata
        )
        
        # Log sample conversion details for the first few documents
        # This provides visibility into how the conversion process is working for procedural documents
        if section_index < 3:
            self._log_sample_conversion(enhanced_metadata, content, section_index)
        
        return document
    
    def _validate_section_content(self, content: str, section_index: int) -> bool:
        """
        Validate that the section content is suitable for processing.
        
        This ensures we don't create empty or invalid documents that would cause issues in the 
        vector database. Early validation saves time and prevents problems downstream in the 
        processing pipeline. Content validation is particularly important for procedural documents 
        because they must contain actionable implementation information to be useful.
        
        Security procedure content validation differs from legal content validation because we're
        looking for implementation guidance rather than legal provisions. The content must be
        substantive enough to provide meaningful procedural guidance.
        """
        if not content or not content.strip():
            self.logger.warning(f"Empty content in internal security procedure section {section_index}, skipping...")
            return False
        
        # Additional validation for procedural documents
        # Check for minimum content length to ensure meaningful implementation guidance
        if len(content.strip()) < 20:
            self.logger.warning(f"Internal security procedure section {section_index} content too short ({len(content)} chars), skipping...")
            return False
        
        # Check for procedural content indicators to ensure this contains implementation guidance
        procedural_indicators = [
            'configure', 'install', 'implement', 'setup', 'create', 'enable', 'disable',
            'step', 'procedure', 'process', 'workflow', 'requirement', 'tool', 'access'
        ]
        
        content_lower = content.lower()
        has_procedural_content = any(indicator in content_lower for indicator in procedural_indicators)
        
        if not has_procedural_content:
            self.logger.warning(f"Internal security procedure section {section_index} appears to lack procedural content")
            # Don't skip, but log the warning for quality monitoring
        
        return True
    
    def _build_enhanced_procedural_metadata(self, section_metadata: Dict[str, Any], 
                                          source_metadata: Dict[str, Any],
                                          processing_timestamp: str, section_index: int) -> Dict[str, Any]:
        """
        Build the enhanced metadata structure that preserves all essential procedural information.
        
        This creates the foundation metadata that includes both internal security procedure-specific 
        information and processing context needed for sophisticated implementation analysis. The metadata 
        structure is designed to be both comprehensive and database-compatible, balancing completeness 
        with simplicity for procedural documents.
        
        The metadata design demonstrates how to balance completeness with simplicity for procedural 
        documents, ensuring that the citation system has all the implementation information it needs 
        while maintaining compatibility with vector database constraints. Procedural metadata differs 
        significantly from legal metadata in its focus on actionable implementation details.
        """
        # Track section type for statistical analysis and processing optimization
        section_type = section_metadata.get('type', 'unknown')
        
        # Create the comprehensive metadata structure for internal security procedure documents
        # This structure includes both standard procedural document metadata and
        # security procedure-specific organizational information for implementation guidance
        enhanced_metadata = {
            # Basic internal security procedure document structure (always simple values for database compatibility)
            'type': section_type,
            'section_number': section_metadata.get('section_number', ''),
            'section_title': section_metadata.get('section_title', ''),
            'procedure_number': section_metadata.get('procedure_number', ''),
            'procedure_title': section_metadata.get('procedure_title', ''),
            'subsection_count': section_metadata.get('subsection_count', 0),
            'policy_reference': section_metadata.get('policy_reference', ''),
            
            # Internal security procedure-specific context for implementation research
            # These fields help the citation system understand the security procedure context
            'document_type': 'internal_security_procedures',
            'document_title': source_metadata.get('document_title', ''),
            'document_id': source_metadata.get('document_id', ''),
            'version': source_metadata.get('version', ''),
            'last_updated': source_metadata.get('last_updated', ''),
            'approved_by': source_metadata.get('approved_by', ''),
            'classification_level': source_metadata.get('classification_level', ''),
            
            # Security procedure system-specific metadata for implementation context
            'compliance_frameworks': self._flatten_list_field(source_metadata.get('compliance_frameworks', [])),
            'applicable_systems': self._flatten_list_field(source_metadata.get('applicable_systems', [])),
            'review_cycle': source_metadata.get('review_cycle', ''),
            
            # Processing metadata for debugging and optimization
            # This information helps track the processing pipeline and debug issues
            'section_index': section_index,
            'processing_timestamp': processing_timestamp
        }
        
        return enhanced_metadata
    
    def _flatten_list_field(self, field_value: Any) -> str:
        """
        Flatten list fields into comma-separated strings for database compatibility.
        
        Vector databases work better with string fields than list fields, so we convert
        lists to comma-separated strings while preserving the information. This approach
        maintains the data while ensuring database compatibility for procedural metadata.
        """
        if isinstance(field_value, list):
            return ', '.join(str(item) for item in field_value)
        elif field_value:
            return str(field_value)
        else:
            return ''
    
    def _apply_procedural_metadata_flattening(self, section_metadata: Dict[str, Any], 
                                            enhanced_metadata: Dict[str, Any], section_index: int) -> None:
        """
        Apply the sophisticated procedural metadata flattening to complex implementation structures.
        
        This is where we take sophisticated nested procedural metadata and apply the flattening algorithm 
        to make it compatible with vector databases while preserving all the information needed for 
        precise procedure citations. The flattening process is the core innovation that makes the entire 
        procedural citation system possible.
        
        The process demonstrates how complex technical challenges can be solved through intelligent design - 
        we transform the problem (complex procedural metadata vs. simple database) into a solution that 
        satisfies both requirements simultaneously. The procedural focus requires different flattening 
        patterns compared to legal documents, emphasizing implementation workflows rather than legal structures.
        """
        # Check if this section has procedural metadata that needs flattening
        has_procedural_metadata = any(key in section_metadata for key in [
            'implementation_steps', 'required_tools', 'responsible_roles', 'configuration_settings'
        ])
        
        if has_procedural_metadata:
            # Apply our sophisticated flattening algorithm adapted for security procedures
            # The flattener understands security procedure-specific patterns and preserves them
            flattened_structure = self.metadata_flattener.flatten_procedure_structure(section_metadata)
            
            # Merge flattened structure into the document metadata
            # This creates the "bilingual" metadata that works with both simple and complex systems
            enhanced_metadata.update(flattened_structure)
            
            # Track that this section has enhanced procedural structure for statistics
            self.conversion_stats['enhanced_procedure_count'] += 1
            
            self.logger.debug(f"Enhanced internal security procedure section {section_index}: "
                           f"Procedure {enhanced_metadata.get('procedure_number', 'N/A')} "
                           f"with {flattened_structure.get('implementation_step_count', 0)} steps, "
                           f"complexity: {flattened_structure.get('procedure_complexity', 'unknown')}")
        else:
            # No enhanced procedural structure - set basic indicators for compatibility
            # This ensures consistent metadata structure across all documents
            enhanced_metadata.update({
                'has_enhanced_procedure': False,
                'implementation_step_count': 0,
                'has_sub_steps': False,
                'required_tools_count': 0,
                'responsible_roles_count': 0,
                'procedure_complexity': 'simple',
                'workflow_type': '',
                'procedure_structure_json': ''
            })
    
    def _update_conversion_statistics(self, section_metadata: Dict[str, Any], 
                                    enhanced_metadata: Dict[str, Any]) -> None:
        """
        Update conversion statistics for monitoring and reporting.
        
        These statistics help track the effectiveness of the conversion process and identify patterns 
        in the processed documents. The data is invaluable for optimizing the system and understanding 
        the characteristics of the internal security procedure collection being processed.
        
        Statistics collection is a best practice in data processing systems because it provides 
        visibility into system performance and helps identify issues before they become serious problems.
        The procedural focus requires different statistics compared to legal documents.
        """
        # Track section types to understand procedural document composition
        section_type = section_metadata.get('type', 'unknown')
        self.conversion_stats['section_types'][section_type] = \
            self.conversion_stats['section_types'].get(section_type, 0) + 1
        
        # Track complexity levels if enhanced procedural structure is present
        if enhanced_metadata.get('has_enhanced_procedure', False):
            complexity = enhanced_metadata.get('procedure_complexity', 'unknown')
            self.conversion_stats['complexity_levels'][complexity] = \
                self.conversion_stats['complexity_levels'].get(complexity, 0) + 1
        
        # Track security procedure-specific patterns
        if enhanced_metadata.get('procedure_number'):
            self.conversion_stats['procedural_specific_patterns']['procedures_found'] = \
                self.conversion_stats['procedural_specific_patterns'].get('procedures_found', 0) + 1
        
        if enhanced_metadata.get('classification_level'):
            classification = enhanced_metadata['classification_level']
            classification_key = f'classification_{classification.lower()}'
            self.conversion_stats['procedural_specific_patterns'][classification_key] = \
                self.conversion_stats['procedural_specific_patterns'].get(classification_key, 0) + 1
        
        if enhanced_metadata.get('compliance_frameworks'):
            self.conversion_stats['procedural_specific_patterns']['compliance_frameworks'] = \
                self.conversion_stats['procedural_specific_patterns'].get('compliance_frameworks', 0) + 1
    
    def _log_sample_conversion(self, enhanced_metadata: Dict[str, Any], 
                              content: str, section_index: int) -> None:
        """
        Log details about sample conversions for verification and debugging.
        
        This provides visibility into how the conversion process is working for the first few documents, 
        helping identify any issues early. Sample logging is a practical compromise between having 
        complete visibility and avoiding information overload in the logs.
        
        The logging demonstrates how to provide just enough information for debugging without overwhelming 
        the system with excessive detail. For procedural documents, we focus on implementation-specific 
        information rather than structural information.
        """
        has_procedure = enhanced_metadata.get('has_enhanced_procedure', False)
        section_type = enhanced_metadata.get('type', 'unknown')
        procedure_number = enhanced_metadata.get('procedure_number', 'N/A')
        complexity = enhanced_metadata.get('procedure_complexity', 'unknown')
        
        self.logger.info(f"Sample internal security procedure section {section_index}: {section_type} - "
                      f"Procedure {procedure_number} - "
                      f"Enhanced: {'✓' if has_procedure else '✗'} - "
                      f"Complexity: {complexity} - "
                      f"Content: {len(content)} chars")
        
        # Log security procedure-specific information
        if enhanced_metadata.get('classification_level'):
            self.logger.info(f"  Classification: {enhanced_metadata['classification_level']}")
        
        if enhanced_metadata.get('compliance_frameworks'):
            self.logger.info(f"  Compliance frameworks: {enhanced_metadata['compliance_frameworks']}")
        
        if enhanced_metadata.get('implementation_step_count', 0) > 0:
            self.logger.info(f"  Implementation steps: {enhanced_metadata['implementation_step_count']}")
    
    def _log_conversion_results(self) -> None:
        """
        Log comprehensive conversion results for transparency and monitoring.
        
        This provides a complete picture of how well the conversion process worked and helps identify 
        any issues that need attention. Comprehensive reporting is essential for maintaining system 
        quality and identifying optimization opportunities for procedural document processing.
        
        The logging structure demonstrates how to present complex statistical information in a clear, 
        actionable format that helps both developers and system administrators understand system 
        performance for procedural documents.
        """
        stats = self.conversion_stats
        
        self.logger.info("=" * 60)
        self.logger.info("ENHANCED INTERNAL SECURITY PROCEDURAL METADATA CONVERSION COMPLETED")
        self.logger.info("=" * 60)
        self.logger.info(f"Total sections processed: {stats['total_sections']}")
        self.logger.info(f"Successful conversions: {stats['successful_conversions']}")
        self.logger.info(f"Enhanced procedures: {stats['enhanced_procedure_count']}")
        self.logger.info(f"Conversion errors: {stats['errors']}")
        
        # Log section type distribution for internal security procedure documents
        if stats['section_types']:
            self.logger.info("Internal security procedure section type distribution:")
            for section_type, count in sorted(stats['section_types'].items()):
                self.logger.info(f"  - {section_type}: {count} sections")
        
        # Log complexity distribution for procedural documents
        if stats['complexity_levels']:
            self.logger.info("Internal security procedure complexity level distribution:")
            for complexity, count in sorted(stats['complexity_levels'].items()):
                self.logger.info(f"  - {complexity}: {count} sections")
        
        # Log security procedure-specific patterns found
        if stats['procedural_specific_patterns']:
            self.logger.info("Internal security procedure-specific patterns found:")
            for pattern, count in sorted(stats['procedural_specific_patterns'].items()):
                self.logger.info(f"  - {pattern}: {count} instances")
        
        # Calculate and log enhancement rate for procedural documents
        if stats['successful_conversions'] > 0:
            enhancement_rate = (stats['enhanced_procedure_count'] / stats['successful_conversions']) * 100
            self.logger.info(f"Internal security procedure enhanced structure rate: {enhancement_rate:.1f}%")
        
        # Log metadata flattening summary from the flattener component
        self.metadata_flattener.log_flattening_summary()
    
    def get_conversion_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the conversion process.
        
        This method provides access to all the statistics collected during the conversion process, 
        making it possible for other components to incorporate this information into their own 
        reporting and analysis. This is particularly important for procedural documents where 
        implementation complexity varies significantly.
        
        Returns:
            Dictionary containing detailed conversion statistics
        """
        stats = dict(self.conversion_stats)
        
        # Add procedural metadata flattening statistics from the flattener component
        # This demonstrates how modular components can share information
        flattening_stats = self.metadata_flattener.get_flattening_statistics()
        stats['procedural_metadata_flattening'] = flattening_stats
        
        return stats


def create_internal_security_document_converter(metadata_flattener: InternalSecurityMetadataFlattener, 
                                              logger: logging.Logger) -> InternalSecurityDocumentConverter:
    """
    Factory function to create a configured internal security document converter.
    
    This provides a clean interface for creating converter instances with proper dependency injection 
    of the metadata flattener and logger. The factory pattern ensures consistent initialization and 
    makes it easy to modify the creation process if needed in the future.
    
    The factory pattern is particularly valuable in complex systems like procedural document processing 
    because it centralizes object creation logic and makes it easier to manage dependencies and 
    configuration across the entire application.
    """
    return InternalSecurityDocumentConverter(metadata_flattener, logger)