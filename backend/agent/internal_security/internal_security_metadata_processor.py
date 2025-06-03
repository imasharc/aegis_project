"""
Internal Security Metadata Processor

This module reconstructs sophisticated procedural information from the flattened metadata
created by the processing pipeline. It serves as the "reverse" of the InternalSecurityMetadataFlattener,
taking simple key-value pairs and rebuilding the complex nested structures needed for
precise procedure citation creation.

Think of this as the "decompression" half of your metadata system:
- Processing Pipeline: Complex procedural structures → Flattened metadata (compression)
- Agent Pipeline: Flattened metadata → Reconstructed structures (decompression)

This approach demonstrates how architectural consistency creates powerful synergies
between different parts of your system, adapted specifically for security procedure
implementation workflows and organizational patterns.
"""

import json
import logging
from typing import Dict, List, Any, Optional


class InternalSecurityMetadataProcessor:
    """
    Processes and reconstructs flattened procedural metadata for sophisticated procedure citation creation.
    
    This class performs the inverse operation of the InternalSecurityMetadataFlattener, taking the
    simple key-value pairs stored in the vector database and reconstructing the complex
    procedural information needed for precise security procedure citations with implementation details.
    
    The processor demonstrates how your flattened procedural metadata approach enables sophisticated
    functionality while maintaining vector database compatibility, specifically adapted for
    security procedure implementation patterns and workflow complexity.
    """
    
    def __init__(self, logger: logging.Logger):
        """
        Initialize the internal security metadata processor.
        
        Args:
            logger: Configured logger for tracking metadata processing operations
        """
        self.logger = logger
        self.logger.info("Internal Security Metadata Processor initialized")
        
        # Track processing statistics across all operations
        self.processing_stats = {
            'total_metadata_processed': 0,
            'enhanced_procedural_structures_reconstructed': 0,
            'json_deserialization_successes': 0,
            'json_deserialization_failures': 0,
            'fallback_to_indicators': 0,
            'processing_errors': 0,
            'procedure_complexity_distribution': {}
        }
    
    def extract_and_reconstruct_procedural_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract and reconstruct sophisticated procedural information from flattened metadata.
        
        This method takes the simple key-value pairs created by your processing pipeline
        and rebuilds the complex procedural information that your citation system needs
        for creating precise security procedure references with implementation step details.
        It demonstrates how the flattened procedural metadata approach preserves all functionality
        while working within vector database constraints.
        
        Args:
            metadata: Flattened metadata dictionary from vector store document
            
        Returns:
            Reconstructed metadata with both quick indicators and full procedural structure
        """
        self.logger.debug("Processing flattened security procedural metadata for structural reconstruction")
        self.processing_stats['total_metadata_processed'] += 1
        
        # Initialize comprehensive metadata structure with safe defaults
        reconstructed_info = self._create_default_procedural_metadata_structure(metadata)
        
        # Extract quick procedural indicators for efficient processing
        if reconstructed_info['has_enhanced_procedure']:
            self._extract_quick_procedural_indicators(metadata, reconstructed_info)
            
            # Attempt to reconstruct complete structure from preserved JSON
            success = self._reconstruct_complete_procedural_structure(metadata, reconstructed_info)
            
            if success:
                self.processing_stats['enhanced_procedural_structures_reconstructed'] += 1
                complexity = reconstructed_info['quick_indicators']['procedure_complexity']
                self.processing_stats['procedure_complexity_distribution'][complexity] = \
                    self.processing_stats['procedure_complexity_distribution'].get(complexity, 0) + 1
                    
                self.logger.debug(f"Successfully reconstructed enhanced procedural structure: "
                                f"{reconstructed_info['quick_indicators']['implementation_step_count']} steps, "
                                f"complexity: {complexity}")
            else:
                self.processing_stats['fallback_to_indicators'] += 1
                self.logger.debug("Using quick procedural indicators only - complete structure unavailable")
        else:
            self.logger.debug(f"Basic procedural metadata processed: Procedure {reconstructed_info['procedure_number']} "
                            f"(no enhanced structure)")
        
        return reconstructed_info
    
    def _create_default_procedural_metadata_structure(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create the default reconstructed procedural metadata structure.
        
        This provides a consistent foundation that works whether we have enhanced
        procedural metadata or just basic document information. The structure is specifically
        designed for security procedures with their unique organizational patterns.
        """
        return {
            # Basic procedural document identifiers (always available from processing pipeline)
            'procedure_number': metadata.get('procedure_number', ''),
            'section_number': metadata.get('section_number', ''),
            'section_title': metadata.get('section_title', ''),
            'procedure_title': metadata.get('procedure_title', ''),
            'policy_reference': metadata.get('policy_reference', ''),
            'document_id': metadata.get('document_id', ''),
            'classification_level': metadata.get('classification_level', ''),
            
            # Enhancement indicators from flattened procedural metadata
            'has_enhanced_procedure': metadata.get('has_enhanced_procedure', False),
            
            # Containers for reconstructed procedural information
            'quick_indicators': {},
            'full_structure': None,
            
            # Processing context
            'reconstruction_successful': False,
            'reconstruction_method': 'none'
        }
    
    def _extract_quick_procedural_indicators(self, metadata: Dict[str, Any], 
                                           reconstructed_info: Dict[str, Any]) -> None:
        """
        Extract quick procedural indicators from flattened metadata.
        
        These indicators provide immediate access to essential procedural information
        without requiring JSON deserialization. This demonstrates how your flattening
        approach creates multiple levels of access to the same implementation information.
        """
        quick_indicators = {
            'implementation_step_count': metadata.get('implementation_step_count', 0),
            'has_sub_steps': metadata.get('has_sub_steps', False),
            'required_tools_count': metadata.get('required_tools_count', 0),
            'responsible_roles_count': metadata.get('responsible_roles_count', 0),
            'procedure_complexity': metadata.get('procedure_complexity', 'simple'),
            'workflow_type': metadata.get('workflow_type', 'sequential')
        }
        
        reconstructed_info['quick_indicators'] = quick_indicators
        reconstructed_info['reconstruction_method'] = 'quick_indicators'
        
        self.logger.debug(f"Extracted quick procedural indicators: {quick_indicators['implementation_step_count']} steps, "
                        f"complexity: {quick_indicators['procedure_complexity']}, "
                        f"workflow: {quick_indicators['workflow_type']}, "
                        f"sub-steps: {quick_indicators['has_sub_steps']}")
    
    def _reconstruct_complete_procedural_structure(self, metadata: Dict[str, Any], 
                                                 reconstructed_info: Dict[str, Any]) -> bool:
        """
        Reconstruct complete procedural information from preserved JSON.
        
        This method demonstrates the power of your flattened procedural metadata approach.
        The complete procedural structure was preserved as a JSON string during processing,
        and now we can deserialize it to access all the sophisticated implementation
        information your citation system needs for security procedures.
        """
        json_str = metadata.get('procedure_structure_json', '')
        
        if not json_str:
            self.logger.debug("No preserved JSON procedural structure available")
            return False
        
        try:
            # Deserialize the complete procedural structure that was preserved during processing
            full_structure = json.loads(json_str)
            reconstructed_info['full_structure'] = full_structure
            reconstructed_info['reconstruction_successful'] = True
            reconstructed_info['reconstruction_method'] = 'full_json_reconstruction'
            
            self.processing_stats['json_deserialization_successes'] += 1
            
            self.logger.debug("Successfully reconstructed complete procedural structure from preserved JSON")
            
            # Validate the reconstructed procedural structure for consistency
            self._validate_reconstructed_procedural_structure(full_structure, reconstructed_info)
            
            return True
            
        except json.JSONDecodeError as e:
            self.processing_stats['json_deserialization_failures'] += 1
            self.logger.warning(f"Failed to deserialize preserved JSON procedural structure: {e}")
            return False
        except Exception as e:
            self.processing_stats['processing_errors'] += 1
            self.logger.warning(f"Error during procedural structure reconstruction: {e}")
            return False
    
    def _validate_reconstructed_procedural_structure(self, full_structure: Dict[str, Any], 
                                                   reconstructed_info: Dict[str, Any]) -> None:
        """
        Validate that the reconstructed procedural structure is consistent with quick indicators.
        
        This validation ensures that the flattening and reconstruction process
        maintained data integrity throughout the pipeline for security procedures. Any inconsistencies
        could indicate issues with the processing pipeline that need attention.
        """
        if not isinstance(full_structure, dict):
            self.logger.warning("Reconstructed procedural structure is not a dictionary")
            return
        
        # Compare quick indicators with reconstructed structure for consistency
        quick_count = reconstructed_info['quick_indicators']['implementation_step_count']
        reconstructed_count = len(full_structure.get('implementation_steps', []))
        
        if quick_count != reconstructed_count:
            self.logger.warning(f"Implementation step count inconsistency: quick={quick_count}, "
                              f"reconstructed={reconstructed_count}")
        
        # Validate implementation step structure if present
        implementation_steps = full_structure.get('implementation_steps', [])
        if implementation_steps and isinstance(implementation_steps, list):
            step_with_tools = sum(1 for step in implementation_steps 
                                if isinstance(step, dict) and step.get('required_tools'))
            self.logger.debug(f"Validated procedural structure: {len(implementation_steps)} steps, "
                            f"{step_with_tools} with tool requirements")
        
        self.logger.debug("Procedural structure validation completed")
    
    def create_procedural_processing_hints(self, reconstructed_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create processing hints for content analysis based on reconstructed procedural metadata.
        
        This method transforms the reconstructed procedural information into hints
        that can guide content parsing for security procedures. It demonstrates how metadata processing
        enables intelligent, guided analysis rather than blind text parsing for procedural documents.
        
        Args:
            reconstructed_info: Reconstructed procedural metadata information
            
        Returns:
            Dictionary of processing hints for content analysis
        """
        if not reconstructed_info['has_enhanced_procedure']:
            return {
                'has_hints': False,
                'use_guided_parsing': False,
                'parsing_strategy': 'simple'
            }
        
        quick_indicators = reconstructed_info['quick_indicators']
        
        # Create comprehensive hints based on available procedural information
        hints = {
            'has_hints': True,
            'use_guided_parsing': True,
            'parsing_strategy': 'enhanced' if reconstructed_info['reconstruction_successful'] else 'indicator_guided',
            
            # Direct indicators for parsing guidance
            'implementation_step_count': quick_indicators['implementation_step_count'],
            'has_sub_steps': quick_indicators['has_sub_steps'],
            'required_tools_count': quick_indicators['required_tools_count'],
            'procedure_complexity': quick_indicators['procedure_complexity'],
            'workflow_type': quick_indicators['workflow_type'],
            
            # Parsing recommendations based on procedural structure
            'recommended_parser': self._recommend_procedural_parser_strategy(quick_indicators),
            'expected_patterns': self._identify_expected_procedural_patterns(quick_indicators),
            
            # Full procedural structure availability
            'full_structure_available': reconstructed_info['reconstruction_successful'],
            'reconstruction_method': reconstructed_info['reconstruction_method']
        }
        
        self.logger.debug(f"Created procedural processing hints: {hints['parsing_strategy']} strategy, "
                        f"parser: {hints['recommended_parser']}")
        
        return hints
    
    def _recommend_procedural_parser_strategy(self, quick_indicators: Dict[str, Any]) -> str:
        """
        Recommend the best parsing strategy based on procedural structural indicators.
        
        This method analyzes the available procedural metadata to determine which parsing
        approach will be most effective for the specific security procedure structure.
        """
        complexity = quick_indicators['procedure_complexity']
        has_sub_steps = quick_indicators['has_sub_steps']
        step_count = quick_indicators['implementation_step_count']
        workflow_type = quick_indicators['workflow_type']
        
        if complexity == 'complex' and has_sub_steps and workflow_type in ['conditional', 'parallel']:
            return 'sophisticated_with_conditional_workflow'
        elif complexity == 'complex' and has_sub_steps:
            return 'sophisticated_with_sub_steps'
        elif has_sub_steps:
            return 'guided_with_sub_steps'
        elif step_count > 3:
            return 'multi_step_procedure'
        else:
            return 'simple_single_step'
    
    def _identify_expected_procedural_patterns(self, quick_indicators: Dict[str, Any]) -> List[str]:
        """
        Identify expected procedural patterns based on metadata indicators.
        
        This helps the content analyzer know what patterns to look for,
        making parsing more reliable and efficient for security procedures.
        """
        patterns = []
        
        workflow_type = quick_indicators['workflow_type']
        has_sub_steps = quick_indicators['has_sub_steps']
        step_count = quick_indicators['implementation_step_count']
        
        # Add pattern expectations based on procedural metadata
        if has_sub_steps:
            patterns.append('hierarchical_implementation_steps')
            if workflow_type == 'conditional':
                patterns.append('conditional_sub_steps')  # If-then logic in steps
            elif workflow_type == 'parallel':
                patterns.append('parallel_sub_steps')    # Concurrent execution steps
        
        # Add security procedure-specific patterns
        patterns.append('security_procedure_structure')
        
        if step_count > 1:
            patterns.append('multi_step_implementation')
        
        if quick_indicators['required_tools_count'] > 0:
            patterns.append('tool_dependent_steps')
        
        # Add workflow-specific patterns
        if workflow_type == 'automated':
            patterns.append('automation_workflow')
        elif workflow_type == 'conditional':
            patterns.append('decision_point_workflow')
        
        return patterns
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about procedural metadata processing operations.
        
        This provides insights into how well the reconstruction process is working
        and helps identify any issues with the flattened procedural metadata approach.
        """
        stats = dict(self.processing_stats)
        
        # Calculate success rates
        if stats['total_metadata_processed'] > 0:
            enhancement_rate = (stats['enhanced_procedural_structures_reconstructed'] / stats['total_metadata_processed']) * 100
            stats['enhancement_rate_percent'] = round(enhancement_rate, 1)
            
            if stats['enhanced_procedural_structures_reconstructed'] > 0:
                json_success_rate = (stats['json_deserialization_successes'] / stats['enhanced_procedural_structures_reconstructed']) * 100
                stats['json_success_rate_percent'] = round(json_success_rate, 1)
            else:
                stats['json_success_rate_percent'] = 0
        else:
            stats['enhancement_rate_percent'] = 0
            stats['json_success_rate_percent'] = 0
        
        return stats
    
    def log_processing_summary(self) -> None:
        """
        Log a comprehensive summary of all procedural metadata processing operations.
        
        This provides visibility into how well the procedural metadata reconstruction
        process is working across all processed security procedure documents.
        """
        stats = self.get_processing_statistics()
        
        self.logger.info("=== INTERNAL SECURITY PROCEDURAL METADATA PROCESSING SUMMARY ===")
        self.logger.info(f"Total metadata processed: {stats['total_metadata_processed']}")
        self.logger.info(f"Enhanced procedural structures reconstructed: {stats['enhanced_procedural_structures_reconstructed']}")
        self.logger.info(f"Enhancement rate: {stats['enhancement_rate_percent']}%")
        self.logger.info(f"JSON reconstruction successes: {stats['json_deserialization_successes']}")
        self.logger.info(f"JSON success rate: {stats['json_success_rate_percent']}%")
        self.logger.info(f"Fallback to indicators: {stats['fallback_to_indicators']}")
        self.logger.info(f"Processing errors: {stats['processing_errors']}")
        
        # Log complexity distribution
        if stats['procedure_complexity_distribution']:
            self.logger.info("Procedure complexity distribution:")
            for complexity, count in sorted(stats['procedure_complexity_distribution'].items()):
                self.logger.info(f"  - {complexity}: {count} procedures")


def create_internal_security_metadata_processor(logger: logging.Logger) -> InternalSecurityMetadataProcessor:
    """
    Factory function to create a configured internal security metadata processor.
    
    This provides a clean interface for creating processor instances with
    proper dependency injection of the logger.
    """
    return InternalSecurityMetadataProcessor(logger)