"""
Internal Security Procedural Metadata Flattening Engine

This module contains the same core innovation as the GDPR and Polish law flatteners, but adapted 
for internal security procedures. The ability to take complex, nested procedural metadata 
structures and "flatten" them into simple key-value pairs that vector databases can handle, 
while preserving all the sophisticated implementation information for later reconstruction.

The fascinating aspect of this implementation is how it adapts the universal flattening principle 
to internal security procedure patterns. While legal documents focus on articles and paragraphs, 
security procedures focus on implementation steps, required tools, configuration settings, and 
security workflows. This demonstrates the flexibility and adaptability of the flattening 
approach across completely different domains.

Think of this as a "universal translator" for procedural document structures - the same core 
algorithm works across different organizational domains (legal, procedural, technical) by 
adapting to their specific patterns and organizational principles.
"""

import json
import logging
from typing import Dict, Any, List, Set


class InternalSecurityMetadataFlattener:
    """
    Handles the intelligent flattening of complex internal security procedural metadata.
    
    This class encapsulates the sophisticated logic for transforming nested procedural
    metadata structures into vector database-compatible formats while preserving all the
    information needed for precise procedure citation creation in the internal security context.
    
    The flattening approach is "lossless" for procedural information - no implementation 
    details are discarded, they're just reorganized for technical compatibility while 
    respecting internal security procedure organizational patterns and implementation workflows.
    
    Unlike legal document flattening which focuses on structural elements (articles, paragraphs),
    procedural flattening focuses on implementation elements (steps, tools, configurations,
    responsible roles, and security controls).
    """
    
    def __init__(self, logger: logging.Logger):
        """
        Initialize the internal security metadata flattener.
        
        Args:
            logger: Configured logger for tracking flattening operations
        """
        self.logger = logger
        self.logger.info("Internal Security Metadata Flattener initialized")
        
        # Track flattening statistics across all procedural operations
        # This helps us understand the patterns and complexity in security procedure documents
        self.flattening_stats = {
            'total_processed': 0,
            'enhanced_procedures_found': 0,
            'complexity_distribution': {},
            'implementation_patterns_found': set(),  # Track security-specific patterns
            'tool_usage_patterns': set(),           # Track required tools patterns
            'workflow_patterns': set(),             # Track workflow complexity patterns
            'flattening_errors': 0
        }
    
    def flatten_procedure_structure(self, procedure_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Intelligently flatten complex internal security procedure metadata for vector database compatibility.
        
        This function applies the same "lossless flattening" approach as the legal systems but is 
        specifically adapted for internal security procedural patterns. Security procedures often have 
        different organizational hierarchies compared to legal documents - they focus on implementation 
        steps, required tools, configuration settings, responsible roles, and security workflows rather 
        than articles and paragraphs.
        
        The beauty of this approach is that we maintain the same conceptual framework (extract simple 
        indicators while preserving complete structure) but adapt the pattern recognition to internal 
        security procedural conventions. This demonstrates how good architectural patterns can be 
        successfully adapted across completely different domains and organizational structures.
        
        Args:
            procedure_metadata: Complex nested metadata structure from security procedure processing
            
        Returns:
            Flattened metadata dictionary with both simple indicators and preserved structure
        """
        self.logger.debug("Starting internal security procedure structure flattening process")
        self.flattening_stats['total_processed'] += 1
        
        # Initialize flattened structure with safe defaults for security procedures
        flattened = self._create_default_flattened_structure()
        
        # Handle empty or invalid input gracefully
        if not self._validate_input_structure(procedure_metadata):
            self.logger.debug("No complex internal security procedure structure to flatten")
            return flattened
        
        try:
            # Extract and flatten the procedural information using security-specific patterns
            self._extract_basic_procedural_indicators(procedure_metadata, flattened)
            self._analyze_implementation_complexity(procedure_metadata, flattened)
            self._analyze_tool_and_role_requirements(procedure_metadata, flattened)
            self._preserve_complete_procedural_structure(procedure_metadata, flattened)
            
            # Update statistics for monitoring and optimization
            self._update_flattening_statistics(flattened)
            
            self.logger.debug(f"Successfully flattened security procedure: {flattened['implementation_step_count']} steps, "
                            f"complexity: {flattened['procedure_complexity']}")
            
            return flattened
            
        except Exception as e:
            self.flattening_stats['flattening_errors'] += 1
            self.logger.warning(f"Error flattening internal security procedure structure: {e}")
            # Return minimal structure to ensure processing continues gracefully
            flattened['procedure_structure_json'] = json.dumps(procedure_metadata) if procedure_metadata else ''
            return flattened
    
    def _create_default_flattened_structure(self) -> Dict[str, Any]:
        """
        Create the default flattened structure template for internal security procedures.
        
        This provides a consistent structure that all flattening operations return, ensuring 
        predictable behavior throughout the system. The structure is adapted for procedural 
        documents while maintaining compatibility with the vector database constraints.
        
        The structure focuses on procedural elements rather than legal elements, demonstrating
        how the same architectural pattern adapts to different document types and organizational needs.
        """
        return {
            'has_enhanced_procedure': False,
            'implementation_step_count': 0,
            'has_sub_steps': False,
            'required_tools_count': 0,
            'responsible_roles_count': 0,
            'procedure_complexity': 'simple',  # simple, moderate, complex
            'workflow_type': '',              # sequential, parallel, conditional, etc.
            'procedure_structure_json': ''    # Complete structure preserved as string
        }
    
    def _validate_input_structure(self, procedure_metadata: Any) -> bool:
        """
        Validate that the input structure is suitable for procedural flattening.
        
        This ensures we only attempt to flatten valid, non-empty procedural structures,
        preventing errors and providing consistent behavior across all processing scenarios.
        Security procedures may have different validation requirements than legal documents.
        """
        return procedure_metadata and isinstance(procedure_metadata, dict)
    
    def _extract_basic_procedural_indicators(self, procedure_metadata: Dict[str, Any], flattened: Dict[str, Any]) -> None:
        """
        Extract basic procedural indicators from the complex security procedure metadata.
        
        These indicators provide quick access to essential procedural information without requiring 
        deserialization of the complete structure. This optimization allows the citation system to 
        make quick decisions about procedure complexity and processing approach for security workflows.
        
        Unlike legal indicators (paragraph counts, sub-paragraphs), procedural indicators focus on
        implementation elements (step counts, tool requirements, role assignments).
        """
        # Check if this metadata contains implementation steps (indicates enhanced procedure)
        implementation_steps = procedure_metadata.get('implementation_steps', [])
        if implementation_steps and len(implementation_steps) > 0:
            flattened['has_enhanced_procedure'] = True
            flattened['implementation_step_count'] = len(implementation_steps)
            
            self.logger.debug(f"Extracted basic procedural indicators: {len(implementation_steps)} implementation steps")
        else:
            self.logger.debug("No implementation steps found - basic procedure structure")
    
    def _analyze_implementation_complexity(self, procedure_metadata: Dict[str, Any], flattened: Dict[str, Any]) -> None:
        """
        Analyze implementation step structure to understand security procedure-specific patterns and complexity.
        
        Internal security procedures have distinct structural patterns that differ from legal documents. 
        For example, security procedures often have sequential implementation steps with tool requirements, 
        configuration settings, validation steps, and conditional workflows based on system environments. 
        This analysis captures those patterns for efficient processing while maintaining compatibility 
        with the universal citation system.
        
        This method demonstrates how domain expertise can be encoded into the flattening process without 
        breaking the overall architectural pattern, but adapting it to procedural rather than legal contexts.
        """
        implementation_steps = procedure_metadata.get('implementation_steps', [])
        if not implementation_steps:
            return
        
        # Initialize analysis variables for security procedure patterns
        has_sub_steps = False
        complexity_indicators = []
        implementation_patterns = []
        workflow_indicators = []
        
        # Analyze each implementation step for security-specific patterns
        for step in implementation_steps:
            if isinstance(step, dict):
                self._analyze_single_implementation_step(
                    step, has_sub_steps, complexity_indicators, 
                    implementation_patterns, workflow_indicators
                )
        
        # Store the analysis results with security procedure adaptations
        flattened['has_sub_steps'] = has_sub_steps
        flattened['procedure_complexity'] = self._determine_procedural_complexity(complexity_indicators)
        flattened['workflow_type'] = self._determine_workflow_type(workflow_indicators)
        
        # Update global statistics with security-specific patterns
        if implementation_patterns:
            self.flattening_stats['implementation_patterns_found'].update(implementation_patterns)
        if workflow_indicators:
            self.flattening_stats['workflow_patterns'].update(workflow_indicators)
            
        self.logger.debug(f"Security procedure complexity analysis: {len(complexity_indicators)} indicators, "
                        f"patterns: {implementation_patterns[:3] if implementation_patterns else []}")
    
    def _analyze_single_implementation_step(self, step: Dict[str, Any], has_sub_steps: bool,
                                          complexity_indicators: List[str], implementation_patterns: List[str],
                                          workflow_indicators: List[str]) -> None:
        """
        Analyze a single implementation step for security-specific complexity indicators.
        
        This method identifies patterns specific to internal security procedure structure, such as
        configuration steps, validation requirements, tool dependencies, and conditional workflows
        that are characteristic of security implementation procedures.
        
        Understanding these patterns is crucial for creating precise procedure citations that follow
        internal security documentation standards while maintaining compatibility with the universal
        citation system architecture.
        """
        # Check for sub-step structure (common in security procedures)
        if any(key.startswith('step_') for key in step.keys() if isinstance(step.get(key), dict)):
            has_sub_steps = True
            complexity_indicators.append('sub_steps')
            implementation_patterns.append('hierarchical_steps')
        
        # Check for configuration complexity
        if step.get('configuration_by_level') or step.get('configuration_settings'):
            complexity_indicators.append('complex_configuration')
            implementation_patterns.append('configuration_dependent')
        
        # Check for tool requirements (very common in security procedures)
        required_tools = step.get('required_tools', [])
        if required_tools:
            if len(required_tools) > 3:
                complexity_indicators.append('many_tools')
            implementation_patterns.append('tool_dependent')
            
            # Track specific tool categories
            self.flattening_stats['tool_usage_patterns'].update(required_tools)
        
        # Check for validation and verification steps
        if step.get('validation') or step.get('verification_steps'):
            complexity_indicators.append('validation_required')
            implementation_patterns.append('validation_workflow')
        
        # Check for automation capabilities
        if step.get('automation_tools') or step.get('automated_execution'):
            implementation_patterns.append('automation_enabled')
            workflow_indicators.append('automated')
        
        # Check for monitoring and alerting
        if step.get('monitoring') or step.get('alerting_config'):
            complexity_indicators.append('monitoring_enabled')
            implementation_patterns.append('monitoring_workflow')
        
        # Check for conditional execution
        if step.get('conditional_execution') or step.get('prerequisites'):
            complexity_indicators.append('conditional_logic')
            workflow_indicators.append('conditional')
        
        # Check for parallel execution capabilities
        if step.get('parallel_execution') or step.get('concurrent_tasks'):
            workflow_indicators.append('parallel')
        
        # Check for rollback procedures
        if step.get('rollback_procedure') or step.get('recovery_steps'):
            complexity_indicators.append('rollback_capable')
            implementation_patterns.append('recovery_workflow')
    
    def _determine_procedural_complexity(self, complexity_indicators: List[str]) -> str:
        """
        Determine the overall procedural complexity based on identified security implementation indicators.
        
        This classification helps citation agents quickly filter and prioritize security procedures
        based on their implementation complexity, enabling more efficient processing strategies
        for procedures of different complexity levels in security operations.
        """
        if len(complexity_indicators) == 0:
            return 'simple'
        elif len(complexity_indicators) <= 3:
            return 'moderate'
        else:
            return 'complex'
    
    def _determine_workflow_type(self, workflow_indicators: List[str]) -> str:
        """
        Determine the primary workflow type based on identified patterns.
        
        This helps understand how the security procedure should be executed and what
        kind of implementation approach is most appropriate for the specific workflow pattern.
        """
        if not workflow_indicators:
            return 'sequential'
        elif 'parallel' in workflow_indicators:
            return 'parallel'
        elif 'conditional' in workflow_indicators:
            return 'conditional'
        elif 'automated' in workflow_indicators:
            return 'automated'
        else:
            return 'sequential'
    
    def _analyze_tool_and_role_requirements(self, procedure_metadata: Dict[str, Any], flattened: Dict[str, Any]) -> None:
        """
        Analyze tool and role requirements across the entire security procedure.
        
        This analysis provides insights into the operational requirements for implementing the 
        security procedure, which is crucial information for procedure citation and implementation 
        planning. Unlike legal documents, security procedures have significant operational dependencies.
        """
        # Collect all required tools across all implementation steps
        all_tools = set()
        implementation_steps = procedure_metadata.get('implementation_steps', [])
        
        for step in implementation_steps:
            if isinstance(step, dict):
                step_tools = step.get('required_tools', [])
                if isinstance(step_tools, list):
                    all_tools.update(step_tools)
                elif isinstance(step_tools, str):
                    all_tools.add(step_tools)
        
        flattened['required_tools_count'] = len(all_tools)
        
        # Collect responsible roles information
        responsible_roles = procedure_metadata.get('responsible_roles', [])
        if isinstance(responsible_roles, list):
            flattened['responsible_roles_count'] = len(responsible_roles)
        else:
            flattened['responsible_roles_count'] = 0
        
        self.logger.debug(f"Tool and role analysis: {len(all_tools)} tools, "
                        f"{flattened['responsible_roles_count']} roles")
    
    def _preserve_complete_procedural_structure(self, procedure_metadata: Dict[str, Any], flattened: Dict[str, Any]) -> None:
        """
        Preserve the complete procedural structure as a JSON string for full reconstruction.
        
        This is the most critical part of the procedural flattening process - ensuring that no 
        implementation information is lost. The complete structure is serialized and stored as a 
        string, which can be deserialized later when precise procedure citations are needed. This 
        approach allows us to have the best of both worlds: simple metadata for database compatibility 
        and complete procedural structure for sophisticated analysis.
        
        This demonstrates the same key principle as legal document flattening: when faced with 
        competing constraints (database simplicity vs. procedural richness), find a solution that 
        satisfies both rather than compromising on either.
        """
        try:
            flattened['procedure_structure_json'] = json.dumps(procedure_metadata)
            self.logger.debug("Complete internal security procedure structure preserved as JSON string")
        except Exception as e:
            self.logger.warning(f"Failed to serialize complete procedure structure: {e}")
            flattened['procedure_structure_json'] = ''
    
    def _update_flattening_statistics(self, flattened: Dict[str, Any]) -> None:
        """
        Update global statistics about the procedural flattening process.
        
        These statistics help monitor the effectiveness of the flattening approach for security
        procedures and identify patterns in the processed internal security documents. This data 
        can inform optimizations and help understand the characteristics of the procedure collection 
        being processed.
        """
        if flattened['has_enhanced_procedure']:
            self.flattening_stats['enhanced_procedures_found'] += 1
            
            complexity = flattened['procedure_complexity']
            self.flattening_stats['complexity_distribution'][complexity] = \
                self.flattening_stats['complexity_distribution'].get(complexity, 0) + 1
    
    def get_flattening_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the procedural flattening operations performed.
        
        This provides insights into the quality and patterns of the processed internal security
        procedures, helping understand how well the flattening approach is working for procedural
        documents and what types of implementation patterns are most common in the procedure collection.
        """
        stats = dict(self.flattening_stats)
        
        # Convert sets to lists for JSON serialization
        stats['implementation_patterns_found'] = list(self.flattening_stats['implementation_patterns_found'])
        stats['tool_usage_patterns'] = list(self.flattening_stats['tool_usage_patterns'])
        stats['workflow_patterns'] = list(self.flattening_stats['workflow_patterns'])
        
        # Calculate enhancement rate for security procedures
        if stats['total_processed'] > 0:
            enhancement_rate = (stats['enhanced_procedures_found'] / stats['total_processed']) * 100
            stats['enhancement_rate_percent'] = round(enhancement_rate, 1)
        else:
            stats['enhancement_rate_percent'] = 0
        
        return stats
    
    def log_flattening_summary(self) -> None:
        """
        Log a comprehensive summary of all internal security procedure flattening operations.
        
        This provides visibility into how well the procedural flattening process worked across all 
        processed documents, highlighting any security-specific patterns that were discovered and 
        preserved. This information is valuable for both debugging and understanding the characteristics 
        of the security procedure collection.
        """
        stats = self.get_flattening_statistics()
        
        self.logger.info("=== INTERNAL SECURITY PROCEDURAL METADATA FLATTENING SUMMARY ===")
        self.logger.info(f"Total procedural structures processed: {stats['total_processed']}")
        self.logger.info(f"Enhanced procedures found: {stats['enhanced_procedures_found']}")
        self.logger.info(f"Enhancement rate: {stats['enhancement_rate_percent']}%")
        self.logger.info(f"Flattening errors: {stats['flattening_errors']}")
        
        if stats['complexity_distribution']:
            self.logger.info("Procedural complexity distribution:")
            for complexity, count in sorted(stats['complexity_distribution'].items()):
                self.logger.info(f"  - {complexity}: {count} procedures")
        
        if stats['implementation_patterns_found']:
            patterns = stats['implementation_patterns_found'][:5]  # Show first 5 patterns
            self.logger.info(f"Implementation patterns found: {', '.join(patterns)}")
            
        if stats['tool_usage_patterns']:
            tools = list(stats['tool_usage_patterns'])[:5]  # Show first 5 tools
            self.logger.info(f"Common tools identified: {', '.join(tools)}")
            
        if stats['workflow_patterns']:
            workflows = stats['workflow_patterns']
            self.logger.info(f"Workflow patterns: {', '.join(workflows)}")


def create_internal_security_metadata_flattener(logger: logging.Logger) -> InternalSecurityMetadataFlattener:
    """
    Factory function to create a configured internal security metadata flattener.
    
    This provides a clean interface for creating flattener instances with proper 
    dependency injection. The factory pattern ensures consistent initialization 
    and makes it easy to modify the creation process if needed in the future.
    
    The same factory pattern works effectively across different document types,
    demonstrating the flexibility and consistency of good architectural choices.
    """
    return InternalSecurityMetadataFlattener(logger)