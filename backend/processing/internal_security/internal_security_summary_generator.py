"""
Internal Security Processing Summary Generator

This module creates comprehensive summaries of the enhanced internal security procedure document 
processing pipeline. Following the same proven pattern as the GDPR and Polish law summary generators, 
it aggregates statistics from all processing stages and generates detailed reports about the success rate, 
quality metrics, and performance of the procedural metadata flattening approach.

The summary generator demonstrates the "Observer" pattern - it watches what happens in the other modules 
and reports on the overall system health and performance. This approach provides valuable insights that 
go beyond what individual components can offer, creating a holistic view of how well the entire 
processing pipeline is working for procedural documents.

The reporting approach showcases several important principles:
- Comprehensive data aggregation from multiple sources
- Statistical analysis that provides actionable insights
- Clear presentation of complex information
- Recommendations based on observed patterns and performance metrics
- Procedural-specific analysis focusing on implementation workflows

This component is particularly valuable for system optimization and troubleshooting because it identifies 
patterns and relationships specific to security procedure processing that might not be obvious when 
looking at individual components in isolation.
"""

import json
import logging
from typing import Dict, List, Any, Set
from datetime import datetime
from langchain.docstore.document import Document


class InternalSecurityProcessingSummaryGenerator:
    """
    Generates comprehensive summaries of the internal security procedure document processing pipeline.
    
    This class aggregates statistics and metrics from all stages of the processing pipeline to create 
    detailed reports about the effectiveness of the procedural metadata flattening approach and overall 
    system performance for internal security procedure documents. The analysis goes beyond simple 
    success/failure metrics to provide insights about procedure quality, processing patterns, and 
    optimization opportunities.
    
    The summary generator demonstrates how to create valuable system insights by combining data from 
    multiple sources. Rather than just reporting what happened, it analyzes why certain patterns 
    occurred and what they mean for system performance and reliability in the context of security 
    procedure management.
    
    The comprehensive analysis includes:
    - Processing pipeline performance across all stages
    - Procedural document quality metrics and patterns
    - Procedural metadata flattening effectiveness analysis
    - Security procedure-specific implementation analysis
    - Actionable recommendations for system optimization
    """
    
    def __init__(self, logger: logging.Logger):
        """
        Initialize the internal security processing summary generator.
        
        The initialization sets up the logging infrastructure that will be used throughout the analysis 
        process. The logger is particularly important for this component because it generates complex 
        reports that benefit from detailed logging of the analysis process.
        
        Args:
            logger: Configured logger for tracking summary generation
        """
        self.logger = logger
        self.logger.info("Internal Security Processing Summary Generator initialized")
    
    def generate_comprehensive_summary(self, docs: List[Document], 
                                     processing_timestamp: str,
                                     loader_stats: Dict[str, Any] = None,
                                     flattener_stats: Dict[str, Any] = None,
                                     converter_stats: Dict[str, Any] = None,
                                     embedder_stats: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate a comprehensive summary of the entire internal security procedure processing pipeline.
        
        This method creates a detailed report that aggregates information from all processing stages 
        to provide a complete picture of how well the enhanced processing worked for internal security 
        procedure documents. The analysis demonstrates how to synthesize complex information from 
        multiple sources into actionable insights.
        
        The comprehensive approach is valuable because it reveals relationships and patterns that 
        wouldn't be obvious when looking at individual components in isolation. For example, the 
        relationship between procedure complexity and processing success rates, or the correlation 
        between metadata quality and implementation step preservation.
        
        Args:
            docs: Final processed documents for analysis
            processing_timestamp: Timestamp of the processing session
            loader_stats: Statistics from the document loader (optional)
            flattener_stats: Statistics from the procedural metadata flattener (optional)
            converter_stats: Statistics from the document converter (optional)
            embedder_stats: Statistics from the embedder (optional)
            
        Returns:
            Comprehensive summary dictionary with detailed statistics and analysis
        """
        self.logger.info("Generating comprehensive internal security procedure processing summary...")
        
        # The summary structure is designed to provide multiple perspectives on the same data,
        # making it easy to understand both high-level performance and detailed patterns
        summary = {
            "processing_timestamp": processing_timestamp,
            "generation_timestamp": datetime.now().isoformat(),
            "document_type": "internal_security_procedures",
            "pipeline_overview": self._create_pipeline_overview(docs, loader_stats, converter_stats, embedder_stats),
            "enhanced_procedural_metadata_analysis": self._analyze_enhanced_procedural_metadata(docs),
            "procedure_quality_metrics": self._calculate_procedure_quality_metrics(docs),
            "processing_stage_performance": self._analyze_stage_performance(loader_stats, flattener_stats, converter_stats, embedder_stats),
            "security_procedure_implementation_analysis": self._perform_security_procedure_implementation_analysis(docs),
            "recommendations": self._generate_recommendations(docs, flattener_stats, embedder_stats),
            "sample_enhanced_procedures": self._collect_sample_procedures(docs)
        }
        
        self.logger.info("Comprehensive internal security procedure processing summary generated successfully")
        return summary
    
    def _create_pipeline_overview(self, docs: List[Document], 
                                 loader_stats: Dict[str, Any],
                                 converter_stats: Dict[str, Any],
                                 embedder_stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a high-level overview of the entire processing pipeline.
        
        This provides key metrics that give a quick assessment of how well the processing worked 
        overall for security procedures. The overview is designed to answer the most important 
        questions first: did the processing work, how well did it work, and are there any critical 
        issues that need immediate attention.
        
        The overview approach demonstrates how to distill complex information into essential insights 
        that can be quickly understood by both technical and non-technical stakeholders.
        """
        overview = {
            "total_documents_processed": len(docs),
            "pipeline_stages_completed": 0,
            "overall_success_rate": 0.0,
            "enhanced_procedural_metadata_rate": 0.0,
            "critical_errors": [],
            "security_procedure_specific_metrics": {
                "procedures_identified": 0,
                "implementation_steps_preserved": 0,
                "procedures_with_enhanced_structure": 0,
                "classification_levels_preserved": 0
            }
        }
        
        # Count completed pipeline stages to assess pipeline completeness
        # This helps identify which stages worked and which may have failed
        stages_completed = [
            loader_stats is not None,
            converter_stats is not None, 
            embedder_stats is not None
        ]
        overview["pipeline_stages_completed"] = sum(stages_completed)
        
        # Calculate overall success rate based on document conversion success
        # This provides a single metric for overall pipeline performance
        if converter_stats:
            total_sections = converter_stats.get('total_sections', 0)
            successful_conversions = converter_stats.get('successful_conversions', 0)
            if total_sections > 0:
                overview["overall_success_rate"] = (successful_conversions / total_sections) * 100
        
        # Calculate enhanced procedural metadata rate to assess flattening effectiveness
        # This metric shows how well the sophisticated procedural metadata processing worked
        enhanced_count = sum(1 for doc in docs if doc.metadata.get('has_enhanced_procedure', False))
        if docs:
            overview["enhanced_procedural_metadata_rate"] = (enhanced_count / len(docs)) * 100
        
        # Analyze security procedure-specific metrics
        for doc in docs:
            metadata = doc.metadata
            if metadata.get('procedure_number'):
                overview["security_procedure_specific_metrics"]["procedures_identified"] += 1
            if metadata.get('implementation_step_count', 0) > 0:
                overview["security_procedure_specific_metrics"]["implementation_steps_preserved"] += metadata['implementation_step_count']
            if metadata.get('has_enhanced_procedure', False):
                overview["security_procedure_specific_metrics"]["procedures_with_enhanced_structure"] += 1
            if metadata.get('classification_level'):
                overview["security_procedure_specific_metrics"]["classification_levels_preserved"] += 1
        
        # Check for critical errors that need immediate attention
        # This helps prioritize issues that need immediate resolution
        if embedder_stats and embedder_stats.get('metadata_errors', 0) > 0:
            overview["critical_errors"].append("Procedural metadata compatibility issues detected")
        
        if converter_stats and converter_stats.get('errors', 0) > converter_stats.get('successful_conversions', 0) * 0.1:
            overview["critical_errors"].append("High document conversion error rate")
        
        return overview
    
    def _analyze_enhanced_procedural_metadata(self, docs: List[Document]) -> Dict[str, Any]:
        """
        Analyze the enhanced procedural metadata across all processed internal security procedure documents.
        
        This provides detailed insights into how well the procedural metadata flattening approach worked 
        and what types of enhanced implementation structures were preserved for security procedure documents. 
        The analysis helps understand the effectiveness of the flattening approach and identifies 
        opportunities for optimization.
        
        The procedural metadata analysis demonstrates how to extract meaningful insights from complex 
        implementation data structures by looking at patterns, distributions, and relationships rather 
        than just counting occurrences.
        """
        analysis = {
            "total_documents": len(docs),
            "enhanced_procedure_count": 0,
            "simple_procedures": 0,
            "complexity_distribution": {},
            "workflow_types": {},
            "implementation_statistics": {
                "min_steps": float('inf'),
                "max_steps": 0,
                "total_steps": 0,
                "procedures_with_sub_steps": 0
            },
            "flattening_effectiveness": {
                "json_preservation_rate": 0.0,
                "procedural_indicators_rate": 0.0
            },
            "security_procedure_patterns": {
                "tool_requirements": {},
                "role_assignments": 0,
                "automation_enabled": 0
            }
        }
        
        # Track various procedural metadata characteristics to understand document patterns
        json_preserved_count = 0
        procedural_indicators_count = 0
        
        for doc in docs:
            metadata = doc.metadata
            
            # Analyze enhanced procedural structure presence and characteristics
            if metadata.get('has_enhanced_procedure', False):
                analysis["enhanced_procedure_count"] += 1
                
                # Analyze complexity distribution to understand procedure patterns
                complexity = metadata.get('procedure_complexity', 'unknown')
                analysis["complexity_distribution"][complexity] = \
                    analysis["complexity_distribution"].get(complexity, 0) + 1
                
                # Analyze workflow types used in security procedures
                workflow_type = metadata.get('workflow_type', '')
                if workflow_type:
                    analysis["workflow_types"][workflow_type] = \
                        analysis["workflow_types"].get(workflow_type, 0) + 1
                
                # Analyze implementation step statistics for procedural understanding
                self._update_implementation_statistics(metadata, analysis["implementation_statistics"])
                
                # Check flattening effectiveness metrics
                if metadata.get('procedure_structure_json', ''):
                    json_preserved_count += 1
                
                if any(metadata.get(field, 0) > 0 for field in ['implementation_step_count', 'has_sub_steps']):
                    procedural_indicators_count += 1
                    
            else:
                analysis["simple_procedures"] += 1
            
            # Analyze security procedure-specific patterns
            self._analyze_security_procedure_patterns(metadata, analysis["security_procedure_patterns"])
        
        # Calculate flattening effectiveness rates
        if analysis["enhanced_procedure_count"] > 0:
            analysis["flattening_effectiveness"]["json_preservation_rate"] = \
                (json_preserved_count / analysis["enhanced_procedure_count"]) * 100
            analysis["flattening_effectiveness"]["procedural_indicators_rate"] = \
                (procedural_indicators_count / analysis["enhanced_procedure_count"]) * 100
        
        # Fix infinite values in implementation statistics
        if analysis["implementation_statistics"]["min_steps"] == float('inf'):
            analysis["implementation_statistics"]["min_steps"] = 0
        
        return analysis
    
    def _update_implementation_statistics(self, metadata: Dict[str, Any], stats: Dict[str, Any]) -> None:
        """
        Update implementation statistics with information from a single document.
        
        This tracks detailed information about the structure of processed procedures to understand 
        the complexity and patterns in the internal security procedures. The statistics help identify 
        optimization opportunities and validate that the processing pipeline is working correctly 
        for procedures of different complexity levels.
        """
        step_count = metadata.get('implementation_step_count', 0)
        if step_count > 0:
            stats["min_steps"] = min(stats["min_steps"], step_count)
            stats["max_steps"] = max(stats["max_steps"], step_count)
            stats["total_steps"] += step_count
            
            if metadata.get('has_sub_steps', False):
                stats["procedures_with_sub_steps"] += 1
    
    def _analyze_security_procedure_patterns(self, metadata: Dict[str, Any], patterns: Dict[str, Any]) -> None:
        """
        Analyze security procedure-specific patterns in the processed documents.
        
        This method identifies patterns that are specific to internal security procedure documents, 
        such as tool requirements, role assignments, and automation capabilities. Understanding these 
        patterns helps validate that the processing pipeline correctly handles security procedure-specific 
        organizational structures and implementation workflows.
        """
        # Track tool requirement patterns
        if metadata.get('required_tools_count', 0) > 0:
            tool_count = metadata['required_tools_count']
            if tool_count <= 2:
                category = 'minimal_tools'
            elif tool_count <= 5:
                category = 'moderate_tools'
            else:
                category = 'extensive_tools'
            
            patterns["tool_requirements"][category] = patterns["tool_requirements"].get(category, 0) + 1
        
        # Track role assignments (important for security procedures)
        if metadata.get('responsible_roles_count', 0) > 0:
            patterns["role_assignments"] += 1
        
        # Track automation capabilities
        workflow_type = metadata.get('workflow_type', '')
        if workflow_type == 'automated':
            patterns["automation_enabled"] += 1
    
    def _calculate_procedure_quality_metrics(self, docs: List[Document]) -> Dict[str, Any]:
        """
        Calculate various quality metrics for the processed internal security procedure documents.
        
        These metrics help assess the overall quality of the processing pipeline and identify any 
        issues that might affect the citation system's performance. Quality metrics are particularly 
        important for procedural document processing because accuracy and completeness are critical 
        for reliable implementation guidance.
        
        The quality analysis demonstrates how to create meaningful metrics that go beyond simple 
        success/failure counts to provide insights about the characteristics and reliability of 
        the processed procedural data.
        """
        metrics = {
            "content_quality": {
                "average_content_length": 0,
                "min_content_length": float('inf'),
                "max_content_length": 0,
                "empty_content_count": 0
            },
            "procedural_metadata_completeness": {
                "procedures_with_numbers": 0,
                "procedures_with_titles": 0,
                "sections_identified": 0,
                "complete_metadata_rate": 0.0
            },
            "section_types": {},
            "procedures_processed": set(),
            "sections_processed": set(),
            "security_procedure_quality": {
                "classification_level_rate": 0.0,
                "enhanced_procedure_rate": 0.0,
                "implementation_guidance_rate": 0.0
            }
        }
        
        total_content_length = 0
        complete_metadata_count = 0
        classification_count = 0
        implementation_guidance_count = 0
        
        for doc in docs:
            # Analyze content quality characteristics
            content_length = len(doc.page_content)
            total_content_length += content_length
            
            if content_length == 0:
                metrics["content_quality"]["empty_content_count"] += 1
            
            metrics["content_quality"]["min_content_length"] = \
                min(metrics["content_quality"]["min_content_length"], content_length)
            metrics["content_quality"]["max_content_length"] = \
                max(metrics["content_quality"]["max_content_length"], content_length)
            
            # Analyze procedural metadata completeness for security procedures
            metadata = doc.metadata
            
            if metadata.get('procedure_number'):
                metrics["procedural_metadata_completeness"]["procedures_with_numbers"] += 1
                metrics["procedures_processed"].add(metadata['procedure_number'])
            
            if metadata.get('procedure_title'):
                metrics["procedural_metadata_completeness"]["procedures_with_titles"] += 1
            
            if metadata.get('section_number'):
                metrics["procedural_metadata_completeness"]["sections_identified"] += 1
                metrics["sections_processed"].add(metadata['section_number'])
            
            # Check for complete procedural metadata (procedure number, title, content, and implementation steps)
            if (metadata.get('procedure_number') and 
                metadata.get('procedure_title') and 
                content_length > 0):
                complete_metadata_count += 1
            
            # Track security procedure-specific quality metrics
            if metadata.get('classification_level'):
                classification_count += 1
            
            # Check for implementation guidance (key for procedural documents)
            if (metadata.get('implementation_step_count', 0) > 0 or
                'implement' in doc.page_content.lower() or
                'configure' in doc.page_content.lower()):
                implementation_guidance_count += 1
            
            # Track section types
            section_type = metadata.get('type', 'unknown')
            metrics["section_types"][section_type] = metrics["section_types"].get(section_type, 0) + 1
        
        # Calculate averages and rates
        if docs:
            metrics["content_quality"]["average_content_length"] = total_content_length / len(docs)
            metrics["procedural_metadata_completeness"]["complete_metadata_rate"] = \
                (complete_metadata_count / len(docs)) * 100
            
            # Calculate security procedure-specific quality rates
            metrics["security_procedure_quality"]["classification_level_rate"] = (classification_count / len(docs)) * 100
            
            enhanced_count = sum(1 for doc in docs if doc.metadata.get('has_enhanced_procedure', False))
            metrics["security_procedure_quality"]["enhanced_procedure_rate"] = (enhanced_count / len(docs)) * 100
            
            metrics["security_procedure_quality"]["implementation_guidance_rate"] = (implementation_guidance_count / len(docs)) * 100
        
        # Fix infinite values
        if metrics["content_quality"]["min_content_length"] == float('inf'):
            metrics["content_quality"]["min_content_length"] = 0
        
        # Convert sets to counts and lists for JSON serialization
        metrics["unique_procedures_count"] = len(metrics["procedures_processed"])
        metrics["unique_sections_count"] = len(metrics["sections_processed"])
        metrics["procedures_processed"] = sorted(list(metrics["procedures_processed"]))
        metrics["sections_processed"] = sorted(list(metrics["sections_processed"]))
        
        return metrics
    
    def _analyze_stage_performance(self, loader_stats: Dict[str, Any],
                                  flattener_stats: Dict[str, Any],
                                  converter_stats: Dict[str, Any],
                                  embedder_stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the performance of each stage in the processing pipeline.
        
        This provides insights into which stages worked well and which might need optimization 
        or attention. Stage-by-stage analysis is valuable for understanding the overall system 
        performance and identifying bottlenecks or reliability issues.
        
        The performance analysis demonstrates how to aggregate information from multiple sources 
        to create a comprehensive view of system performance that goes beyond what any individual 
        component can provide.
        """
        performance = {
            "loader_performance": self._analyze_loader_performance(loader_stats),
            "flattener_performance": self._analyze_flattener_performance(flattener_stats),
            "converter_performance": self._analyze_converter_performance(converter_stats),
            "embedder_performance": self._analyze_embedder_performance(embedder_stats)
        }
        
        return performance
    
    def _analyze_loader_performance(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze document loader performance metrics specific to internal security procedures."""
        if not stats:
            return {"status": "no_statistics_available"}
        
        return {
            "status": "completed",
            "files_processed": 1,  # Single JSON file processed
            "validation_errors": len(stats.get('validation_errors', [])),
            "sections_validated": stats.get('total_sections', 0),
            "enhancement_rate": stats.get('enhancement_rate_percent', 0),
            "security_specific_elements": {
                "procedures_found": stats.get('procedures_found', 0),
                "implementation_steps": stats.get('implementation_steps_found', 0)
            }
        }
    
    def _analyze_flattener_performance(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze procedural metadata flattener performance metrics for security procedures."""
        if not stats:
            return {"status": "no_statistics_available"}
        
        return {
            "status": "completed",
            "structures_processed": stats.get('total_processed', 0),
            "enhanced_procedures_found": stats.get('enhanced_procedures_found', 0),
            "enhancement_rate": stats.get('enhancement_rate_percent', 0),
            "flattening_errors": stats.get('flattening_errors', 0),
            "complexity_distribution": stats.get('complexity_distribution', {}),
            "implementation_patterns_found": stats.get('implementation_patterns_found', []),
            "tool_usage_patterns": stats.get('tool_usage_patterns', []),
            "workflow_patterns": stats.get('workflow_patterns', [])
        }
    
    def _analyze_converter_performance(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze document converter performance metrics for security procedures."""
        if not stats:
            return {"status": "no_statistics_available"}
        
        success_rate = 0
        if stats.get('total_sections', 0) > 0:
            success_rate = (stats.get('successful_conversions', 0) / stats['total_sections']) * 100
        
        return {
            "status": "completed",
            "total_sections": stats.get('total_sections', 0),
            "successful_conversions": stats.get('successful_conversions', 0),
            "success_rate": success_rate,
            "conversion_errors": stats.get('errors', 0),
            "enhanced_procedure_count": stats.get('enhanced_procedure_count', 0),
            "section_types": stats.get('section_types', {}),
            "complexity_levels": stats.get('complexity_levels', {}),
            "procedural_specific_patterns": stats.get('procedural_specific_patterns', {})
        }
    
    def _analyze_embedder_performance(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze vector store embedder performance metrics for security procedures."""
        if not stats:
            return {"status": "no_statistics_available"}
        
        return {
            "status": "completed",
            "total_documents": stats.get('total_documents', 0),
            "successful_batches": stats.get('successful_batches', 0),
            "failed_batches": stats.get('failed_batches', 0),
            "metadata_compatibility_tested": stats.get('metadata_compatibility_tested', False),
            "metadata_errors": stats.get('metadata_errors', 0),
            "retry_attempts": stats.get('retry_attempts', 0),
            "procedural_specific_metrics": stats.get('procedural_specific_metrics', {})
        }
    
    def _perform_security_procedure_implementation_analysis(self, docs: List[Document]) -> Dict[str, Any]:
        """
        Perform detailed implementation analysis of the processed internal security procedure documents.
        
        This analysis helps understand the patterns and organization of the internal security procedures 
        as represented in the processed documents. The implementation analysis is particularly important 
        for security procedures because it focuses on actionable implementation guidance rather than 
        just structural organization.
        """
        analysis = {
            "procedure_coverage": {},
            "section_coverage": {},
            "implementation_patterns": {},
            "citation_readiness": {},
            "security_procedure_characteristics": {}
        }
        
        # Analyze procedure and section coverage
        procedures_by_section = {}
        implementation_complexity = {}
        tool_requirements = {}
        
        for doc in docs:
            metadata = doc.metadata
            
            procedure_num = metadata.get('procedure_number', '')
            section_num = metadata.get('section_number', '')
            
            if procedure_num:
                # Track procedures by section
                if section_num:
                    if section_num not in procedures_by_section:
                        procedures_by_section[section_num] = set()
                    procedures_by_section[section_num].add(procedure_num)
                
                # Track implementation complexity
                if metadata.get('has_enhanced_procedure', False):
                    complexity = metadata.get('procedure_complexity', 'unknown')
                    implementation_complexity[procedure_num] = complexity
                
                # Track tool requirements
                tool_count = metadata.get('required_tools_count', 0)
                if tool_count > 0:
                    tool_requirements[procedure_num] = tool_count
        
        # Convert sets to lists for JSON serialization
        analysis["procedure_coverage"] = {
            section: sorted(list(procedures)) 
            for section, procedures in procedures_by_section.items()
        }
        
        analysis["section_coverage"] = {
            "total_sections": len(procedures_by_section),
            "sections_with_procedures": list(procedures_by_section.keys())
        }
        
        analysis["implementation_patterns"] = {
            "procedures_by_complexity": self._group_procedures_by_complexity(implementation_complexity),
            "complexity_distribution": self._calculate_complexity_distribution(implementation_complexity),
            "tool_requirement_distribution": self._analyze_tool_requirements(tool_requirements)
        }
        
        # Assess citation readiness for security procedure documents
        analysis["citation_readiness"] = self._assess_procedure_citation_readiness(docs)
        
        # Analyze security procedure-specific characteristics
        analysis["security_procedure_characteristics"] = self._analyze_security_procedure_characteristics(docs)
        
        return analysis
    
    def _group_procedures_by_complexity(self, implementation_complexity: Dict[str, str]) -> Dict[str, List[str]]:
        """Group procedures by their implementation complexity level."""
        complexity_groups = {}
        
        for procedure, complexity in implementation_complexity.items():
            if complexity not in complexity_groups:
                complexity_groups[complexity] = []
            complexity_groups[complexity].append(procedure)
        
        # Sort procedures within each complexity group
        for complexity in complexity_groups:
            complexity_groups[complexity].sort()
        
        return complexity_groups
    
    def _calculate_complexity_distribution(self, implementation_complexity: Dict[str, str]) -> Dict[str, float]:
        """Calculate the distribution of complexity levels across procedures."""
        if not implementation_complexity:
            return {}
        
        complexity_counts = {}
        for complexity in implementation_complexity.values():
            complexity_counts[complexity] = complexity_counts.get(complexity, 0) + 1
        
        total_procedures = len(implementation_complexity)
        complexity_distribution = {
            complexity: (count / total_procedures) * 100
            for complexity, count in complexity_counts.items()
        }
        
        return complexity_distribution
    
    def _analyze_tool_requirements(self, tool_requirements: Dict[str, int]) -> Dict[str, Any]:
        """Analyze tool requirement patterns across procedures."""
        if not tool_requirements:
            return {}
        
        tool_counts = list(tool_requirements.values())
        return {
            "min_tools": min(tool_counts),
            "max_tools": max(tool_counts),
            "average_tools": sum(tool_counts) / len(tool_counts),
            "procedures_with_tools": len(tool_requirements)
        }
    
    def _assess_procedure_citation_readiness(self, docs: List[Document]) -> Dict[str, Any]:
        """
        Assess how ready the processed documents are for precise procedure citation creation.
        
        This analysis evaluates whether the documents have the metadata structure needed for 
        the sophisticated citation system to work effectively with internal security procedure documents.
        """
        readiness = {
            "documents_ready_for_precise_citations": 0,
            "documents_with_basic_citations": 0,
            "documents_needing_improvement": 0,
            "readiness_rate": 0.0,
            "missing_capabilities": []
        }
        
        for doc in docs:
            metadata = doc.metadata
            
            # Check citation readiness criteria for security procedures
            has_procedure = bool(metadata.get('procedure_number'))
            has_enhanced_structure = metadata.get('has_enhanced_procedure', False)
            has_preserved_json = bool(metadata.get('procedure_structure_json'))
            has_complexity_info = bool(metadata.get('procedure_complexity'))
            
            if has_enhanced_structure and has_preserved_json and has_complexity_info:
                readiness["documents_ready_for_precise_citations"] += 1
            elif has_procedure:
                readiness["documents_with_basic_citations"] += 1
            else:
                readiness["documents_needing_improvement"] += 1
        
        # Calculate readiness rate
        if docs:
            readiness["readiness_rate"] = \
                (readiness["documents_ready_for_precise_citations"] / len(docs)) * 100
        
        return readiness
    
    def _analyze_security_procedure_characteristics(self, docs: List[Document]) -> Dict[str, Any]:
        """
        Analyze characteristics specific to internal security procedure documents.
        
        This method identifies patterns and features that are specific to internal security 
        procedure documents, helping validate that the processing pipeline correctly handles 
        security procedure-specific organizational structures and implementation patterns.
        """
        characteristics = {
            "classification_levels": {"count": 0, "distribution": {}},
            "compliance_frameworks": {"count": 0, "rate": 0.0},
            "implementation_guidance": {"count": 0, "rate": 0.0},
            "tool_dependencies": {"count": 0, "rate": 0.0},
            "automation_capabilities": {"count": 0, "rate": 0.0}
        }
        
        for doc in docs:
            metadata = doc.metadata
            content = doc.page_content.lower()
            
            # Check for classification levels (important for security procedures)
            if metadata.get('classification_level'):
                classification = metadata['classification_level']
                characteristics["classification_levels"]["count"] += 1
                characteristics["classification_levels"]["distribution"][classification] = \
                    characteristics["classification_levels"]["distribution"].get(classification, 0) + 1
            
            # Check for compliance frameworks
            if metadata.get('compliance_frameworks'):
                characteristics["compliance_frameworks"]["count"] += 1
            
            # Check for implementation guidance
            implementation_terms = ['configure', 'install', 'implement', 'setup', 'create']
            if any(term in content for term in implementation_terms):
                characteristics["implementation_guidance"]["count"] += 1
            
            # Check for tool dependencies
            if metadata.get('required_tools_count', 0) > 0:
                characteristics["tool_dependencies"]["count"] += 1
            
            # Check for automation capabilities
            automation_terms = ['automate', 'script', 'scheduled', 'automatic']
            if any(term in content for term in automation_terms):
                characteristics["automation_capabilities"]["count"] += 1
        
        # Calculate rates
        if docs:
            total_docs = len(docs)
            for category in characteristics:
                if "count" in characteristics[category]:
                    characteristics[category]["rate"] = \
                        (characteristics[category]["count"] / total_docs) * 100
        
        return characteristics
    
    def _generate_recommendations(self, docs: List[Document],
                                flattener_stats: Dict[str, Any],
                                embedder_stats: Dict[str, Any]) -> List[str]:
        """
        Generate actionable recommendations based on the processing results.
        
        This provides specific, actionable insights for improving the processing pipeline or 
        addressing any issues that were discovered during internal security procedure document 
        processing. The recommendations demonstrate how to transform statistical analysis into 
        practical guidance for system optimization.
        """
        recommendations = []
        
        # Check enhancement rate and provide specific guidance
        enhanced_count = sum(1 for doc in docs if doc.metadata.get('has_enhanced_procedure', False))
        enhancement_rate = (enhanced_count / len(docs)) * 100 if docs else 0
        
        if enhancement_rate < 50:
            recommendations.append(
                f"Enhancement rate is {enhancement_rate:.1f}% for internal security procedures. "
                f"Consider reviewing the input JSON structure to ensure more documents contain "
                f"enhanced procedural metadata specific to security implementation patterns."
            )
        elif enhancement_rate > 90:
            recommendations.append(
                f"Excellent enhancement rate of {enhancement_rate:.1f}% for internal security procedures. "
                f"The procedural metadata flattening approach is working very effectively with security workflows."
            )
        
        # Check for procedural metadata errors
        if embedder_stats and embedder_stats.get('metadata_errors', 0) > 0:
            error_count = embedder_stats['metadata_errors']
            recommendations.append(
                f"Found {error_count} procedural metadata compatibility errors. Review the metadata "
                f"flattening approach to ensure security procedure-specific structures (implementation "
                f"steps, tool requirements, etc.) are properly handled."
            )
        
        # Check flattening effectiveness for security procedure patterns
        if flattener_stats:
            flattening_errors = flattener_stats.get('flattening_errors', 0)
            if flattening_errors > 0:
                recommendations.append(
                    f"Found {flattening_errors} flattening errors in security procedure processing. "
                    f"Consider adding more robust error handling for security procedure-specific "
                    f"edge cases in the procedural metadata structure."
                )
            
            # Check for security-specific patterns
            implementation_patterns = flattener_stats.get('implementation_patterns_found', [])
            if not implementation_patterns:
                recommendations.append(
                    "No security procedure-specific implementation patterns were detected during flattening. "
                    "Verify that the processing pipeline correctly identifies and preserves security "
                    "procedure organizational structures and workflow patterns."
                )
        
        # Check citation readiness for security procedures
        precise_citation_count = sum(1 for doc in docs 
                                   if doc.metadata.get('has_enhanced_procedure', False) and
                                      doc.metadata.get('procedure_structure_json', ''))
        
        if precise_citation_count < enhanced_count:
            recommendations.append(
                "Some enhanced security procedures are missing preserved JSON structure. Ensure "
                "complete structure preservation for maximum citation precision in security contexts."
            )
        
        # Check security procedure-specific quality metrics
        classification_count = sum(1 for doc in docs if doc.metadata.get('classification_level'))
        if classification_count == 0:
            recommendations.append(
                "No classification level information found in security procedures. Consider adding "
                "classification metadata to enhance security procedure organization and access control."
            )
        
        tool_count = sum(1 for doc in docs if doc.metadata.get('required_tools_count', 0) > 0)
        if tool_count == 0:
            recommendations.append(
                "No tool requirement information found in security procedures. Consider adding tool "
                "dependency metadata to enhance implementation guidance and resource planning."
            )
        
        if not recommendations:
            recommendations.append(
                "Processing completed successfully with no significant issues detected. The procedural "
                "metadata flattening approach is working optimally for internal security procedures. "
                "All security procedure-specific patterns and implementation structures appear to be "
                "correctly preserved."
            )
        
        return recommendations
    
    def _collect_sample_procedures(self, docs: List[Document], max_samples: int = 5) -> List[Dict[str, Any]]:
        """
        Collect sample documents that demonstrate the enhanced processing results.
        
        These samples help understand what the processed internal security procedure documents 
        look like and verify that the procedural metadata flattening approach preserved the 
        important implementation information specific to security procedure structure and workflows.
        """
        samples = []
        
        # Prioritize documents with enhanced procedural structure for samples
        enhanced_docs = [doc for doc in docs if doc.metadata.get('has_enhanced_procedure', False)]
        
        # If we don't have enough enhanced docs, include some basic ones
        sample_docs = enhanced_docs[:max_samples]
        if len(sample_docs) < max_samples:
            basic_docs = [doc for doc in docs if not doc.metadata.get('has_enhanced_procedure', False)]
            sample_docs.extend(basic_docs[:max_samples - len(sample_docs)])
        
        for i, doc in enumerate(sample_docs):
            metadata = doc.metadata
            
            sample = {
                "sample_index": i + 1,
                "procedure_number": metadata.get('procedure_number', 'N/A'),
                "procedure_title": metadata.get('procedure_title', 'N/A'),
                "section_info": f"Section {metadata.get('section_number', 'N/A')}: {metadata.get('section_title', 'N/A')}" if metadata.get('section_number') else "No section",
                "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "enhanced_metadata": {
                    "has_enhanced_procedure": metadata.get('has_enhanced_procedure', False),
                    "implementation_step_count": metadata.get('implementation_step_count', 0),
                    "has_sub_steps": metadata.get('has_sub_steps', False),
                    "procedure_complexity": metadata.get('procedure_complexity', 'unknown'),
                    "workflow_type": metadata.get('workflow_type', ''),
                    "json_structure_preserved": bool(metadata.get('procedure_structure_json', ''))
                },
                "security_procedure_specific": {
                    "classification_level": metadata.get('classification_level', 'N/A'),
                    "compliance_frameworks": metadata.get('compliance_frameworks', 'N/A'),
                    "required_tools_count": metadata.get('required_tools_count', 0),
                    "responsible_roles_count": metadata.get('responsible_roles_count', 0)
                }
            }
            
            samples.append(sample)
        
        return samples
    
    def save_summary_to_file(self, summary: Dict[str, Any], output_path: str) -> None:
        """
        Save the comprehensive summary to a JSON file.
        
        This creates a permanent record of the processing results that can be referenced later 
        for analysis, debugging, or optimization planning. The saved summary provides a complete 
        historical record of how well the processing worked for internal security procedure documents.
        """
        self.logger.info(f"Saving comprehensive internal security procedure processing summary to: {output_path}")
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            # Log key summary statistics for immediate visibility
            self._log_summary_highlights(summary)
            
        except Exception as e:
            self.logger.error(f"Error saving internal security procedure processing summary: {e}")
            raise
    
    def _log_summary_highlights(self, summary: Dict[str, Any]) -> None:
        """
        Log key highlights from the summary for immediate visibility.
        
        This provides a quick overview of the most important results without requiring users 
        to read through the complete detailed summary. The highlights focus on the most critical 
        metrics for internal security procedure document processing.
        """
        overview = summary.get("pipeline_overview", {})
        metadata_analysis = summary.get("enhanced_procedural_metadata_analysis", {})
        
        self.logger.info("=== INTERNAL SECURITY PROCEDURE PROCESSING SUMMARY HIGHLIGHTS ===")
        self.logger.info(f"Total documents processed: {overview.get('total_documents_processed', 0)}")
        self.logger.info(f"Enhanced procedures: {metadata_analysis.get('enhanced_procedure_count', 0)}")
        enhancement_rate = 0
        if overview.get('total_documents_processed', 0) > 0:
            enhancement_rate = (metadata_analysis.get('enhanced_procedure_count', 0) / overview['total_documents_processed']) * 100
        self.logger.info(f"Enhancement rate: {enhancement_rate:.1f}%")
        self.logger.info(f"Overall success rate: {overview.get('overall_success_rate', 0):.1f}%")
        self.logger.info(f"Pipeline stages completed: {overview.get('pipeline_stages_completed', 0)}")
        
        # Log security procedure-specific highlights
        security_metrics = overview.get("security_procedure_specific_metrics", {})
        self.logger.info(f"Procedures identified: {security_metrics.get('procedures_identified', 0)}")
        self.logger.info(f"Implementation steps preserved: {security_metrics.get('implementation_steps_preserved', 0)}")
        
        recommendations = summary.get("recommendations", [])
        if recommendations:
            self.logger.info(f"Key recommendation: {recommendations[0]}")


def create_internal_security_summary_generator(logger: logging.Logger) -> InternalSecurityProcessingSummaryGenerator:
    """
    Factory function to create a configured internal security processing summary generator.
    
    This provides a clean interface for creating summary generator instances with proper dependency 
    injection. The factory pattern ensures consistent initialization and makes it easy to modify 
    the creation process if needed in the future.
    """
    return InternalSecurityProcessingSummaryGenerator(logger)