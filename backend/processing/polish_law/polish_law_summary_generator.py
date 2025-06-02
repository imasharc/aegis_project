"""
Polish Law Processing Summary Generator

This module creates comprehensive summaries of the enhanced Polish law document processing
pipeline. Following the same proven pattern as the GDPR summary generator, it aggregates 
statistics from all processing stages and generates detailed reports about the success rate, 
quality metrics, and performance of the metadata flattening approach.

The summary generator demonstrates the "Observer" pattern - it watches what happens in the 
other modules and reports on the overall system health and performance. This approach provides 
valuable insights that go beyond what individual components can offer, creating a holistic 
view of how well the entire processing pipeline is working.

The reporting approach showcases several important principles:
- Comprehensive data aggregation from multiple sources
- Statistical analysis that provides actionable insights
- Clear presentation of complex information
- Recommendations based on observed patterns and performance metrics

This component is particularly valuable for system optimization and troubleshooting because
it identifies patterns and relationships that might not be obvious when looking at individual
components in isolation.
"""

import json
import logging
from typing import Dict, List, Any, Set
from datetime import datetime
from langchain.docstore.document import Document


class PolishLawProcessingSummaryGenerator:
    """
    Generates comprehensive summaries of the Polish law document processing pipeline.
    
    This class aggregates statistics and metrics from all stages of the processing pipeline 
    to create detailed reports about the effectiveness of the metadata flattening approach 
    and overall system performance for Polish law documents. The analysis goes beyond simple 
    success/failure metrics to provide insights about document quality, processing patterns, 
    and optimization opportunities.
    
    The summary generator demonstrates how to create valuable system insights by combining 
    data from multiple sources. Rather than just reporting what happened, it analyzes why 
    certain patterns occurred and what they mean for system performance and reliability.
    
    The comprehensive analysis includes:
    - Processing pipeline performance across all stages
    - Document quality metrics and patterns
    - Metadata flattening effectiveness analysis
    - Polish law-specific structural analysis
    - Actionable recommendations for system optimization
    """
    
    def __init__(self, logger: logging.Logger):
        """
        Initialize the Polish law processing summary generator.
        
        The initialization sets up the logging infrastructure that will be used throughout
        the analysis process. The logger is particularly important for this component because
        it generates complex reports that benefit from detailed logging of the analysis process.
        
        Args:
            logger: Configured logger for tracking summary generation
        """
        self.logger = logger
        self.logger.info("Polish Law Processing Summary Generator initialized")
    
    def generate_comprehensive_summary(self, docs: List[Document], 
                                     processing_timestamp: str,
                                     loader_stats: Dict[str, Any] = None,
                                     flattener_stats: Dict[str, Any] = None,
                                     converter_stats: Dict[str, Any] = None,
                                     embedder_stats: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate a comprehensive summary of the entire Polish law processing pipeline.
        
        This method creates a detailed report that aggregates information from all processing 
        stages to provide a complete picture of how well the enhanced processing worked for 
        Polish law documents. The analysis demonstrates how to synthesize complex information 
        from multiple sources into actionable insights.
        
        The comprehensive approach is valuable because it reveals relationships and patterns 
        that wouldn't be obvious when looking at individual components in isolation. For 
        example, the relationship between document complexity and processing success rates, 
        or the correlation between metadata quality and citation precision.
        
        Args:
            docs: Final processed documents for analysis
            processing_timestamp: Timestamp of the processing session
            loader_stats: Statistics from the document loader (optional)
            flattener_stats: Statistics from the metadata flattener (optional)
            converter_stats: Statistics from the document converter (optional)
            embedder_stats: Statistics from the embedder (optional)
            
        Returns:
            Comprehensive summary dictionary with detailed statistics and analysis
        """
        self.logger.info("Generating comprehensive Polish law processing summary...")
        
        # The summary structure is designed to provide multiple perspectives on the same data,
        # making it easy to understand both high-level performance and detailed patterns
        summary = {
            "processing_timestamp": processing_timestamp,
            "generation_timestamp": datetime.now().isoformat(),
            "document_type": "polish_data_protection_law",
            "pipeline_overview": self._create_pipeline_overview(docs, loader_stats, converter_stats, embedder_stats),
            "enhanced_metadata_analysis": self._analyze_enhanced_metadata(docs),
            "document_quality_metrics": self._calculate_quality_metrics(docs),
            "processing_stage_performance": self._analyze_stage_performance(loader_stats, flattener_stats, converter_stats, embedder_stats),
            "polish_law_structural_analysis": self._perform_polish_law_structural_analysis(docs),
            "recommendations": self._generate_recommendations(docs, flattener_stats, embedder_stats),
            "sample_enhanced_documents": self._collect_sample_documents(docs)
        }
        
        self.logger.info("Comprehensive Polish law processing summary generated successfully")
        return summary
    
    def _create_pipeline_overview(self, docs: List[Document], 
                                 loader_stats: Dict[str, Any],
                                 converter_stats: Dict[str, Any],
                                 embedder_stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a high-level overview of the entire processing pipeline.
        
        This provides key metrics that give a quick assessment of how well the processing 
        worked overall. The overview is designed to answer the most important questions 
        first: did the processing work, how well did it work, and are there any critical 
        issues that need immediate attention.
        
        The overview approach demonstrates how to distill complex information into 
        essential insights that can be quickly understood by both technical and 
        non-technical stakeholders.
        """
        overview = {
            "total_documents_processed": len(docs),
            "pipeline_stages_completed": 0,
            "overall_success_rate": 0.0,
            "enhanced_metadata_rate": 0.0,
            "critical_errors": [],
            "polish_law_specific_metrics": {
                "sections_identified": 0,
                "gazette_references_preserved": 0,
                "articles_with_enhanced_structure": 0
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
            total_chunks = converter_stats.get('total_chunks', 0)
            successful_conversions = converter_stats.get('successful_conversions', 0)
            if total_chunks > 0:
                overview["overall_success_rate"] = (successful_conversions / total_chunks) * 100
        
        # Calculate enhanced metadata rate to assess flattening effectiveness
        # This metric shows how well the sophisticated metadata processing worked
        enhanced_count = sum(1 for doc in docs if doc.metadata.get('has_enhanced_structure', False))
        if docs:
            overview["enhanced_metadata_rate"] = (enhanced_count / len(docs)) * 100
        
        # Analyze Polish law-specific metrics
        for doc in docs:
            metadata = doc.metadata
            if metadata.get('section_number'):
                overview["polish_law_specific_metrics"]["sections_identified"] += 1
            if metadata.get('gazette_reference'):
                overview["polish_law_specific_metrics"]["gazette_references_preserved"] += 1
            if metadata.get('has_enhanced_structure', False):
                overview["polish_law_specific_metrics"]["articles_with_enhanced_structure"] += 1
        
        # Check for critical errors that need immediate attention
        # This helps prioritize issues that need immediate resolution
        if embedder_stats and embedder_stats.get('metadata_errors', 0) > 0:
            overview["critical_errors"].append("Metadata compatibility issues detected")
        
        if converter_stats and converter_stats.get('errors', 0) > converter_stats.get('successful_conversions', 0) * 0.1:
            overview["critical_errors"].append("High document conversion error rate")
        
        return overview
    
    def _analyze_enhanced_metadata(self, docs: List[Document]) -> Dict[str, Any]:
        """
        Analyze the enhanced metadata across all processed Polish law documents.
        
        This provides detailed insights into how well the metadata flattening approach 
        worked and what types of enhanced structures were preserved for Polish law documents. 
        The analysis helps understand the effectiveness of the flattening approach and 
        identifies opportunities for optimization.
        
        The metadata analysis demonstrates how to extract meaningful insights from complex 
        data structures by looking at patterns, distributions, and relationships rather 
        than just counting occurrences.
        """
        analysis = {
            "total_documents": len(docs),
            "enhanced_structure_count": 0,
            "simple_documents": 0,
            "complexity_distribution": {},
            "numbering_styles": {},
            "paragraph_statistics": {
                "min_paragraphs": float('inf'),
                "max_paragraphs": 0,
                "total_paragraphs": 0,
                "articles_with_sub_paragraphs": 0
            },
            "flattening_effectiveness": {
                "json_preservation_rate": 0.0,
                "structural_indicators_rate": 0.0
            },
            "polish_law_patterns": {
                "sections_usage": {},
                "gazette_references": 0,
                "cross_references": 0
            }
        }
        
        # Track various metadata characteristics to understand document patterns
        json_preserved_count = 0
        structural_indicators_count = 0
        
        for doc in docs:
            metadata = doc.metadata
            
            # Analyze enhanced structure presence and characteristics
            if metadata.get('has_enhanced_structure', False):
                analysis["enhanced_structure_count"] += 1
                
                # Analyze complexity distribution to understand document patterns
                complexity = metadata.get('complexity_level', 'unknown')
                analysis["complexity_distribution"][complexity] = \
                    analysis["complexity_distribution"].get(complexity, 0) + 1
                
                # Analyze numbering styles used in Polish law documents
                numbering_style = metadata.get('numbering_style', '')
                if numbering_style:
                    analysis["numbering_styles"][numbering_style] = \
                        analysis["numbering_styles"].get(numbering_style, 0) + 1
                
                # Analyze paragraph statistics for structural understanding
                self._update_paragraph_statistics(metadata, analysis["paragraph_statistics"])
                
                # Check flattening effectiveness metrics
                if metadata.get('article_structure_json', ''):
                    json_preserved_count += 1
                
                if any(metadata.get(field, 0) > 0 for field in ['paragraph_count', 'has_sub_paragraphs']):
                    structural_indicators_count += 1
                    
            else:
                analysis["simple_documents"] += 1
            
            # Analyze Polish law-specific patterns
            self._analyze_polish_law_patterns(metadata, analysis["polish_law_patterns"])
        
        # Calculate flattening effectiveness rates
        if analysis["enhanced_structure_count"] > 0:
            analysis["flattening_effectiveness"]["json_preservation_rate"] = \
                (json_preserved_count / analysis["enhanced_structure_count"]) * 100
            analysis["flattening_effectiveness"]["structural_indicators_rate"] = \
                (structural_indicators_count / analysis["enhanced_structure_count"]) * 100
        
        # Fix infinite values in paragraph statistics
        if analysis["paragraph_statistics"]["min_paragraphs"] == float('inf'):
            analysis["paragraph_statistics"]["min_paragraphs"] = 0
        
        return analysis
    
    def _update_paragraph_statistics(self, metadata: Dict[str, Any], stats: Dict[str, Any]) -> None:
        """
        Update paragraph statistics with information from a single document.
        
        This tracks detailed information about the structure of processed articles to 
        understand the complexity and patterns in the Polish law regulation. The statistics 
        help identify optimization opportunities and validate that the processing pipeline 
        is working correctly for documents of different complexity levels.
        """
        para_count = metadata.get('paragraph_count', 0)
        if para_count > 0:
            stats["min_paragraphs"] = min(stats["min_paragraphs"], para_count)
            stats["max_paragraphs"] = max(stats["max_paragraphs"], para_count)
            stats["total_paragraphs"] += para_count
            
            if metadata.get('has_sub_paragraphs', False):
                stats["articles_with_sub_paragraphs"] += 1
    
    def _analyze_polish_law_patterns(self, metadata: Dict[str, Any], patterns: Dict[str, Any]) -> None:
        """
        Analyze Polish law-specific patterns in the processed documents.
        
        This method identifies patterns that are specific to Polish legal documents, such as 
        section usage, gazette references, and cross-references. Understanding these patterns 
        helps validate that the processing pipeline correctly handles Polish law-specific 
        organizational structures.
        """
        # Track section usage patterns
        if metadata.get('section_number'):
            section_num = metadata['section_number']
            patterns["sections_usage"][section_num] = patterns["sections_usage"].get(section_num, 0) + 1
        
        # Track gazette references (important for Polish law authenticity)
        if metadata.get('gazette_reference'):
            patterns["gazette_references"] += 1
        
        # Track cross-references (common in Polish legal documents)
        if metadata.get('has_cross_references', False):
            patterns["cross_references"] += 1
    
    def _calculate_quality_metrics(self, docs: List[Document]) -> Dict[str, Any]:
        """
        Calculate various quality metrics for the processed Polish law documents.
        
        These metrics help assess the overall quality of the processing pipeline and identify 
        any issues that might affect the citation system's performance. Quality metrics are 
        particularly important for legal document processing because accuracy and completeness 
        are critical for reliable legal citations.
        
        The quality analysis demonstrates how to create meaningful metrics that go beyond 
        simple success/failure counts to provide insights about the characteristics and 
        reliability of the processed data.
        """
        metrics = {
            "content_quality": {
                "average_content_length": 0,
                "min_content_length": float('inf'),
                "max_content_length": 0,
                "empty_content_count": 0
            },
            "metadata_completeness": {
                "articles_with_numbers": 0,
                "articles_with_titles": 0,
                "chapters_identified": 0,
                "sections_identified": 0,
                "complete_metadata_rate": 0.0
            },
            "document_types": {},
            "articles_processed": set(),
            "chapters_processed": set(),
            "sections_processed": set(),
            "polish_law_quality": {
                "gazette_reference_rate": 0.0,
                "enhanced_structure_rate": 0.0,
                "section_organization_rate": 0.0
            }
        }
        
        total_content_length = 0
        complete_metadata_count = 0
        gazette_ref_count = 0
        
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
            
            # Analyze metadata completeness for Polish law documents
            metadata = doc.metadata
            
            if metadata.get('article_number'):
                metrics["metadata_completeness"]["articles_with_numbers"] += 1
                metrics["articles_processed"].add(metadata['article_number'])
            
            if metadata.get('article_title'):
                metrics["metadata_completeness"]["articles_with_titles"] += 1
            
            if metadata.get('chapter_number'):
                metrics["metadata_completeness"]["chapters_identified"] += 1
                metrics["chapters_processed"].add(metadata['chapter_number'])
            
            if metadata.get('section_number'):
                metrics["metadata_completeness"]["sections_identified"] += 1
                metrics["sections_processed"].add(metadata['section_number'])
            
            # Check for complete metadata (article number, title, content, and Polish law-specific fields)
            if (metadata.get('article_number') and 
                metadata.get('article_title') and 
                content_length > 0):
                complete_metadata_count += 1
            
            # Track Polish law-specific quality metrics
            if metadata.get('gazette_reference'):
                gazette_ref_count += 1
            
            # Track document types
            doc_type = metadata.get('type', 'unknown')
            metrics["document_types"][doc_type] = metrics["document_types"].get(doc_type, 0) + 1
        
        # Calculate averages and rates
        if docs:
            metrics["content_quality"]["average_content_length"] = total_content_length / len(docs)
            metrics["metadata_completeness"]["complete_metadata_rate"] = \
                (complete_metadata_count / len(docs)) * 100
            
            # Calculate Polish law-specific quality rates
            metrics["polish_law_quality"]["gazette_reference_rate"] = (gazette_ref_count / len(docs)) * 100
            
            enhanced_count = sum(1 for doc in docs if doc.metadata.get('has_enhanced_structure', False))
            metrics["polish_law_quality"]["enhanced_structure_rate"] = (enhanced_count / len(docs)) * 100
            
            section_count = len(metrics["sections_processed"])
            if section_count > 0:
                metrics["polish_law_quality"]["section_organization_rate"] = \
                    (metrics["metadata_completeness"]["sections_identified"] / len(docs)) * 100
        
        # Fix infinite values
        if metrics["content_quality"]["min_content_length"] == float('inf'):
            metrics["content_quality"]["min_content_length"] = 0
        
        # Convert sets to counts and lists for JSON serialization
        metrics["unique_articles_count"] = len(metrics["articles_processed"])
        metrics["unique_chapters_count"] = len(metrics["chapters_processed"])
        metrics["unique_sections_count"] = len(metrics["sections_processed"])
        metrics["articles_processed"] = sorted(list(metrics["articles_processed"]))
        metrics["chapters_processed"] = sorted(list(metrics["chapters_processed"]))
        metrics["sections_processed"] = sorted(list(metrics["sections_processed"]))
        
        return metrics
    
    def _analyze_stage_performance(self, loader_stats: Dict[str, Any],
                                  flattener_stats: Dict[str, Any],
                                  converter_stats: Dict[str, Any],
                                  embedder_stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the performance of each stage in the processing pipeline.
        
        This provides insights into which stages worked well and which might need 
        optimization or attention. Stage-by-stage analysis is valuable for understanding 
        the overall system performance and identifying bottlenecks or reliability issues.
        
        The performance analysis demonstrates how to aggregate information from multiple 
        sources to create a comprehensive view of system performance that goes beyond 
        what any individual component can provide.
        """
        performance = {
            "loader_performance": self._analyze_loader_performance(loader_stats),
            "flattener_performance": self._analyze_flattener_performance(flattener_stats),
            "converter_performance": self._analyze_converter_performance(converter_stats),
            "embedder_performance": self._analyze_embedder_performance(embedder_stats)
        }
        
        return performance
    
    def _analyze_loader_performance(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze document loader performance metrics specific to Polish law documents."""
        if not stats:
            return {"status": "no_statistics_available"}
        
        return {
            "status": "completed",
            "files_processed": 1,  # Single JSON file processed
            "validation_errors": len(stats.get('validation_errors', [])),
            "chunks_validated": stats.get('total_chunks', 0),
            "enhancement_rate": stats.get('enhancement_rate_percent', 0),
            "polish_specific_elements": {
                "sections_found": stats.get('sections_found', 0),
                "gazette_references": stats.get('gazette_references', 0)
            }
        }
    
    def _analyze_flattener_performance(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze metadata flattener performance metrics for Polish law documents."""
        if not stats:
            return {"status": "no_statistics_available"}
        
        return {
            "status": "completed",
            "structures_processed": stats.get('total_processed', 0),
            "enhanced_structures_found": stats.get('enhanced_structures_found', 0),
            "enhancement_rate": stats.get('enhancement_rate_percent', 0),
            "flattening_errors": stats.get('flattening_errors', 0),
            "complexity_distribution": stats.get('complexity_distribution', {}),
            "numbering_styles_found": stats.get('numbering_styles_found', []),
            "polish_specific_patterns": stats.get('polish_specific_patterns', [])
        }
    
    def _analyze_converter_performance(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze document converter performance metrics for Polish law documents."""
        if not stats:
            return {"status": "no_statistics_available"}
        
        success_rate = 0
        if stats.get('total_chunks', 0) > 0:
            success_rate = (stats.get('successful_conversions', 0) / stats['total_chunks']) * 100
        
        return {
            "status": "completed",
            "total_chunks": stats.get('total_chunks', 0),
            "successful_conversions": stats.get('successful_conversions', 0),
            "success_rate": success_rate,
            "conversion_errors": stats.get('errors', 0),
            "enhanced_structure_count": stats.get('enhanced_structure_count', 0),
            "chunk_types": stats.get('chunk_types', {}),
            "complexity_levels": stats.get('complexity_levels', {}),
            "polish_specific_patterns": stats.get('polish_specific_patterns', {})
        }
    
    def _analyze_embedder_performance(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze vector store embedder performance metrics for Polish law documents."""
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
            "polish_specific_metrics": stats.get('polish_specific_metrics', {})
        }
    
    def _perform_polish_law_structural_analysis(self, docs: List[Document]) -> Dict[str, Any]:
        """
        Perform detailed structural analysis of the processed Polish law documents.
        
        This analysis helps understand the patterns and organization of the Polish data 
        protection law as represented in the processed documents. The structural analysis 
        is particularly important for Polish law because it has specific organizational 
        patterns that differ from EU regulations.
        """
        analysis = {
            "article_coverage": {},
            "chapter_coverage": {},
            "section_coverage": {},  # Important for Polish law
            "structural_patterns": {},
            "citation_readiness": {},
            "polish_law_characteristics": {}
        }
        
        # Analyze article, chapter, and section coverage
        articles_by_chapter = {}
        sections_by_chapter = {}
        article_complexity = {}
        
        for doc in docs:
            metadata = doc.metadata
            
            article_num = metadata.get('article_number', '')
            chapter_num = metadata.get('chapter_number', '')
            section_num = metadata.get('section_number', '')
            
            if article_num:
                # Track articles by chapter
                if chapter_num:
                    if chapter_num not in articles_by_chapter:
                        articles_by_chapter[chapter_num] = set()
                    articles_by_chapter[chapter_num].add(article_num)
                
                # Track article complexity
                if metadata.get('has_enhanced_structure', False):
                    complexity = metadata.get('complexity_level', 'unknown')
                    article_complexity[article_num] = complexity
            
            # Track sections (important for Polish law organization)
            if section_num and chapter_num:
                if chapter_num not in sections_by_chapter:
                    sections_by_chapter[chapter_num] = set()
                sections_by_chapter[chapter_num].add(section_num)
        
        # Convert sets to lists for JSON serialization
        analysis["article_coverage"] = {
            chapter: sorted(list(articles)) 
            for chapter, articles in articles_by_chapter.items()
        }
        
        analysis["section_coverage"] = {
            chapter: sorted(list(sections))
            for chapter, sections in sections_by_chapter.items()
        }
        
        analysis["chapter_coverage"] = {
            "total_chapters": len(articles_by_chapter),
            "chapters_with_articles": list(articles_by_chapter.keys()),
            "chapters_with_sections": list(sections_by_chapter.keys())
        }
        
        analysis["structural_patterns"] = {
            "articles_by_complexity": self._group_articles_by_complexity(article_complexity),
            "complexity_distribution": self._calculate_complexity_distribution(article_complexity)
        }
        
        # Assess citation readiness for Polish law documents
        analysis["citation_readiness"] = self._assess_citation_readiness(docs)
        
        # Analyze Polish law-specific characteristics
        analysis["polish_law_characteristics"] = self._analyze_polish_law_characteristics(docs)
        
        return analysis
    
    def _group_articles_by_complexity(self, article_complexity: Dict[str, str]) -> Dict[str, List[str]]:
        """Group articles by their structural complexity level."""
        complexity_groups = {}
        
        for article, complexity in article_complexity.items():
            if complexity not in complexity_groups:
                complexity_groups[complexity] = []
            complexity_groups[complexity].append(article)
        
        # Sort articles within each complexity group
        for complexity in complexity_groups:
            complexity_groups[complexity].sort()
        
        return complexity_groups
    
    def _calculate_complexity_distribution(self, article_complexity: Dict[str, str]) -> Dict[str, float]:
        """Calculate the distribution of complexity levels across articles."""
        if not article_complexity:
            return {}
        
        complexity_counts = {}
        for complexity in article_complexity.values():
            complexity_counts[complexity] = complexity_counts.get(complexity, 0) + 1
        
        total_articles = len(article_complexity)
        complexity_distribution = {
            complexity: (count / total_articles) * 100
            for complexity, count in complexity_counts.items()
        }
        
        return complexity_distribution
    
    def _assess_citation_readiness(self, docs: List[Document]) -> Dict[str, Any]:
        """
        Assess how ready the processed documents are for precise citation creation.
        
        This analysis evaluates whether the documents have the metadata structure needed 
        for the sophisticated citation system to work effectively with Polish law documents.
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
            
            # Check citation readiness criteria for Polish law
            has_article = bool(metadata.get('article_number'))
            has_enhanced_structure = metadata.get('has_enhanced_structure', False)
            has_preserved_json = bool(metadata.get('article_structure_json'))
            has_complexity_info = bool(metadata.get('complexity_level'))
            
            if has_enhanced_structure and has_preserved_json and has_complexity_info:
                readiness["documents_ready_for_precise_citations"] += 1
            elif has_article:
                readiness["documents_with_basic_citations"] += 1
            else:
                readiness["documents_needing_improvement"] += 1
        
        # Calculate readiness rate
        if docs:
            readiness["readiness_rate"] = \
                (readiness["documents_ready_for_precise_citations"] / len(docs)) * 100
        
        return readiness
    
    def _analyze_polish_law_characteristics(self, docs: List[Document]) -> Dict[str, Any]:
        """
        Analyze characteristics specific to Polish law documents.
        
        This method identifies patterns and features that are specific to Polish legal 
        documents, helping validate that the processing pipeline correctly handles 
        Polish law-specific organizational structures and content patterns.
        """
        characteristics = {
            "gazette_references": {"count": 0, "rate": 0.0},
            "section_organization": {"sections_found": 0, "rate": 0.0},
            "cross_references": {"count": 0, "rate": 0.0},
            "amendment_info": {"count": 0, "rate": 0.0},
            "polish_terminology": {"documents_with_polish_terms": 0, "rate": 0.0}
        }
        
        for doc in docs:
            metadata = doc.metadata
            content = doc.page_content.lower()
            
            # Check for gazette references (important for Polish law authenticity)
            if metadata.get('gazette_reference'):
                characteristics["gazette_references"]["count"] += 1
            
            # Check for section organization
            if metadata.get('section_number'):
                characteristics["section_organization"]["sections_found"] += 1
            
            # Check for amendment information
            if metadata.get('amendment_info'):
                characteristics["amendment_info"]["count"] += 1
            
            # Check for Polish legal terminology
            polish_terms = ['ustawa', 'artykuł', 'rozdział', 'przepis', 'dziennik ustaw']
            if any(term in content for term in polish_terms):
                characteristics["polish_terminology"]["documents_with_polish_terms"] += 1
        
        # Calculate rates
        if docs:
            total_docs = len(docs)
            for category in characteristics:
                if "count" in characteristics[category]:
                    characteristics[category]["rate"] = \
                        (characteristics[category]["count"] / total_docs) * 100
                elif "sections_found" in characteristics[category]:
                    characteristics[category]["rate"] = \
                        (characteristics[category]["sections_found"] / total_docs) * 100
                elif "documents_with_polish_terms" in characteristics[category]:
                    characteristics[category]["rate"] = \
                        (characteristics[category]["documents_with_polish_terms"] / total_docs) * 100
        
        return characteristics
    
    def _generate_recommendations(self, docs: List[Document],
                                flattener_stats: Dict[str, Any],
                                embedder_stats: Dict[str, Any]) -> List[str]:
        """
        Generate actionable recommendations based on the processing results.
        
        This provides specific, actionable insights for improving the processing pipeline 
        or addressing any issues that were discovered during Polish law document processing. 
        The recommendations demonstrate how to transform statistical analysis into practical 
        guidance for system optimization.
        """
        recommendations = []
        
        # Check enhancement rate and provide specific guidance
        enhanced_count = sum(1 for doc in docs if doc.metadata.get('has_enhanced_structure', False))
        enhancement_rate = (enhanced_count / len(docs)) * 100 if docs else 0
        
        if enhancement_rate < 50:
            recommendations.append(
                f"Enhancement rate is {enhancement_rate:.1f}% for Polish law documents. "
                f"Consider reviewing the input JSON structure to ensure more documents "
                f"contain enhanced metadata specific to Polish legal document patterns."
            )
        elif enhancement_rate > 90:
            recommendations.append(
                f"Excellent enhancement rate of {enhancement_rate:.1f}% for Polish law documents. "
                f"The metadata flattening approach is working very effectively with Polish legal structures."
            )
        
        # Check for metadata errors specific to Polish law
        if embedder_stats and embedder_stats.get('metadata_errors', 0) > 0:
            error_count = embedder_stats['metadata_errors']
            recommendations.append(
                f"Found {error_count} metadata compatibility errors in Polish law processing. "
                f"Review the metadata flattening approach to ensure Polish law-specific "
                f"structures (sections, gazette references, etc.) are properly handled."
            )
        
        # Check flattening effectiveness for Polish law patterns
        if flattener_stats:
            flattening_errors = flattener_stats.get('flattening_errors', 0)
            if flattening_errors > 0:
                recommendations.append(
                    f"Found {flattening_errors} flattening errors in Polish law processing. "
                    f"Consider adding more robust error handling for Polish law-specific "
                    f"edge cases in the metadata structure."
                )
            
            # Check for Polish-specific patterns
            polish_patterns = flattener_stats.get('polish_specific_patterns', [])
            if not polish_patterns:
                recommendations.append(
                    "No Polish law-specific patterns were detected during flattening. "
                    "Verify that the processing pipeline correctly identifies and preserves "
                    "Polish legal document organizational structures."
                )
        
        # Check citation readiness for Polish law documents
        precise_citation_count = sum(1 for doc in docs 
                                   if doc.metadata.get('has_enhanced_structure', False) and
                                      doc.metadata.get('article_structure_json', ''))
        
        if precise_citation_count < enhanced_count:
            recommendations.append(
                "Some enhanced Polish law documents are missing preserved JSON structure. "
                "Ensure complete structure preservation for maximum citation precision "
                "in Polish legal contexts."
            )
        
        # Check Polish law-specific quality metrics
        section_count = sum(1 for doc in docs if doc.metadata.get('section_number'))
        if section_count == 0:
            recommendations.append(
                "No section information found in Polish law documents. "
                "Verify that the processing pipeline correctly identifies Polish law "
                "section organization, which is important for accurate citations."
            )
        
        gazette_ref_count = sum(1 for doc in docs if doc.metadata.get('gazette_reference'))
        if gazette_ref_count == 0:
            recommendations.append(
                "No gazette references found in Polish law documents. "
                "Consider adding gazette reference information to enhance the "
                "authenticity and traceability of Polish law citations."
            )
        
        if not recommendations:
            recommendations.append(
                "Processing completed successfully with no significant issues detected. "
                "The metadata flattening approach is working optimally for Polish law documents. "
                "All Polish law-specific patterns and structures appear to be correctly preserved."
            )
        
        return recommendations
    
    def _collect_sample_documents(self, docs: List[Document], max_samples: int = 5) -> List[Dict[str, Any]]:
        """
        Collect sample documents that demonstrate the enhanced processing results.
        
        These samples help understand what the processed Polish law documents look like 
        and verify that the metadata flattening approach preserved the important information 
        specific to Polish legal document structure and organization.
        """
        samples = []
        
        # Prioritize documents with enhanced structure for samples
        enhanced_docs = [doc for doc in docs if doc.metadata.get('has_enhanced_structure', False)]
        
        # If we don't have enough enhanced docs, include some basic ones
        sample_docs = enhanced_docs[:max_samples]
        if len(sample_docs) < max_samples:
            basic_docs = [doc for doc in docs if not doc.metadata.get('has_enhanced_structure', False)]
            sample_docs.extend(basic_docs[:max_samples - len(sample_docs)])
        
        for i, doc in enumerate(sample_docs):
            metadata = doc.metadata
            
            sample = {
                "sample_index": i + 1,
                "article_number": metadata.get('article_number', 'N/A'),
                "article_title": metadata.get('article_title', 'N/A'),
                "chapter_info": f"Chapter {metadata.get('chapter_number', 'N/A')}: {metadata.get('chapter_title', 'N/A')}",
                "section_info": f"Section {metadata.get('section_number', 'N/A')}: {metadata.get('section_title', 'N/A')}" if metadata.get('section_number') else "No section",
                "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "enhanced_metadata": {
                    "has_enhanced_structure": metadata.get('has_enhanced_structure', False),
                    "paragraph_count": metadata.get('paragraph_count', 0),
                    "has_sub_paragraphs": metadata.get('has_sub_paragraphs', False),
                    "complexity_level": metadata.get('complexity_level', 'unknown'),
                    "numbering_style": metadata.get('numbering_style', ''),
                    "json_structure_preserved": bool(metadata.get('article_structure_json', ''))
                },
                "polish_law_specific": {
                    "gazette_reference": metadata.get('gazette_reference', 'N/A'),
                    "amendment_info": metadata.get('amendment_info', 'N/A'),
                    "parliament_session": metadata.get('parliament_session', 'N/A')
                }
            }
            
            samples.append(sample)
        
        return samples
    
    def save_summary_to_file(self, summary: Dict[str, Any], output_path: str) -> None:
        """
        Save the comprehensive summary to a JSON file.
        
        This creates a permanent record of the processing results that can be referenced 
        later for analysis, debugging, or optimization planning. The saved summary provides 
        a complete historical record of how well the processing worked for Polish law documents.
        """
        self.logger.info(f"Saving comprehensive Polish law processing summary to: {output_path}")
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            # Log key summary statistics for immediate visibility
            self._log_summary_highlights(summary)
            
        except Exception as e:
            self.logger.error(f"Error saving Polish law processing summary: {e}")
            raise
    
    def _log_summary_highlights(self, summary: Dict[str, Any]) -> None:
        """
        Log key highlights from the summary for immediate visibility.
        
        This provides a quick overview of the most important results without requiring 
        users to read through the complete detailed summary. The highlights focus on 
        the most critical metrics for Polish law document processing.
        """
        overview = summary.get("pipeline_overview", {})
        metadata_analysis = summary.get("enhanced_metadata_analysis", {})
        
        self.logger.info("=== POLISH LAW PROCESSING SUMMARY HIGHLIGHTS ===")
        self.logger.info(f"Total documents processed: {overview.get('total_documents_processed', 0)}")
        self.logger.info(f"Enhanced documents: {metadata_analysis.get('enhanced_structure_count', 0)}")
        self.logger.info(f"Enhancement rate: {metadata_analysis.get('enhanced_structure_count', 0) / overview.get('total_documents_processed', 1) * 100:.1f}%")
        self.logger.info(f"Overall success rate: {overview.get('overall_success_rate', 0):.1f}%")
        self.logger.info(f"Pipeline stages completed: {overview.get('pipeline_stages_completed', 0)}")
        
        # Log Polish law-specific highlights
        polish_metrics = overview.get("polish_law_specific_metrics", {})
        self.logger.info(f"Sections identified: {polish_metrics.get('sections_identified', 0)}")
        self.logger.info(f"Gazette references preserved: {polish_metrics.get('gazette_references_preserved', 0)}")
        
        recommendations = summary.get("recommendations", [])
        if recommendations:
            self.logger.info(f"Key recommendation: {recommendations[0]}")


def create_polish_law_summary_generator(logger: logging.Logger) -> PolishLawProcessingSummaryGenerator:
    """
    Factory function to create a configured Polish law processing summary generator.
    
    This provides a clean interface for creating summary generator instances with proper 
    dependency injection. The factory pattern ensures consistent initialization and makes 
    it easy to modify the creation process if needed in the future.
    """
    return PolishLawProcessingSummaryGenerator(logger)