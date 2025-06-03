"""
Summarization Response Builder

This module handles the sophisticated challenge of creating comprehensive action plans
by integrating LLM capabilities with the precise citation system built by all the
other components. It represents the "AI integration layer" that takes the technical
sophistication and transforms it into actionable user guidance.

Think of this as the "translator" between your sophisticated technical system and
user value. The citation manager creates precise references, the precision analyzer
validates quality, the formatter prepares presentation - but this component uses
AI to weave everything together into coherent, actionable compliance guidance.

The response builder demonstrates how architectural sophistication can enhance AI
rather than replace it. Instead of hoping the AI produces perfect output, we use
our sophisticated components to guide, validate, and enhance AI-generated content.
"""

import logging
from typing import Dict, List, Any, Optional
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate


class SummarizationResponseBuilder:
    """
    Builds comprehensive action plan responses using LLM integration with sophisticated citation systems.
    
    This class solves a critical challenge in AI-assisted systems: how do you combine
    the creativity and language capabilities of LLMs with the precision and reliability
    of sophisticated technical processing? The response builder demonstrates how
    architectural sophistication can guide AI to produce better, more reliable outputs.
    
    The builder doesn't just send citations to an LLM and hope for the best. Instead,
    it uses structured prompts that leverage the precision analysis, citation quality,
    and cross-domain insights to guide the LLM toward creating truly valuable outputs
    that justify the sophisticated processing pipeline behind them.
    """
    
    def __init__(self, logger: logging.Logger):
        """
        Initialize the summarization response builder.
        
        Args:
            logger: Configured logger for tracking response building operations
        """
        self.logger = logger
        self.logger.info("Summarization Response Builder initialized")
        
        # Initialize LLM with appropriate configuration
        self.model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        # Track response building statistics
        self.building_stats = {
            'total_responses_built': 0,
            'sophisticated_prompts_used': 0,
            'fallback_prompts_used': 0,
            'llm_integration_successes': 0,
            'llm_integration_failures': 0,
            'average_response_quality_score': 0.0,
            'citation_integration_success_rate': 0.0,
            'building_errors': 0
        }
        
        # Configure sophisticated prompt templates
        self._setup_prompt_templates()
    
    def _setup_prompt_templates(self) -> None:
        """
        Set up sophisticated prompt templates that leverage the precision of the citation system.
        
        These prompts are designed to work with the sophisticated citations and analysis
        results to guide the LLM toward creating responses that showcase the system's
        capabilities while providing maximum user value.
        """
        # Main sophisticated prompt for comprehensive responses
        self.sophisticated_prompt = ChatPromptTemplate.from_template(
            """You are a comprehensive compliance expert creating actionable guidance based on sophisticated multi-domain analysis.

            You have access to precisely cited information from three authoritative domains:
            1. GDPR (European regulation) - with paragraph-level legal precision
            2. Polish Data Protection Act (local implementation) - with enhanced structural references
            3. Internal Security Procedures (organizational implementation) - with implementation step details

            SYSTEM SOPHISTICATION CONTEXT:
            The citations you're working with represent sophisticated analysis:
            - Enhanced structural analysis with metadata flattening approaches
            - Cross-domain integration preserving individual domain precision
            - Implementation-level detail for actionable security guidance
            - Legal paragraph precision for compliance accuracy

            QUALITY INDICATORS FOR THIS RESPONSE:
            - Overall system precision: {overall_precision_score}%
            - Domain integration quality: {integration_score}%
            - Citation enhancement rate: {enhancement_rate}%
            - Total authoritative sources: {total_citations}

            {sophistication_highlights}

            USER QUERY: {user_query}

            Create a comprehensive step-by-step action plan that:
            1. Integrates requirements from all available domains seamlessly
            2. Uses numbered citations [1] [2] to reference the precise sources
            3. Provides specific, implementable steps that leverage the detailed analysis
            4. Demonstrates the comprehensive coverage achieved through multi-domain integration
            5. Prioritizes actions based on legal requirements and security implementation needs

            FORMAT: Numbered action items starting with action verbs, with precise citations.

            AVAILABLE AUTHORITATIVE SOURCES:
            {formatted_citations}

            Create a comprehensive action plan that showcases the sophisticated analysis while providing clear implementation guidance."""
        )
        
        # Fallback prompt for simpler responses
        self.standard_prompt = ChatPromptTemplate.from_template(
            """You are a compliance expert providing clear, actionable guidance based on retrieved documentation.

            User Query: {user_query}

            Based on the following sources, create a numbered action plan with citations [1] [2] etc.

            Available Sources:
            {formatted_citations}

            Provide specific, implementable steps with proper source citations."""
        )
        
        self.logger.info("Sophisticated prompt templates configured for LLM integration")
    
    def build_comprehensive_action_plan(self, user_query: str, formatted_citations: str,
                                      precision_analysis: Dict[str, Any], 
                                      response_characteristics: Dict[str, Any]) -> str:
        """
        Build a comprehensive action plan using sophisticated LLM integration.
        
        This method represents the culmination of AI-assisted guidance creation. It takes
        the sophisticated citation system, precision analysis, and system insights to
        guide the LLM toward creating responses that demonstrate the value of the
        entire architectural approach while providing maximum actionable value.
        
        Args:
            user_query: The original user query to address
            formatted_citations: Precisely formatted citations from all domains
            precision_analysis: Comprehensive precision analysis results
            response_characteristics: Analysis of response sophistication level
            
        Returns:
            Comprehensive action plan that showcases system capabilities and provides actionable guidance
        """
        self.logger.info("Building comprehensive action plan with sophisticated LLM integration")
        self.building_stats['total_responses_built'] += 1
        
        try:
            # Choose prompt strategy based on response sophistication
            prompt_strategy = self._select_prompt_strategy(precision_analysis, response_characteristics)
            
            # Prepare context for LLM based on sophistication level
            llm_context = self._prepare_sophisticated_llm_context(
                precision_analysis, response_characteristics, formatted_citations
            )
            
            # Execute LLM integration with appropriate prompt
            action_plan = self._execute_llm_integration(
                user_query, llm_context, prompt_strategy
            )
            
            # Validate and enhance the LLM response
            enhanced_action_plan = self._validate_and_enhance_response(
                action_plan, precision_analysis, response_characteristics
            )
            
            # Update success statistics
            self._update_success_statistics(enhanced_action_plan, precision_analysis)
            
            self.logger.info(f"Comprehensive action plan built successfully using {prompt_strategy} strategy")
            
            return enhanced_action_plan
            
        except Exception as e:
            self.building_stats['building_errors'] += 1
            self.building_stats['llm_integration_failures'] += 1
            self.logger.error(f"Error building comprehensive action plan: {e}")
            
            # Return fallback response to maintain workflow
            return self._create_fallback_action_plan(user_query, formatted_citations)
    
    def _select_prompt_strategy(self, precision_analysis: Dict[str, Any], 
                              response_characteristics: Dict[str, Any]) -> str:
        """
        Select the appropriate prompt strategy based on response sophistication.
        
        This method determines whether to use sophisticated prompts that highlight
        system capabilities or simpler prompts for more straightforward responses.
        """
        # Get quality indicators
        overall_precision = precision_analysis.get('system_metrics', {}).get('overall_precision_score', 0)
        integration_score = precision_analysis.get('integration_analysis', {}).get('integration_score', 0)
        sophistication_level = response_characteristics.get('precision_level', 'basic')
        citation_count = response_characteristics.get('citation_count', 0)
        
        # Determine sophistication threshold
        if (overall_precision >= 75 and integration_score >= 70 and 
            sophistication_level in ['exceptional', 'high'] and citation_count >= 4):
            return 'sophisticated'
        elif overall_precision >= 60 and citation_count >= 2:
            return 'enhanced'
        else:
            return 'standard'
    
    def _prepare_sophisticated_llm_context(self, precision_analysis: Dict[str, Any], 
                                         response_characteristics: Dict[str, Any], 
                                         formatted_citations: str) -> Dict[str, Any]:
        """
        Prepare sophisticated context that helps the LLM understand the quality of analysis.
        
        This context enables the LLM to create responses that appropriately showcase
        the sophistication of the system while focusing on user value.
        """
        # Extract system metrics
        system_metrics = precision_analysis.get('system_metrics', {})
        integration_analysis = precision_analysis.get('integration_analysis', {})
        
        # Create sophistication highlights for the prompt
        highlights = response_characteristics.get('sophistication_highlights', [])
        if highlights:
            sophistication_text = "SOPHISTICATION ACHIEVED:\n" + "\n".join(f"• {highlight}" for highlight in highlights)
        else:
            sophistication_text = "ANALYSIS COMPLETED: Multi-domain compliance analysis performed"
        
        context = {
            'overall_precision_score': system_metrics.get('overall_precision_score', 0),
            'integration_score': integration_analysis.get('integration_score', 0),
            'enhancement_rate': system_metrics.get('system_enhancement_rate', 0),
            'total_citations': response_characteristics.get('citation_count', 0),
            'sophistication_highlights': sophistication_text,
            'formatted_citations': formatted_citations
        }
        
        self.logger.debug(f"Sophisticated LLM context prepared: {context['overall_precision_score']}% precision, "
                        f"{context['total_citations']} citations")
        
        return context
    
    def _execute_llm_integration(self, user_query: str, llm_context: Dict[str, Any], 
                               prompt_strategy: str) -> str:
        """
        Execute the LLM integration using the appropriate prompt strategy.
        
        This method handles the actual LLM interaction, using the sophisticated
        prompts and context to guide the AI toward creating valuable responses.
        """
        try:
            # Select appropriate prompt based on strategy
            if prompt_strategy == 'sophisticated':
                prompt_template = self.sophisticated_prompt
                self.building_stats['sophisticated_prompts_used'] += 1
                self.logger.debug("Using sophisticated prompt template with full context")
            else:
                prompt_template = self.standard_prompt
                self.building_stats['fallback_prompts_used'] += 1
                self.logger.debug("Using standard prompt template")
            
            # Create the chain and execute
            chain = prompt_template | self.model
            
            # Prepare prompt variables
            prompt_variables = {
                'user_query': user_query,
                **llm_context
            }
            
            # Execute LLM integration
            response = chain.invoke(prompt_variables)
            
            self.building_stats['llm_integration_successes'] += 1
            self.logger.debug(f"LLM integration successful: {len(response.content)} characters generated")
            
            return response.content
            
        except Exception as e:
            self.building_stats['llm_integration_failures'] += 1
            self.logger.error(f"LLM integration failed: {e}")
            raise
    
    def _validate_and_enhance_response(self, action_plan: str, precision_analysis: Dict[str, Any], 
                                     response_characteristics: Dict[str, Any]) -> str:
        """
        Validate and enhance the LLM response to ensure quality and consistency.
        
        This method performs quality checks on the LLM output and applies
        enhancements to ensure the response meets the standards expected
        from a sophisticated compliance system.
        """
        # Validate basic response structure
        validation_results = self._validate_response_structure(action_plan)
        
        # Apply quality enhancements based on validation
        enhanced_response = self._apply_quality_enhancements(
            action_plan, validation_results, response_characteristics
        )
        
        # Calculate response quality score
        quality_score = self._calculate_response_quality_score(
            enhanced_response, precision_analysis, validation_results
        )
        
        # Update quality statistics
        self._update_quality_statistics(quality_score)
        
        self.logger.debug(f"Response validation completed: {quality_score:.1f}% quality score")
        
        return enhanced_response
    
    def _validate_response_structure(self, action_plan: str) -> Dict[str, Any]:
        """
        Validate the structure and content of the LLM response.
        
        This ensures the response meets basic quality standards and contains
        the expected elements for effective compliance guidance.
        """
        validation_results = {
            'has_numbered_items': False,
            'has_citations': False,
            'has_action_verbs': False,
            'has_sufficient_length': False,
            'citation_count': 0,
            'action_item_count': 0,
            'structural_quality': 'unknown'
        }
        
        # Check for numbered action items
        import re
        numbered_items = re.findall(r'^\d+\.', action_plan, re.MULTILINE)
        validation_results['action_item_count'] = len(numbered_items)
        validation_results['has_numbered_items'] = len(numbered_items) >= 3
        
        # Check for citation usage
        citations = re.findall(r'\[\d+\]', action_plan)
        validation_results['citation_count'] = len(citations)
        validation_results['has_citations'] = len(citations) >= 3
        
        # Check for action verbs
        action_verbs = ['implement', 'configure', 'establish', 'ensure', 'verify', 'document', 'review', 'update']
        verb_count = sum(1 for verb in action_verbs if verb.lower() in action_plan.lower())
        validation_results['has_action_verbs'] = verb_count >= 3
        
        # Check response length
        validation_results['has_sufficient_length'] = len(action_plan) >= 500
        
        # Assess overall structural quality
        quality_indicators = [
            validation_results['has_numbered_items'],
            validation_results['has_citations'],
            validation_results['has_action_verbs'],
            validation_results['has_sufficient_length']
        ]
        
        quality_score = sum(quality_indicators) / len(quality_indicators)
        
        if quality_score >= 0.8:
            validation_results['structural_quality'] = 'excellent'
        elif quality_score >= 0.6:
            validation_results['structural_quality'] = 'good'
        elif quality_score >= 0.4:
            validation_results['structural_quality'] = 'fair'
        else:
            validation_results['structural_quality'] = 'poor'
        
        return validation_results
    
    def _apply_quality_enhancements(self, action_plan: str, validation_results: Dict[str, Any], 
                                   response_characteristics: Dict[str, Any]) -> str:
        """
        Apply quality enhancements to improve the response based on validation results.
        
        This method fixes common issues and adds improvements to ensure the
        response meets the high standards expected from the sophisticated system.
        """
        enhanced_plan = action_plan
        
        # Enhance formatting if needed
        if validation_results['structural_quality'] in ['fair', 'poor']:
            enhanced_plan = self._improve_response_formatting(enhanced_plan)
        
        # Add sophistication indicators if appropriate
        if (response_characteristics.get('precision_level') in ['exceptional', 'high'] and 
            validation_results['structural_quality'] in ['excellent', 'good']):
            enhanced_plan = self._add_sophistication_indicators(enhanced_plan, response_characteristics)
        
        return enhanced_plan
    
    def _improve_response_formatting(self, action_plan: str) -> str:
        """
        Improve the formatting and structure of the response.
        
        This method applies consistent formatting to ensure professional
        presentation that reflects the sophistication of the underlying system.
        """
        # Ensure proper line spacing and structure
        lines = action_plan.split('\n')
        improved_lines = []
        
        for line in lines:
            line = line.strip()
            if line:
                # Ensure numbered items have consistent formatting
                import re
                if re.match(r'^\d+\.', line):
                    # Ensure space after period
                    line = re.sub(r'^(\d+\.)(\S)', r'\1 \2', line)
                
                improved_lines.append(line)
        
        return '\n\n'.join(improved_lines)
    
    def _add_sophistication_indicators(self, action_plan: str, 
                                     response_characteristics: Dict[str, Any]) -> str:
        """
        Add subtle indicators that showcase the sophistication of the analysis.
        
        This method adds context that helps users understand the quality of
        analysis they received without overwhelming the actionable content.
        """
        # Only add indicators for highly sophisticated responses
        if response_characteristics.get('domain_count', 0) >= 3:
            sophistication_note = ("\n\n*This action plan integrates guidance from multiple specialized domains "
                                 f"with {response_characteristics.get('citation_count', 0)} precise authoritative references.*")
            return action_plan + sophistication_note
        
        return action_plan
    
    def _calculate_response_quality_score(self, response: str, precision_analysis: Dict[str, Any], 
                                        validation_results: Dict[str, Any]) -> float:
        """
        Calculate a quality score for the generated response.
        
        This score helps track how well the LLM integration is working and
        whether the sophisticated context is producing better outputs.
        """
        # Base score from structural validation
        structure_indicators = [
            validation_results['has_numbered_items'],
            validation_results['has_citations'],
            validation_results['has_action_verbs'],
            validation_results['has_sufficient_length']
        ]
        structure_score = (sum(structure_indicators) / len(structure_indicators)) * 40
        
        # Citation integration score
        citation_count = validation_results['citation_count']
        expected_citations = min(10, precision_analysis.get('system_metrics', {}).get('total_citations', 0))
        
        if expected_citations > 0:
            citation_integration = min(citation_count / expected_citations, 1.0) * 30
        else:
            citation_integration = 0
        
        # Content quality indicators
        content_score = 20  # Base content score
        
        # Action orientation
        action_items = validation_results['action_item_count']
        if action_items >= 5:
            content_score += 10
        elif action_items >= 3:
            content_score += 5
        
        total_score = structure_score + citation_integration + content_score
        
        return round(min(total_score, 100), 1)
    
    def _update_quality_statistics(self, quality_score: float) -> None:
        """
        Update quality statistics based on the response quality assessment.
        
        This tracking helps understand how well the LLM integration is performing
        and whether sophisticated prompts are producing better results.
        """
        # Update average quality score
        current_avg = self.building_stats['average_response_quality_score']
        response_count = self.building_stats['total_responses_built']
        
        new_avg = ((current_avg * (response_count - 1)) + quality_score) / response_count
        self.building_stats['average_response_quality_score'] = round(new_avg, 1)
    
    def _update_success_statistics(self, enhanced_response: str, precision_analysis: Dict[str, Any]) -> None:
        """
        Update success statistics based on the completed response building process.
        
        This tracking provides insights into how well the sophisticated integration
        approach is working compared to simpler alternatives.
        """
        # Calculate citation integration success
        import re
        citations_used = len(re.findall(r'\[\d+\]', enhanced_response))
        total_available = precision_analysis.get('system_metrics', {}).get('total_citations', 0)
        
        if total_available > 0:
            integration_rate = min((citations_used / total_available), 1.0) * 100
            
            # Update rolling average
            current_rate = self.building_stats['citation_integration_success_rate']
            success_count = self.building_stats['llm_integration_successes']
            
            if success_count > 1:
                new_rate = ((current_rate * (success_count - 1)) + integration_rate) / success_count
                self.building_stats['citation_integration_success_rate'] = round(new_rate, 1)
            else:
                self.building_stats['citation_integration_success_rate'] = round(integration_rate, 1)
    
    def _create_fallback_action_plan(self, user_query: str, formatted_citations: str) -> str:
        """
        Create a fallback action plan when sophisticated integration fails.
        
        This ensures users receive actionable guidance even when the advanced
        LLM integration encounters issues, demonstrating graceful degradation.
        """
        self.logger.warning("Creating fallback action plan due to LLM integration failure")
        
        fallback_parts = [
            f"COMPLIANCE ACTION PLAN FOR: {user_query}",
            "",
            "Based on available compliance documentation, implement the following steps:",
            "",
            "1. Review applicable regulatory requirements from retrieved sources",
            "2. Assess current organizational compliance status",
            "3. Implement necessary policy and procedure updates",
            "4. Establish monitoring and validation processes",
            "5. Document compliance activities and maintain records",
            "",
            "REFERENCE SOURCES:",
            formatted_citations if formatted_citations else "Compliance documentation retrieved from internal systems",
            "",
            "*This is a fallback response. For enhanced guidance, please retry your query.*"
        ]
        
        return "\n".join(fallback_parts)
    
    def assess_response_sophistication(self, response: str, precision_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess the sophistication level achieved in the generated response.
        
        This method provides insights into how well the LLM integration leveraged
        the sophisticated citation system and analysis to create valuable output.
        
        Args:
            response: The generated action plan response
            precision_analysis: Comprehensive precision analysis results
            
        Returns:
            Assessment of response sophistication and effectiveness
        """
        # Analyze response characteristics
        response_analysis = self._analyze_response_characteristics(response)
        
        # Assess citation utilization
        citation_utilization = self._assess_citation_utilization(response, precision_analysis)
        
        # Evaluate integration effectiveness
        integration_effectiveness = self._evaluate_integration_effectiveness(
            response_analysis, citation_utilization, precision_analysis
        )
        
        # Calculate overall sophistication score
        sophistication_score = self._calculate_sophistication_score(
            response_analysis, citation_utilization, integration_effectiveness
        )
        
        sophistication_assessment = {
            'sophistication_score': sophistication_score,
            'response_characteristics': response_analysis,
            'citation_utilization': citation_utilization,
            'integration_effectiveness': integration_effectiveness,
            'achievement_level': self._categorize_sophistication_level(sophistication_score),
            'improvement_recommendations': self._generate_improvement_recommendations(
                response_analysis, citation_utilization
            )
        }
        
        self.logger.debug(f"Response sophistication assessed: {sophistication_score:.1f}% achievement level")
        
        return sophistication_assessment
    
    def _analyze_response_characteristics(self, response: str) -> Dict[str, Any]:
        """
        Analyze the characteristics of the generated response.
        
        This method examines the response structure, content quality, and
        sophistication indicators to understand how well the LLM performed.
        """
        import re
        
        characteristics = {
            'length': len(response),
            'action_items': len(re.findall(r'^\d+\.', response, re.MULTILINE)),
            'citations_used': len(re.findall(r'\[\d+\]', response)),
            'action_verbs': self._count_action_verbs(response),
            'technical_terms': self._count_technical_terms(response),
            'structure_quality': self._assess_structure_quality(response)
        }
        
        return characteristics
    
    def _count_action_verbs(self, response: str) -> int:
        """Count action-oriented verbs that indicate actionable guidance."""
        action_verbs = [
            'implement', 'configure', 'establish', 'ensure', 'verify', 'document', 
            'review', 'update', 'create', 'develop', 'maintain', 'monitor'
        ]
        
        response_lower = response.lower()
        return sum(1 for verb in action_verbs if verb in response_lower)
    
    def _count_technical_terms(self, response: str) -> int:
        """Count technical terms that indicate sophisticated analysis."""
        technical_terms = [
            'gdpr', 'compliance', 'regulation', 'procedure', 'security', 'privacy',
            'implementation', 'assessment', 'monitoring', 'documentation', 'validation'
        ]
        
        response_lower = response.lower()
        return sum(1 for term in technical_terms if term in response_lower)
    
    def _assess_structure_quality(self, response: str) -> str:
        """Assess the overall structural quality of the response."""
        import re
        
        # Check for consistent numbering
        numbered_items = re.findall(r'^\d+\.', response, re.MULTILINE)
        has_consistent_numbering = len(numbered_items) >= 3
        
        # Check for proper paragraph structure
        paragraphs = [p.strip() for p in response.split('\n\n') if p.strip()]
        has_good_paragraphs = len(paragraphs) >= 3
        
        # Check for citations integration
        has_citations = '[' in response and ']' in response
        
        quality_indicators = [has_consistent_numbering, has_good_paragraphs, has_citations]
        quality_score = sum(quality_indicators) / len(quality_indicators)
        
        if quality_score >= 0.8:
            return 'excellent'
        elif quality_score >= 0.6:
            return 'good'
        elif quality_score >= 0.4:
            return 'fair'
        else:
            return 'poor'
    
    def _assess_citation_utilization(self, response: str, precision_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess how effectively the response utilized the available citations.
        
        This measures whether the LLM effectively integrated the sophisticated
        citation system into actionable guidance.
        """
        import re
        
        citations_used = len(re.findall(r'\[\d+\]', response))
        total_available = precision_analysis.get('system_metrics', {}).get('total_citations', 0)
        
        if total_available > 0:
            utilization_rate = (citations_used / total_available) * 100
        else:
            utilization_rate = 0
        
        # Assess utilization quality
        if utilization_rate >= 80:
            utilization_quality = 'excellent'
        elif utilization_rate >= 60:
            utilization_quality = 'good'
        elif utilization_rate >= 40:
            utilization_quality = 'fair'
        elif utilization_rate >= 20:
            utilization_quality = 'poor'
        else:
            utilization_quality = 'very_poor'
        
        return {
            'citations_used': citations_used,
            'total_available': total_available,
            'utilization_rate': round(utilization_rate, 1),
            'utilization_quality': utilization_quality
        }
    
    def _evaluate_integration_effectiveness(self, response_analysis: Dict[str, Any], 
                                          citation_utilization: Dict[str, Any], 
                                          precision_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate how effectively the response integrated sophisticated system capabilities.
        
        This assesses whether the LLM successfully leveraged the architectural
        sophistication to create responses that justify the complex processing.
        """
        # Get sophistication indicators from precision analysis
        system_metrics = precision_analysis.get('system_metrics', {})
        overall_precision = system_metrics.get('overall_precision_score', 0)
        
        # Calculate integration score based on multiple factors
        structure_score = 25 if response_analysis['structure_quality'] == 'excellent' else 15
        citation_score = min(citation_utilization['utilization_rate'] * 0.3, 30)
        content_score = min((response_analysis['action_verbs'] + response_analysis['technical_terms']) * 2, 25)
        precision_bonus = min(overall_precision * 0.2, 20)
        
        integration_score = structure_score + citation_score + content_score + precision_bonus
        
        # Assess effectiveness level
        if integration_score >= 85:
            effectiveness_level = 'exceptional'
        elif integration_score >= 70:
            effectiveness_level = 'high'
        elif integration_score >= 55:
            effectiveness_level = 'moderate'
        elif integration_score >= 40:
            effectiveness_level = 'low'
        else:
            effectiveness_level = 'poor'
        
        return {
            'integration_score': round(integration_score, 1),
            'effectiveness_level': effectiveness_level,
            'component_scores': {
                'structure': structure_score,
                'citations': citation_score,
                'content': content_score,
                'precision_bonus': precision_bonus
            }
        }
    
    def _calculate_sophistication_score(self, response_analysis: Dict[str, Any], 
                                      citation_utilization: Dict[str, Any], 
                                      integration_effectiveness: Dict[str, Any]) -> float:
        """
        Calculate an overall sophistication score for the response.
        
        This provides a single metric that captures how well the response
        demonstrates and utilizes the sophisticated system capabilities.
        """
        # Weight different aspects of sophistication
        structure_weight = 0.25
        citation_weight = 0.35
        integration_weight = 0.40
        
        # Normalize scores to 0-100 scale
        structure_score = 100 if response_analysis['structure_quality'] == 'excellent' else 60
        citation_score = citation_utilization['utilization_rate']
        integration_score = integration_effectiveness['integration_score']
        
        sophistication_score = (
            structure_score * structure_weight +
            citation_score * citation_weight +
            integration_score * integration_weight
        )
        
        return round(min(sophistication_score, 100), 1)
    
    def _categorize_sophistication_level(self, sophistication_score: float) -> str:
        """
        Categorize the sophistication level achieved in the response.
        
        This provides a human-readable assessment of how well the response
        leveraged the sophisticated system capabilities.
        """
        if sophistication_score >= 90:
            return 'exceptional_sophistication'
        elif sophistication_score >= 80:
            return 'high_sophistication'
        elif sophistication_score >= 70:
            return 'good_sophistication'
        elif sophistication_score >= 60:
            return 'moderate_sophistication'
        elif sophistication_score >= 50:
            return 'basic_sophistication'
        else:
            return 'limited_sophistication'
    
    def _generate_improvement_recommendations(self, response_analysis: Dict[str, Any], 
                                            citation_utilization: Dict[str, Any]) -> List[str]:
        """
        Generate recommendations for improving response sophistication.
        
        These recommendations help optimize the LLM integration to better
        leverage the sophisticated system capabilities.
        """
        recommendations = []
        
        # Structure recommendations
        if response_analysis['structure_quality'] in ['fair', 'poor']:
            recommendations.append("Improve response structure with consistent numbering and clear paragraphs")
        
        # Citation utilization recommendations
        utilization_rate = citation_utilization['utilization_rate']
        if utilization_rate < 60:
            recommendations.append("Increase citation utilization to better leverage sophisticated analysis")
        
        # Content depth recommendations
        if response_analysis['action_verbs'] < 5:
            recommendations.append("Include more action-oriented guidance for implementation")
        
        if response_analysis['technical_terms'] < 8:
            recommendations.append("Incorporate more domain-specific terminology to demonstrate expertise")
        
        # Length and detail recommendations
        if response_analysis['length'] < 800:
            recommendations.append("Expand response detail to fully utilize available sophisticated sources")
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    def get_building_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about response building operations.
        
        Returns:
            Dictionary containing detailed building metrics and performance data
        """
        stats = dict(self.building_stats)
        
        # Calculate success rates
        if stats['total_responses_built'] > 0:
            llm_success_rate = (stats['llm_integration_successes'] / stats['total_responses_built']) * 100
            stats['llm_success_rate_percent'] = round(llm_success_rate, 1)
            
            sophisticated_usage_rate = (stats['sophisticated_prompts_used'] / stats['total_responses_built']) * 100
            stats['sophisticated_usage_rate_percent'] = round(sophisticated_usage_rate, 1)
        
        return stats
    
    def log_building_summary(self) -> None:
        """
        Log a comprehensive summary of all response building operations.
        
        This provides insights into how well the LLM integration is working
        and whether sophisticated prompts are producing better results.
        """
        stats = self.get_building_statistics()
        
        self.logger.info("=== SOPHISTICATED RESPONSE BUILDING SUMMARY ===")
        self.logger.info(f"Total responses built: {stats['total_responses_built']}")
        self.logger.info(f"LLM integration success rate: {stats.get('llm_success_rate_percent', 0)}%")
        self.logger.info(f"Sophisticated prompts used: {stats['sophisticated_prompts_used']} ({stats.get('sophisticated_usage_rate_percent', 0)}%)")
        self.logger.info(f"Average response quality: {stats['average_response_quality_score']:.1f}%")
        self.logger.info(f"Citation integration success: {stats['citation_integration_success_rate']:.1f}%")
        self.logger.info(f"Building errors: {stats['building_errors']}")
        
        # Provide performance assessment
        avg_quality = stats['average_response_quality_score']
        if avg_quality >= 85:
            self.logger.info("Excellent LLM integration - sophisticated prompts producing high-quality results")
        elif avg_quality >= 75:
            self.logger.info("Good LLM integration - system sophistication effectively leveraged")
        elif avg_quality >= 65:
            self.logger.info("Moderate LLM integration - optimization opportunities available")
        else:
            self.logger.info("LLM integration needs improvement - review prompt strategies and context")


def create_summarization_response_builder(logger: logging.Logger) -> SummarizationResponseBuilder:
    """
    Factory function to create a configured summarization response builder.
    
    This provides a clean interface for creating builder instances with
    proper dependency injection of the logger.
    """
    return SummarizationResponseBuilder(logger)