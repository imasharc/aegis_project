"""
EU-Focused Realistic Business Test Queries for GDPR/Polish Law Compliance System
Designed specifically for European business scenarios that require sophisticated 
multi-domain analysis within the EU regulatory framework
"""

EU_BUSINESS_TEST_QUERIES = {
    "hr_employment_scenarios": [
        "Our Warsaw office wants to implement employee productivity monitoring software that tracks keystrokes, screen time, and application usage, with data processing handled by our German IT service provider. What specific steps do we need to take to ensure this complies with GDPR worker surveillance rules, Polish employment law, and our internal security procedures for cross-border employee monitoring within the EU?",
        
        "We're conducting background checks on Polish job candidates that include criminal record searches and reference checks, using a Dutch HR technology platform that processes data across multiple EU member states. How do we structure our data processing agreements and consent processes to meet GDPR Article 6 and 9 requirements while following Polish employment regulations and EU-wide background check standards?",
        
        "Our Polish subsidiary needs to transfer employee performance data and training records to our Irish headquarters for EU-wide HR analytics and talent management. What specific documentation, internal approvals, and GDPR compliance measures do we need for this intra-EU data processing?",
        
        "We discovered that a former employee in our Krakow office downloaded customer contact lists to their personal device before leaving and may have shared this data with a competitor based in the Czech Republic. Our internal incident response procedure was triggered, but what are our specific obligations for breach notification under Polish law and GDPR Article 33 and 34 requirements for this cross-border incident?",
        
        "Our Polish remote workers are requesting to use personal devices for accessing customer data through our CRM system, which is hosted by a French cloud service provider. How do we update our BYOD policy and technical security controls to meet GDPR security requirements while complying with Polish labor law and ensuring proper vendor management across EU jurisdictions?"
    ],
    
    "customer_data_marketing": [
        "Our marketing team wants to create detailed customer personas using purchase history and website behavior from our Polish customers, then share these insights with our advertising agency in Berlin and our analytics partner in Amsterdam. What consent mechanisms, data processing agreements, and internal approval workflows do we need for this multi-country EU data processing?",
        
        "We're launching an AI-powered recommendation engine for our Polish e-commerce site that analyzes customer behavior patterns. The AI system is developed by a Italian technology company and hosted on servers in Ireland. What GDPR Article 22 compliance steps and internal AI governance procedures do we need to implement for this EU-wide automated decision-making system?",
        
        "Our customer service team in Gdansk handles support tickets containing personal data, and we want to use a cloud platform provided by a Danish company with data centers in Frankfurt and Dublin. The platform includes AI-powered sentiment analysis and automatic categorization. How do we structure this to comply with GDPR requirements for automated processing and intra-EU data flows?",
        
        "We need to implement cookie consent management on our Polish website that integrates with our marketing automation platform operated by a Spanish company serving all EU markets. The system needs to handle granular consent for analytics, advertising, and personalization while maintaining effectiveness across different EU member state cookie regulations. What technical and legal requirements must we meet?",
        
        "Our loyalty program in Poland collects purchase data, location information, and preference data to offer personalized rewards. We want to expand this program to include partners like German hotels and French restaurants who would access customer profiles for cross-border reward redemption. How do we structure the consent, data sharing agreements, and internal controls for this multi-party EU processing?"
    ],
    
    "it_security_operations": [
        "We're migrating our Polish operations' customer database to cloud services provided by a German company with infrastructure distributed across Ireland, Netherlands, and Germany. Our internal security procedures require specific encryption and access controls. How do we ensure the migration meets GDPR technical and organizational measures while maintaining our security compliance requirements across these EU jurisdictions?",
        
        "Our IT security team detected suspicious network activity in our Warsaw office and needs to collect and analyze employee internet activity logs, email metadata, and system access logs for forensic investigation. Some of this data may have been accessed from our branch offices in Prague and Vienna. What are our obligations and limitations under GDPR and Polish law for this cross-border EU internal investigation?",
        
        "We want to implement behavioral analytics software provided by a Swedish security company that monitors user access patterns across our Polish office systems and flags anomalous behavior. This involves processing employee badge data with biometric components and analyzing work patterns. How do we balance security needs with GDPR privacy requirements for this EU-based security solution?",
        
        "Our Polish office experienced a ransomware attack that encrypted customer databases. We have backups stored with a Luxembourg-based service provider, but the attackers claim to have exfiltrated personal data before encryption. We're 18 hours post-incident and need to coordinate our technical recovery with legal notification obligations under Polish supervisory authority requirements and GDPR Article 33 timelines.",
        
        "We're implementing single sign-on (SSO) across our European operations, which means employee authentication data from our Polish office will be processed by our identity management system hosted in Ireland and managed by a Belgian IT services company. The system includes privileged access management for sensitive systems. What intra-EU data flow protections and security controls do we need?"
    ],
    
    "vendor_third_party_management": [
        "Our Polish manufacturing facility wants to share production quality data with our suppliers in the Czech Republic and Slovakia to improve defect prediction through an Austrian Industry 4.0 platform. This data includes timestamps, employee IDs, and production metrics that could be considered personal data. How do we structure data processing agreements and GDPR compliance for this multi-country EU industrial IoT scenario?",
        
        "We're implementing a customer support platform where our Polish customers' support requests might be handled by agents in our Romanian, Hungarian, or Portuguese offices depending on language skills and availability. The platform, provided by a German company, includes call recording, chat transcripts, and AI-powered ticket routing. What multi-jurisdictional EU compliance framework do we need?",
        
        "Our accounting team in Poland uses a expense management platform provided by a Finnish SaaS company that processes employee credit card transactions, travel bookings, and receipt images. The vendor's servers are distributed across Stockholm, Frankfurt, and Dublin with data replication. How do we ensure our vendor contract and technical controls meet GDPR requirements for this EU-wide processing?",
        
        "We're conducting due diligence for acquiring a competitor based in the Czech Republic, which requires analyzing their customer database, employee records, and business intelligence data. Some analysis will be performed by our consultants in Germany and France. What legal basis under GDPR, safeguards, and internal controls do we need for this temporary cross-border EU data processing?",
        
        "Our Polish retail locations want to implement facial recognition technology for loss prevention, with video analytics processed by a Spanish security company using Italian facial recognition algorithms. The system would create biometric profiles and behavioral analysis reports. How do we evaluate the privacy impact and implement appropriate safeguards under GDPR for this EU-wide biometric processing system?"
    ],
    
    "financial_regulatory_scenarios": [
        "Our Polish fintech subsidiary needs to share transaction monitoring data with our anti-money laundering system hosted in Luxembourg and managed by a German compliance technology company. This includes customer transaction patterns, risk scores, and suspicious activity reports across EU jurisdictions. How do we balance AML regulatory requirements with GDPR data protection obligations in this multi-country EU financial services context?",
        
        "We're implementing open banking APIs in Poland that will allow third-party financial apps from across the EU to access our customers' account data with their consent. Our internal risk management requires additional monitoring and controls for these data flows to German, French, and Dutch fintech partners. What consent mechanisms, technical controls, and governance procedures do we need under PSD2 and GDPR?",
        
        "Our insurance claims processing system in Poland uses AI developed by a Dutch company to automatically evaluate claims photos and detect potential fraud. The system makes recommendations that human adjusters usually follow and includes data sharing with our reinsurance partners in Germany and Switzerland. What GDPR compliance framework do we need for this automated decision-making in EU insurance operations?",
        
        "We discovered that our payment processor in Poland has been sharing transaction data with data analytics companies in Ireland and the Netherlands for market research purposes without explicit customer consent. We need to assess our liability, notification obligations, and remediation steps while maintaining payment processing operations across EU markets.",
        
        "Our Polish investment advisory service wants to implement robo-advisor capabilities developed by a French fintech company that make automated investment recommendations. The system processes data from multiple EU financial markets and executes trades through brokers in Frankfurt and Milan. How do we ensure compliance with both MiFID II financial regulations and GDPR automated decision-making rules across these EU jurisdictions?"
    ],
    
    "healthcare_life_sciences": [
        "Our pharmaceutical research division in Poland is conducting clinical trials that involve collecting genetic data, health records, and lifestyle information from participants. We need to share this data with research partners in Germany, Belgium, and the Netherlands for collaborative analysis under EU clinical trial regulations. What special protections and legal bases do we need for this international EU health research under GDPR Article 9?",
        
        "We're implementing an employee wellness program in our Polish offices that tracks fitness data through wearable devices provided by a Danish company, with health analytics performed by a German platform. The program monitors cafeteria purchases and provides health recommendations that may affect insurance premiums. How do we structure this program to meet GDPR health data protection requirements across these EU jurisdictions?",
        
        "Our telemedicine platform serving Polish patients stores consultation recordings, prescription data, and diagnostic images on a cloud platform managed by a Irish company with data centers in Dublin and Frankfurt. We want to implement AI-powered symptom checking developed by a Belgian company. What technical and legal safeguards do we need under GDPR for this EU-wide health data processing?",
        
        "We discovered that our corporate health insurance provider in Poland has been sharing employee health claims data with our HR departments in Prague and Vienna for absence management purposes across our EU operations. We need to assess whether this processing was lawful under GDPR and what corrective actions we need to take.",
        
        "Our Polish laboratory wants to implement blockchain technology developed by a Swiss company for secure sharing of medical test results with healthcare providers across the EU. The system would create immutable records stored across nodes in multiple EU countries and enable patient-controlled access to their data. How do we ensure this meets GDPR requirements for data accuracy, deletion rights, and controller/processor relationships in a distributed EU healthcare network?"
    ],
    
    "emerging_technology_scenarios": [
        "We're piloting an IoT-enabled smart office system in our Warsaw headquarters using sensors manufactured by a German company and analytics provided by a Swedish platform. The system monitors space utilization, energy usage, and can identify individual employees through badge interactions and movement patterns. How do we implement this while meeting GDPR privacy-by-design requirements for this EU-wide IoT deployment?",
        
        "Our Polish e-commerce platform wants to implement voice commerce capabilities through smart speakers, allowing customers to make purchases using voice commands processed by a French AI company with servers in Ireland. This involves processing voice biometrics, purchase history, and behavioral analytics. What consent mechanisms and technical controls do we need under GDPR for this EU-based voice processing system?",
        
        "We're developing a machine learning model using customer transaction data from our Polish operations to detect fraudulent activities. We want to share model insights with our subsidiaries in Hungary, Romania, and the Czech Republic, and potentially with other EU financial institutions for collective fraud prevention. How do we structure this collaborative approach under GDPR while maintaining competitive advantages?",
        
        "Our Polish mobile app collects location data, device identifiers, and usage patterns to provide personalized services. We want to monetize this data by offering insights to retail partners across the EU about foot traffic and consumer behavior patterns. A Belgian data analytics company would process and distribute these insights. What legal basis, consent mechanisms, and technical controls do we need under GDPR?",
        
        "We're implementing robotic process automation (RPA) in our Polish finance operations using software developed by a Danish company that will automatically process invoices, expense reports, and payroll data. The RPA system logs all processed information and makes decisions based on machine learning algorithms trained on EU financial data. How do we ensure compliance with GDPR automated processing requirements for this EU-wide financial automation?"
    ],
    
    "cross_border_eu_coordination": [
        "Our Polish data protection officer needs to coordinate with supervisory authorities in Germany, France, and Ireland regarding a cross-border processing operation that affects customers in all four countries. We're using a Belgian cloud service provider for the technical infrastructure. What are our obligations under GDPR's one-stop-shop mechanism and how do we manage multi-authority coordination?",
        
        "We're establishing a EU-wide customer data lake that consolidates information from our operations in Poland, Germany, Czech Republic, and Austria. The data lake will be managed by a Dutch technology company and hosted in Irish data centers. What governance framework, technical controls, and legal documentation do we need for this comprehensive EU data consolidation project?",
        
        "Our Polish subsidiary wants to implement a blockchain-based supply chain tracking system that involves partners in Germany, Italy, Hungary, and Romania. The system would record personal data about drivers, warehouse workers, and quality inspectors across the supply chain. How do we structure GDPR compliance for this multi-country EU blockchain implementation?",
        
        "We're launching a EU-wide loyalty program that allows customers to earn and redeem points across our Polish, German, Czech, and Austrian locations. Customer profiles would be managed by a Spanish customer experience platform with AI-powered personalization. What consent mechanisms and cross-border data governance do we need under GDPR?",
        
        "Our crisis management team needs to implement an incident response procedure that can handle data breaches affecting multiple EU countries simultaneously. We have operations in Poland, Germany, Netherlands, and Belgium, each with different local supervisory authority requirements. How do we create internal procedures that ensure coordinated GDPR compliance across these jurisdictions?"
    ]
}

def get_all_eu_business_queries():
    """
    Return a comprehensive list of EU-focused realistic business test queries.
    
    These queries are specifically designed to test GDPR compliance capabilities
    within the European Union context, including:
    - Intra-EU data transfers and processing coordination
    - Multi-member state regulatory compliance requirements  
    - EU-based vendor and service provider relationships
    - Cross-border incident response and breach notification
    - Emerging technology implementation within EU regulatory framework
    
    This focused approach provides more targeted validation of the system's
    ability to handle realistic European business scenarios while testing
    the sophisticated coordination between GDPR knowledge, Polish law specifics,
    and internal security procedures.
    """
    all_queries = []
    for category in EU_BUSINESS_TEST_QUERIES:
        all_queries.extend(EU_BUSINESS_TEST_QUERIES[category])
    return all_queries

def get_eu_business_queries_by_category(category):
    """Return EU business queries from a specific operational category."""
    return EU_BUSINESS_TEST_QUERIES.get(category, [])

def get_eu_business_categories():
    """Return all available EU business scenario categories."""
    return list(EU_BUSINESS_TEST_QUERIES.keys())

def get_complex_eu_multi_domain_queries():
    """
    Return EU queries that specifically require sophisticated coordination between
    all three knowledge domains (GDPR, Polish law, and internal procedures).
    
    These represent the most challenging test cases for validating the enhanced
    multi-agent system's ability to create comprehensive, actionable compliance
    guidance within the EU regulatory environment.
    """
    complex_queries = []
    
    # EU-specific multi-domain indicators that require sophisticated coordination
    eu_multi_domain_indicators = [
        "cross-border", "multi-country", "EU-wide", "intra-EU", "member state",
        "supervisory authority", "one-stop-shop", "AI", "automated", "biometric",
        "incident", "breach", "internal", "procedure", "vendor", "cloud"
    ]
    
    for category, queries in EU_BUSINESS_TEST_QUERIES.items():
        for query in queries:
            # Count EU-specific complexity indicators
            indicator_count = sum(1 for indicator in eu_multi_domain_indicators 
                                if indicator.lower() in query.lower())
            
            # Select queries with high EU complexity indicators
            if indicator_count >= 3:
                complex_queries.append(query)
    
    return complex_queries

def get_intra_eu_coordination_queries():
    """
    Return queries specifically focused on coordination between EU member states.
    
    These queries test the system's understanding of how GDPR works across
    different EU member state implementations and regulatory approaches.
    This is particularly valuable for validating the Polish Law Agent's
    ability to identify meaningful differences in member state approaches.
    """
    return EU_BUSINESS_TEST_QUERIES.get("cross_border_eu_coordination", [])

def get_eu_scenario_analysis(query):
    """
    Analyze what makes an EU business query complex and what domains it should engage.
    
    This EU-focused analysis helps understand why certain queries are effective
    tests for the multi-agent system by identifying which knowledge domains they
    require and what types of intra-EU coordination they demand.
    """
    analysis = {
        "query": query,
        "gdpr_domains": [],
        "polish_law_aspects": [],
        "internal_procedure_needs": [],
        "eu_complexity_factors": [],
        "member_states_involved": []
    }
    
    # GDPR domain analysis (same as before but EU-focused)
    if "consent" in query.lower():
        analysis["gdpr_domains"].append("Articles 6/7 - Legal basis and consent")
    if "transfer" in query.lower() or "cross-border" in query.lower():
        analysis["gdpr_domains"].append("Intra-EU data flows and coordination")
    if "breach" in query.lower() or "incident" in query.lower():
        analysis["gdpr_domains"].append("Articles 33/34 - Breach notification and one-stop-shop")
    if "automated" in query.lower() or "AI" in query.lower():
        analysis["gdpr_domains"].append("Article 22 - Automated decision-making")
    if "biometric" in query.lower() or "health" in query.lower():
        analysis["gdpr_domains"].append("Article 9 - Special category data")
    
    # Polish law and member state coordination aspects
    if "polish" in query.lower() or "poland" in query.lower():
        analysis["polish_law_aspects"].append("Polish GDPR implementation specifics")
    if "supervisory authority" in query.lower() or "coordination" in query.lower():
        analysis["polish_law_aspects"].append("Multi-authority coordination requirements")
    if "employee" in query.lower() or "worker" in query.lower():
        analysis["polish_law_aspects"].append("Polish employment law intersection with GDPR")
    
    # Internal procedure requirements
    if "internal" in query.lower() or "policy" in query.lower():
        analysis["internal_procedure_needs"].append("Internal policy framework for EU operations")
    if "security" in query.lower() or "technical" in query.lower():
        analysis["internal_procedure_needs"].append("Technical security controls for EU compliance")
    if "vendor" in query.lower() or "service provider" in query.lower():
        analysis["internal_procedure_needs"].append("EU vendor management procedures")
    
    # EU-specific complexity factors
    if "multi-country" in query.lower() or "EU-wide" in query.lower():
        analysis["eu_complexity_factors"].append("Multi-member state coordination")
    if any(tech in query.lower() for tech in ["AI", "blockchain", "IoT", "biometric"]):
        analysis["eu_complexity_factors"].append("Emerging technology within EU framework")
    if "cloud" in query.lower() or "platform" in query.lower():
        analysis["eu_complexity_factors"].append("EU cloud service provider management")
    
    # Identify EU member states mentioned
    eu_countries = [
        "germany", "german", "france", "french", "ireland", "irish", "netherlands", "dutch",
        "belgium", "belgian", "italy", "italian", "spain", "spanish", "sweden", "swedish",
        "denmark", "danish", "austria", "austrian", "czech republic", "czech", "hungary", 
        "hungarian", "romania", "romanian", "portugal", "portuguese", "slovakia", "slovak",
        "luxembourg", "finland", "finnish"
    ]
    
    for country in eu_countries:
        if country in query.lower():
            analysis["member_states_involved"].append(country.title())
    
    return analysis

# Example usage and validation for EU-focused testing
if __name__ == "__main__":
    print("EU-Focused Business Test Query Categories:")
    for category in get_eu_business_categories():
        print(f"- {category}: {len(get_eu_business_queries_by_category(category))} queries")
    
    print(f"\nTotal EU business queries: {len(get_all_eu_business_queries())}")
    print(f"Complex EU multi-domain queries: {len(get_complex_eu_multi_domain_queries())}")
    print(f"Intra-EU coordination queries: {len(get_intra_eu_coordination_queries())}")
    
    # Example analysis of EU-specific complexity
    sample_query = get_complex_eu_multi_domain_queries()[0]
    analysis = get_eu_scenario_analysis(sample_query)
    print(f"\nSample EU Query Analysis:")
    print(f"Query: {analysis['query'][:100]}...")
    print(f"GDPR Domains: {len(analysis['gdpr_domains'])}")
    print(f"EU Complexity Factors: {len(analysis['eu_complexity_factors'])}")
    print(f"Member States Involved: {analysis['member_states_involved']}")