"""
Sample Compliance Queries for UI Demo and User Guidance

This module provides curated, user-friendly sample queries organized by business
scenarios. These samples serve dual purposes:

1. **User Education**: Help users understand what types of questions work well
2. **System Demonstration**: Showcase the sophisticated multi-agent capabilities

The queries are designed to be:
- Realistic business scenarios that EU companies commonly face
- Concise enough for easy reading but detailed enough for meaningful analysis
- Representative of different compliance domains (GDPR, Polish law, security)
- Progressively complex to demonstrate system sophistication

Design Philosophy:
Rather than overwhelming users with blank forms, we provide starting points
that they can modify or use directly. This reduces friction and helps users
learn effective query formulation patterns.
"""

# Sample queries organized by common business scenarios
SAMPLE_QUERY_CATEGORIES = {
    "employee_monitoring_hr": {
        "title": "👥 Employee Monitoring & HR",
        "description": "Workplace surveillance, HR data processing, and employee rights",
        "icon": "👥",
        "queries": [
            {
                "title": "Employee Productivity Monitoring",
                "query": "We want to implement software that tracks employee keystrokes and screen time in our Warsaw office. The monitoring data will be processed by a German IT service provider. What GDPR compliance steps do we need for employee surveillance across EU borders?"
            },
            {
                "title": "Background Check Processing", 
                "query": "Our Polish subsidiary conducts background checks on job candidates using a Dutch HR platform that processes data across multiple EU countries. How do we structure consent and data processing agreements under GDPR Articles 6 and 9?"
            },
            {
                "title": "Remote Work BYOD Policy",
                "query": "Polish remote workers want to use personal devices to access customer data through our CRM system hosted by a French cloud provider. How do we update our BYOD policy to meet GDPR security requirements and Polish labor law?"
            },
            {
                "title": "Cross-Border HR Analytics",
                "query": "We need to transfer employee performance data from our Polish office to Irish headquarters for EU-wide talent management analytics. What internal approvals and GDPR documentation do we need for this intra-EU processing?"
            }
        ]
    },
    
    "customer_data_marketing": {
        "title": "🎯 Customer Data & Marketing", 
        "description": "Customer analytics, marketing automation, and consent management",
        "icon": "🎯",
        "queries": [
            {
                "title": "AI-Powered Personalization",
                "query": "We're launching an AI recommendation engine for our Polish e-commerce site that analyzes customer behavior. The AI system is developed by an Italian company and hosted in Ireland. What GDPR Article 22 compliance steps do we need for automated decision-making?"
            },
            {
                "title": "Cross-Border Marketing Analytics",
                "query": "Our marketing team wants to create customer personas using Polish customer data, then share insights with our advertising agency in Berlin and analytics partner in Amsterdam. What consent mechanisms and data processing agreements do we need?"
            },
            {
                "title": "Cookie Consent Management", 
                "query": "We need cookie consent management on our Polish website that integrates with our Spanish marketing automation platform serving all EU markets. The system handles granular consent for analytics and advertising. What technical and legal requirements must we meet?"
            },
            {
                "title": "Multi-Country Loyalty Program",
                "query": "Our Polish loyalty program collects purchase and location data. We want to expand to include German hotels and French restaurants for cross-border reward redemption. How do we structure consent and data sharing agreements for this multi-party EU processing?"
            }
        ]
    },
    
    "it_security_cloud": {
        "title": "🔒 IT Security & Cloud Services",
        "description": "Cloud migrations, security monitoring, and technical safeguards", 
        "icon": "🔒",
        "queries": [
            {
                "title": "Cloud Migration Security",
                "query": "We're migrating our Polish customer database to German cloud services with infrastructure across Ireland, Netherlands, and Germany. How do we ensure the migration meets GDPR technical and organizational measures while maintaining our internal security requirements?"
            },
            {
                "title": "Security Incident Investigation",
                "query": "Our IT security team detected suspicious activity in Warsaw and needs to analyze employee internet logs and email metadata for forensic investigation. Some data was accessed from Prague and Vienna offices. What are our GDPR obligations for this cross-border EU investigation?"
            },
            {
                "title": "Behavioral Analytics Implementation",
                "query": "We want to implement behavioral analytics software from a Swedish company that monitors user access patterns and flags anomalous behavior using biometric badge data. How do we balance security needs with GDPR privacy requirements for this EU security solution?"
            },
            {
                "title": "Single Sign-On Deployment",
                "query": "We're implementing SSO across European operations, meaning Polish employee authentication data will be processed by our Irish identity management system managed by a Belgian IT services company. What intra-EU data flow protections do we need?"
            }
        ]
    },
    
    "vendor_third_party": {
        "title": "🤝 Vendor Management & Third Parties",
        "description": "Service provider relationships, due diligence, and outsourcing",
        "icon": "🤝", 
        "queries": [
            {
                "title": "Multi-Country Customer Support",
                "query": "We're implementing a customer support platform where Polish customers' requests might be handled by agents in Romania, Hungary, or Portugal depending on availability. The German-provided platform includes call recording and AI-powered routing. What multi-jurisdictional EU compliance framework do we need?"
            },
            {
                "title": "Expense Management SaaS",
                "query": "Our Polish accounting team uses a Finnish expense management platform processing employee credit card transactions and receipt images. The vendor's servers are distributed across Stockholm, Frankfurt, and Dublin. How do we ensure vendor contracts meet GDPR requirements?"
            },
            {
                "title": "Due Diligence Data Processing",
                "query": "We're conducting due diligence for acquiring a Czech competitor, requiring analysis of their customer and employee databases by our German and French consultants. What legal basis and safeguards do we need for this temporary cross-border EU processing?"
            },
            {
                "title": "Industrial IoT Data Sharing",
                "query": "Our Polish manufacturing facility wants to share production quality data with Czech and Slovak suppliers through an Austrian Industry 4.0 platform. This includes timestamps and employee IDs. How do we structure GDPR compliance for this multi-country EU industrial IoT scenario?"
            }
        ]
    },
    
    "cross_border_transfers": {
        "title": "🌍 Cross-Border Data Transfers",
        "description": "International data flows, adequacy decisions, and transfer mechanisms",
        "icon": "🌍",
        "queries": [
            {
                "title": "EU-Wide Data Lake Implementation", 
                "query": "We're establishing a EU-wide customer data lake consolidating information from Poland, Germany, Czech Republic, and Austria. The data lake will be managed by a Dutch company and hosted in Irish data centers. What governance framework do we need?"
            },
            {
                "title": "Blockchain Supply Chain Tracking",
                "query": "Our Polish subsidiary wants to implement blockchain supply chain tracking involving partners in Germany, Italy, Hungary, and Romania. The system would record personal data about drivers and warehouse workers. How do we structure GDPR compliance for multi-country blockchain?"
            },
            {
                "title": "Multi-Authority Coordination",
                "query": "Our Polish DPO needs to coordinate with supervisory authorities in Germany, France, and Ireland regarding cross-border processing using a Belgian cloud provider. What are our obligations under GDPR's one-stop-shop mechanism?"
            },
            {
                "title": "Open Banking API Integration",
                "query": "We're implementing open banking APIs in Poland allowing third-party EU financial apps to access customer account data. Our risk management requires monitoring data flows to German, French, and Dutch fintech partners. What consent and governance procedures do we need under PSD2 and GDPR?"
            }
        ]
    },
    
    "incident_breach_response": {
        "title": "🚨 Incident Response & Breaches",
        "description": "Data breaches, notification requirements, and crisis management",
        "icon": "🚨",
        "queries": [
            {
                "title": "Ransomware Attack Response",
                "query": "Our Polish office experienced ransomware that encrypted customer databases. We have backups with a Luxembourg provider, but attackers claim to have exfiltrated data. We're 18 hours post-incident and need to coordinate technical recovery with GDPR notification obligations."
            },
            {
                "title": "Employee Data Exfiltration",
                "query": "A former employee in Krakow downloaded customer contact lists to their personal device and may have shared this with a Czech competitor. Our incident response was triggered, but what are our specific breach notification obligations under Polish law and GDPR Articles 33 and 34?"
            },
            {
                "title": "Unauthorized Data Sharing Discovery",
                "query": "We discovered our payment processor in Poland has been sharing transaction data with analytics companies in Ireland and Netherlands for market research without customer consent. We need to assess liability, notification obligations, and remediation steps."
            },
            {
                "title": "Multi-Country Crisis Response",
                "query": "Our crisis management team needs incident response procedures for breaches affecting multiple EU countries simultaneously. We have operations in Poland, Germany, Netherlands, and Belgium with different supervisory authority requirements. How do we create coordinated GDPR compliance procedures?"
            }
        ]
    }
}

def get_sample_categories():
    """
    Return all available sample query categories for UI display.
    
    This function provides the structure needed for creating tabs or dropdown
    selections in the user interface. Each category includes metadata for
    proper display and user guidance.
    
    Returns:
        Dict: Categories with titles, descriptions, icons, and query lists
    """
    return SAMPLE_QUERY_CATEGORIES

def get_category_queries(category_key):
    """
    Get all sample queries for a specific category.
    
    This function enables the UI to display category-specific query options
    when users select a particular business scenario tab.
    
    Args:
        category_key: The key identifier for the desired category
        
    Returns:
        List: Sample queries for the specified category, or empty list if not found
    """
    category = SAMPLE_QUERY_CATEGORIES.get(category_key, {})
    return category.get("queries", [])

def get_all_sample_queries():
    """
    Return all sample queries across all categories for comprehensive testing.
    
    This function is useful for system validation or when you want to provide
    a complete list of available samples for advanced users.
    
    Returns:
        List: All sample queries with category metadata included
    """
    all_queries = []
    for category_key, category_data in SAMPLE_QUERY_CATEGORIES.items():
        for query in category_data["queries"]:
            query_with_metadata = query.copy()
            query_with_metadata["category"] = category_key
            query_with_metadata["category_title"] = category_data["title"]
            all_queries.append(query_with_metadata)
    
    return all_queries

def search_sample_queries(search_term):
    """
    Search sample queries by title or content for enhanced discoverability.
    
    This function enables users to find relevant samples quickly without
    browsing through all categories manually.
    
    Args:
        search_term: Text to search for in query titles and content
        
    Returns:
        List: Matching sample queries with relevance scoring
    """
    search_term_lower = search_term.lower()
    matching_queries = []
    
    for category_key, category_data in SAMPLE_QUERY_CATEGORIES.items():
        for query in category_data["queries"]:
            # Check if search term appears in title or query text
            title_match = search_term_lower in query["title"].lower()
            query_match = search_term_lower in query["query"].lower()
            category_match = search_term_lower in category_data["title"].lower()
            
            if title_match or query_match or category_match:
                result = query.copy()
                result["category"] = category_key
                result["category_title"] = category_data["title"]
                
                # Simple relevance scoring
                relevance_score = 0
                if title_match:
                    relevance_score += 3
                if query_match:
                    relevance_score += 2
                if category_match:
                    relevance_score += 1
                
                result["relevance_score"] = relevance_score
                matching_queries.append(result)
    
    # Sort by relevance score (highest first)
    matching_queries.sort(key=lambda x: x["relevance_score"], reverse=True)
    return matching_queries

def get_category_stats():
    """
    Return statistics about sample queries for administrative purposes.
    
    This function provides insight into the coverage and distribution of
    sample queries across different business scenarios.
    
    Returns:
        Dict: Statistics including total queries, category distribution, etc.
    """
    stats = {
        "total_categories": len(SAMPLE_QUERY_CATEGORIES),
        "total_queries": 0,
        "category_distribution": {},
        "average_queries_per_category": 0
    }
    
    for category_key, category_data in SAMPLE_QUERY_CATEGORIES.items():
        query_count = len(category_data["queries"])
        stats["total_queries"] += query_count
        stats["category_distribution"][category_data["title"]] = query_count
    
    if stats["total_categories"] > 0:
        stats["average_queries_per_category"] = round(
            stats["total_queries"] / stats["total_categories"], 1
        )
    
    return stats

def validate_sample_queries():
    """
    Validate the structure and content of sample queries for system integrity.
    
    This function helps ensure that all sample queries meet the expected
    format and quality standards for reliable system operation.
    
    Returns:
        Dict: Validation results with any issues identified
    """
    validation_results = {
        "valid": True,
        "issues": [],
        "warnings": []
    }
    
    required_fields = ["title", "query"]
    
    for category_key, category_data in SAMPLE_QUERY_CATEGORIES.items():
        # Check category structure
        if "title" not in category_data:
            validation_results["issues"].append(f"Category {category_key} missing title")
            validation_results["valid"] = False
        
        if "queries" not in category_data:
            validation_results["issues"].append(f"Category {category_key} missing queries")
            validation_results["valid"] = False
            continue
        
        # Check individual queries
        for i, query in enumerate(category_data["queries"]):
            for field in required_fields:
                if field not in query:
                    validation_results["issues"].append(
                        f"Category {category_key}, query {i}: missing {field}"
                    )
                    validation_results["valid"] = False
            
            # Quality checks
            if len(query.get("query", "")) < 100:
                validation_results["warnings"].append(
                    f"Category {category_key}, query {i}: query text seems short"
                )
            
            if len(query.get("title", "")) < 10:
                validation_results["warnings"].append(
                    f"Category {category_key}, query {i}: title seems short"
                )
    
    return validation_results

# Example usage and system validation
if __name__ == "__main__":
    print("Sample Query System Overview:")
    print("=" * 50)
    
    stats = get_category_stats()
    print(f"Total Categories: {stats['total_categories']}")
    print(f"Total Sample Queries: {stats['total_queries']}")
    print(f"Average per Category: {stats['average_queries_per_category']}")
    
    print("\nCategory Distribution:")
    for category, count in stats["category_distribution"].items():
        print(f"  {category}: {count} queries")
    
    print("\nValidation Results:")
    validation = validate_sample_queries()
    if validation["valid"]:
        print("✅ All sample queries are valid")
    else:
        print("❌ Validation issues found:")
        for issue in validation["issues"]:
            print(f"  - {issue}")
    
    if validation["warnings"]:
        print("⚠️ Warnings:")
        for warning in validation["warnings"]:
            print(f"  - {warning}")
    
    print("\nSearch Test:")
    gdpr_queries = search_sample_queries("GDPR")
    print(f"Found {len(gdpr_queries)} queries mentioning 'GDPR'")
    
    print("\nSample Query Example:")
    first_category = list(SAMPLE_QUERY_CATEGORIES.keys())[0]
    first_query = get_category_queries(first_category)[0]
    print(f"Title: {first_query['title']}")
    print(f"Query: {first_query['query'][:100]}...")