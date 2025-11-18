# Crunchbase AI Categories Taxonomy

**Purpose:** Comprehensive list of categories to filter for AI-related companies in Crunchbase searches.

---

## Current Implementation

**Status:** Currently, the workflow does NOT filter by categories - it searches all companies and then filters for USA + founders.

**Opportunity:** Add category filtering to focus specifically on AI/ML companies and reduce noise.

---

## Core AI/ML Technology Categories

### Primary AI Categories (High Priority)

These are the core categories that should definitely be included:

1. **Artificial Intelligence** (`artificial_intelligence`)
   - General AI companies
   - AI platforms and infrastructure
   - AI consulting and services

2. **Machine Learning** (`machine_learning`)
   - ML algorithms and models
   - ML platforms and tools
   - ML consulting

3. **Deep Learning** (`deep_learning`)
   - Neural networks
   - Deep learning frameworks
   - Deep learning applications

4. **Natural Language Processing** (`natural_language_processing` or `nlp`)
   - Text analysis
   - Language models
   - Chatbots and conversational AI
   - Sentiment analysis

5. **Computer Vision** (`computer_vision`)
   - Image recognition
   - Video analysis
   - Facial recognition
   - Medical imaging

6. **Robotics** (`robotics`)
   - Autonomous robots
   - Industrial automation
   - Service robots
   - Robotic process automation (RPA)

7. **Generative AI** (`generative_ai` or `genai`)
   - Large Language Models (LLMs)
   - Image generation
   - Content creation AI
   - AI art and media

8. **Predictive Analytics** (`predictive_analytics`)
   - Forecasting models
   - Predictive maintenance
   - Risk prediction

---

## AI-Adjacent Technology Categories

### Data & Analytics (Medium Priority)

9. **Big Data** (`big_data`)
   - Data processing platforms
   - Data warehouses
   - Data lakes

10. **Data Analytics** (`data_analytics`)
    - Business intelligence
    - Data visualization
    - Analytics platforms

11. **Data Science** (`data_science`)
    - Data science platforms
    - Data science consulting

12. **Business Intelligence** (`business_intelligence`)
    - BI platforms
    - Reporting tools

### Automation & Intelligence (Medium Priority)

13. **Automation** (`automation`)
    - Process automation
    - Workflow automation
    - Intelligent automation

14. **Intelligent Systems** (`intelligent_systems`)
    - Smart systems
    - Cognitive computing

15. **Cognitive Computing** (`cognitive_computing`)
    - Cognitive platforms
    - AI reasoning systems

---

## Industry Verticals Using AI (Lower Priority - Optional)

These are industries that heavily use AI but may not be pure AI companies:

### Healthcare AI
16. **Health Care** (`health_care`) - Filter for AI subcategories
17. **Medical Devices** (`medical_devices`) - AI-powered devices
18. **Biotechnology** (`biotechnology`) - AI in biotech

### FinTech AI
19. **Financial Services** (`financial_services`) - AI in finance
20. **FinTech** (`fintech`) - AI-powered financial tech
21. **Insurance** (`insurance`) - InsurTech with AI

### Enterprise AI
22. **Enterprise Software** (`enterprise_software`) - AI features
23. **SaaS** (`saas`) - AI-powered SaaS
24. **CRM** (`crm`) - AI-enhanced CRM

### Autonomous Systems
25. **Autonomous Vehicles** (`autonomous_vehicles`)
26. **Drones** (`drones`) - AI-powered drones
27. **Autonomous Systems** (`autonomous_systems`)

### Other AI Applications
28. **E-commerce** (`e_commerce`) - AI recommendations
29. **EdTech** (`edtech`) - AI in education
30. **Legal Tech** (`legal_tech`) - AI in legal
31. **HR Tech** (`hr_tech`) - AI in HR
32. **Marketing Tech** (`marketing_tech`) - AI marketing

---

## Recommended Category Filter Strategy

### Option 1: Core AI Only (Recommended for Start)

**Focus:** Pure AI/ML companies

```python
AI_CATEGORIES = [
    "artificial_intelligence",
    "machine_learning", 
    "deep_learning",
    "natural_language_processing",
    "computer_vision",
    "robotics",
    "generative_ai"
]
```

**Pros:**
- High precision (fewer false positives)
- Clear AI focus
- Easier to manage

**Cons:**
- May miss AI companies in other categories
- Some companies may use AI but not be categorized as AI

---

### Option 2: Core AI + Adjacent (Balanced)

**Focus:** AI companies + data/analytics companies using AI

```python
AI_CATEGORIES = [
    # Core AI
    "artificial_intelligence",
    "machine_learning",
    "deep_learning", 
    "natural_language_processing",
    "computer_vision",
    "robotics",
    "generative_ai",
    # Adjacent
    "big_data",
    "data_analytics",
    "data_science",
    "automation",
    "predictive_analytics"
]
```

**Pros:**
- Broader coverage
- Catches AI-adjacent companies
- Still focused

**Cons:**
- Some non-AI companies may slip through
- Need to verify AI usage

---

### Option 3: Comprehensive (Maximum Coverage)

**Focus:** All AI-related categories + industry verticals

```python
AI_CATEGORIES = [
    # Core AI (all from Option 1)
    # Adjacent (all from Option 2)
    # Plus industry verticals:
    "autonomous_vehicles",
    "drones",
    "health_care",  # With AI subcategory check
    "fintech",      # With AI subcategory check
    # etc.
]
```

**Pros:**
- Maximum coverage
- Catches AI in all industries

**Cons:**
- Many false positives
- Need additional filtering
- More complex

---

## Implementation in Crunchbase API

### Current Search (No Category Filter)

```python
payload = {
    "query": [
        {
            "type": "predicate",
            "field_id": "founded_on",
            "operator_id": "gte",
            "values": [start_date]
        },
        {
            "type": "predicate",
            "field_id": "founded_on",
            "operator_id": "lte",
            "values": [end_date]
        }
    ]
}
```

### With Category Filter (Recommended)

```python
payload = {
    "query": [
        {
            "type": "predicate",
            "field_id": "founded_on",
            "operator_id": "gte",
            "values": [start_date]
        },
        {
            "type": "predicate",
            "field_id": "founded_on",
            "operator_id": "lte",
            "values": [end_date]
        },
        {
            "type": "predicate",
            "field_id": "categories",
            "operator_id": "contains_any",  # OR logic - company has ANY of these
            "values": [
                "artificial_intelligence",
                "machine_learning",
                "deep_learning",
                "natural_language_processing",
                "computer_vision",
                "robotics",
                "generative_ai"
            ]
        }
    ]
}
```

### Alternative: Category Groups

Crunchbase also has `category_groups` which are higher-level groupings:

```python
{
    "type": "predicate",
    "field_id": "category_groups",
    "operator_id": "contains_any",
    "values": [
        "Artificial Intelligence",
        "Machine Learning",
        "AI/ML"
    ]
}
```

**Note:** Category groups use display names (with spaces), categories use slugs (with underscores).

---

## Category Format in Crunchbase

### Category Slugs (API Format)
- Use underscores: `artificial_intelligence`
- Lowercase: `machine_learning`
- No spaces: `natural_language_processing`

### Category Display Names (Human Readable)
- Use spaces: `Artificial Intelligence`
- Title case: `Machine Learning`
- Used in `category_groups`

---

## Recommended Implementation

### Phase 1: Start with Core AI Categories

**Recommended categories to start with:**

```python
CORE_AI_CATEGORIES = [
    "artificial_intelligence",
    "machine_learning",
    "deep_learning",
    "natural_language_processing",
    "computer_vision",
    "robotics",
    "generative_ai"
]
```

**Why these 7?**
- High precision (pure AI companies)
- Covers all major AI subfields
- Easy to manage and verify
- Reduces noise significantly

### Phase 2: Add Adjacent Categories (If Needed)

If Phase 1 doesn't capture enough companies, add:

```python
ADJACENT_CATEGORIES = [
    "big_data",
    "data_analytics",
    "data_science",
    "automation",
    "predictive_analytics"
]
```

### Phase 3: Industry Verticals (Optional)

Only if you want to catch AI companies in specific industries:

```python
INDUSTRY_AI_CATEGORIES = [
    "autonomous_vehicles",
    "drones",
    # Add others as needed
]
```

---

## Testing Category Filters

### How to Test

1. **Search with categories** - See how many companies match
2. **Compare to current results** - How many are we missing?
3. **Check false positives** - Are non-AI companies included?
4. **Verify AI usage** - Do companies actually use AI?

### Expected Impact

**Current:** ~31 companies found (all categories, USA + founders)  
**With Core AI filter:** ~10-15 companies (estimated)  
**Precision:** Higher (fewer false positives)  
**Recall:** Lower (may miss some AI companies)

---

## Category Validation

### How to Verify Categories

After fetching companies, check their categories:

```python
categories = company.get('categories', [])
category_values = [cat.get('value', '') if isinstance(cat, dict) else str(cat) for cat in categories]

# Check if any match our AI categories
is_ai_company = any(
    cat.lower() in CORE_AI_CATEGORIES 
    for cat in category_values
)
```

---

## Next Steps

1. **Implement Core AI filter** - Add category filter to search
2. **Test on recent data** - Compare results with/without filter
3. **Monitor precision** - Check if companies are actually AI-focused
4. **Adjust as needed** - Add/remove categories based on results

---

## References

- Crunchbase API Documentation: https://data.crunchbase.com/docs
- Category List: Check Crunchbase API explorer
- Current Implementation: `workflow_1_crunchbase_daily_monitor.py` line 163-184

---

**Last Updated:** November 18, 2025  
**Status:** Ready for Implementation

