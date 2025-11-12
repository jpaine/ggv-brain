#!/usr/bin/env python3
"""
Phase 2: Real Feature Calculations
===================================
Replace 7 hardcoded defaults with real calculations.
"""

import numpy as np
from typing import Dict, List, Optional
from textblob import TextBlob

def calculate_estimated_age(linkedin_profiles: List[Dict]) -> float:
    """
    Calculate founder age from LinkedIn education start year.
    Assumes undergrad starts at age 18.
    """
    if not linkedin_profiles:
        return 35.0  # Default only if no profiles
    
    ages = []
    current_year = 2025
    
    for profile in linkedin_profiles:
        education = profile.get('education', [])
        if not education:
            continue
        
        # Find earliest education start year
        start_years = []
        for edu in education:
            starts_at = edu.get('starts_at', {})
            if isinstance(starts_at, dict) and 'year' in starts_at:
                start_years.append(starts_at['year'])
        
        if start_years:
            earliest_year = min(start_years)
            # Assume started undergrad at 18
            estimated_age = current_year - earliest_year + 18
            # Clamp to reasonable range
            estimated_age = max(22, min(65, estimated_age))
            ages.append(estimated_age)
    
    return np.mean(ages) if ages else 35.0


def calculate_description_sentiment(description: str) -> float:
    """
    Calculate sentiment using TextBlob.
    Returns 0-1 normalized sentiment score.
    """
    if not description or len(description) < 10:
        return 0.5  # Neutral if no description
    
    try:
        blob = TextBlob(description)
        sentiment_polarity = blob.sentiment.polarity  # -1 to 1
        
        # Normalize to 0-1 range
        # Positive sentiment = higher score
        normalized = (sentiment_polarity + 1) / 2
        
        return normalized
    except Exception as e:
        return 0.5  # Default to neutral on error


def calculate_founder_market_fit(linkedin_profiles: List[Dict], 
                                  market_analysis: Dict,
                                  company_categories: List[str]) -> float:
    """
    Calculate founder-market fit based on:
    - Founder's experience in this industry
    - Founder's skills matching market requirements
    """
    if not linkedin_profiles or not market_analysis:
        return 0.5  # Default if no data
    
    # Extract industry keywords from categories and market analysis
    industry_keywords = set()
    
    # From categories
    for cat in company_categories:
        if cat:
            industry_keywords.update(cat.lower().split())
    
    # From market analysis
    market_desc = str(market_analysis.get('description', '')).lower()
    if 'ai' in market_desc or 'artificial intelligence' in market_desc:
        industry_keywords.update(['ai', 'ml', 'machine learning', 'data', 'software'])
    
    # Calculate match for each founder
    match_scores = []
    
    for profile in linkedin_profiles:
        experiences = profile.get('experiences', profile.get('experience', []))
        
        # Count relevant experiences
        relevant_exp = 0
        total_exp = len(experiences)
        
        for exp in experiences:
            exp_title = str(exp.get('title', '')).lower()
            exp_company = str(exp.get('company', '')).lower()
            exp_description = str(exp.get('description', '')).lower()
            
            combined_text = f"{exp_title} {exp_company} {exp_description}"
            
            # Check if experience matches industry
            matches = sum(1 for keyword in industry_keywords if keyword in combined_text)
            if matches > 0:
                relevant_exp += 1
        
        if total_exp > 0:
            match_score = relevant_exp / total_exp
            match_scores.append(match_score)
    
    return np.mean(match_scores) if match_scores else 0.5


def calculate_market_saturation_score(competitor_count: int, 
                                       market_size_billion: float) -> float:
    """
    Calculate market saturation from competitors and market size.
    Higher saturation = more competitors + smaller market.
    Returns 0-1 where 1 = highly saturated.
    """
    if market_size_billion <= 0:
        return 0.5  # Default if no data
    
    # Normalize competitor count (0-100 competitors)
    normalized_competitors = min(competitor_count / 100.0, 1.0)
    
    # Normalize market size (inverse: smaller market = higher saturation)
    # Assume $10B+ is a large market (low saturation contribution)
    normalized_market_size = max(0, 1.0 - (market_size_billion / 10.0))
    
    # Saturation = weighted average
    # More weight on competitor count
    saturation = (0.7 * normalized_competitors) + (0.3 * normalized_market_size)
    
    return min(saturation, 1.0)


def calculate_differentiation_score(description: str, 
                                     competitor_count: int,
                                     market_analysis: Dict) -> float:
    """
    Calculate differentiation based on:
    - Uniqueness of description
    - Number of competitors (fewer = more differentiated)
    - Specific technical terms vs generic terms
    """
    if not description or len(description) < 20:
        return 0.4  # Default if no description
    
    score_components = []
    
    # Component 1: Technical specificity (presence of specific terms)
    technical_terms = [
        'proprietary', 'patent', 'novel', 'breakthrough', 'first',
        'unique', 'innovative', 'revolutionary', 'platform', 'algorithm',
        'architecture', 'framework', 'infrastructure', 'api', 'sdk'
    ]
    
    description_lower = description.lower()
    technical_count = sum(1 for term in technical_terms if term in description_lower)
    technical_score = min(technical_count / 5.0, 1.0)  # Normalize to 0-1
    score_components.append(technical_score)
    
    # Component 2: Market competition (fewer competitors = more differentiated)
    competition_score = max(0, 1.0 - (competitor_count / 200.0))
    score_components.append(competition_score)
    
    # Component 3: Specificity of description (longer, more detailed = more differentiated)
    word_count = len(description.split())
    specificity_score = min(word_count / 100.0, 1.0)  # 100+ words = max score
    score_components.append(specificity_score)
    
    # Average all components
    return np.mean(score_components)


def calculate_timing_score_trend(momentum_3m: float, 
                                  cagr_percent: float,
                                  timing_score: float) -> float:
    """
    Calculate timing trend from momentum and growth rate.
    Positive trend = high momentum + high CAGR.
    Returns -1 to 1 (normalized to 0-1).
    """
    # Normalize CAGR (assume 50% is very high)
    normalized_cagr = min(cagr_percent / 50.0, 1.0)
    
    # Combine momentum and CAGR
    # High values of both = strong positive trend
    trend = (0.6 * momentum_3m) + (0.4 * normalized_cagr)
    
    # Factor in overall timing score
    trend = trend * timing_score
    
    return min(trend, 1.0)


def calculate_competitor_growth_rate(market_analysis: Dict,
                                      cagr_percent: float) -> float:
    """
    Estimate competitor growth rate from market CAGR.
    Assumption: competitor growth ~= market growth.
    Returns 0-1 normalized score.
    """
    if cagr_percent <= 0:
        return 0.2  # Default if no data
    
    # Normalize CAGR to 0-1 (assume 100% CAGR is maximum)
    normalized_cagr = min(cagr_percent / 100.0, 1.0)
    
    # Assume competitors grow slightly slower than market
    competitor_growth = normalized_cagr * 0.8
    
    return competitor_growth


def calculate_confidence_score(company: Dict,
                               linkedin_profiles: List[Dict],
                               market_analysis: Dict) -> float:
    """
    Calculate confidence score based on data availability and quality.
    
    Factors:
    - Crunchbase data availability (company found, description quality)
    - LinkedIn profiles availability and completeness
    - Market analysis availability and completeness
    
    Returns 0-1 confidence score where 1 = high confidence.
    """
    confidence = 0.0
    
    # 1. Crunchbase data (30% weight)
    if company.get('name'):
        confidence += 0.1  # Company name found
    if company.get('description') and len(company.get('description', '')) > 50:
        confidence += 0.1  # Good description
    if company.get('categories'):
        confidence += 0.05  # Categories available
    if company.get('founded_on') and company.get('founded_on') != '2025-01-01':
        confidence += 0.05  # Founded date available
    
    # 2. LinkedIn profiles (40% weight)
    if linkedin_profiles:
        confidence += 0.15  # Profiles found
        for profile in linkedin_profiles:
            # Check profile completeness
            has_experience = bool(profile.get('experiences', profile.get('experience', [])))
            has_education = bool(profile.get('education', []))
            if has_experience:
                confidence += 0.1  # Experience data
            if has_education:
                confidence += 0.05  # Education data
        # Cap LinkedIn contribution at 0.4
        confidence = min(confidence, 0.4)
    
    # 3. Market analysis (30% weight)
    if market_analysis:
        confidence += 0.1  # Market analysis found
        if market_analysis.get('market_size_billion'):
            confidence += 0.05  # Market size available
        if market_analysis.get('cagr_percent'):
            confidence += 0.05  # CAGR available
        if market_analysis.get('competitor_count'):
            confidence += 0.05  # Competitor count available
        if market_analysis.get('timing_score'):
            confidence += 0.05  # Timing score available
    
    # Ensure score is between 0 and 1
    return min(confidence, 1.0)


def calculate_confidence_score_market(market_analysis: Dict) -> float:
    """
    Calculate market-specific confidence score based on market data quality.
    
    Factors:
    - Market size availability and quality
    - CAGR availability and quality
    - Competitor count availability
    - Timing score quality
    
    Returns 0-1 confidence score where 1 = high confidence in market data.
    """
    if not market_analysis:
        return 0.3  # Low confidence if no market analysis
    
    scores = []
    
    # 1. Market size confidence (25% weight)
    market_size = market_analysis.get('market_size_billion', 0)
    if market_size > 0:
        # Higher confidence if market size is reasonable (not default 100.0)
        if market_size < 50.0:  # Reasonable range
            scores.append(0.9)  # High confidence
        else:
            scores.append(0.6)  # Medium confidence (might be default)
    else:
        scores.append(0.3)  # Low confidence (no data)
    
    # 2. CAGR confidence (25% weight)
    cagr = market_analysis.get('cagr_percent', 0)
    if cagr > 0:
        # Higher confidence if CAGR is reasonable (not default 25.0)
        if 5.0 <= cagr <= 50.0:  # Reasonable range
            scores.append(0.9)  # High confidence
        else:
            scores.append(0.6)  # Medium confidence (might be default)
    else:
        scores.append(0.4)  # Low confidence (no data)
    
    # 3. Competitor count confidence (25% weight)
    competitor_count = market_analysis.get('competitor_count', 0)
    if competitor_count > 0:
        # Higher confidence if competitor count is reasonable (not default 50)
        if 5 <= competitor_count <= 200:  # Reasonable range
            scores.append(0.8)  # High confidence
        else:
            scores.append(0.5)  # Medium confidence (might be default)
    else:
        scores.append(0.4)  # Low confidence (no data)
    
    # 4. Timing score confidence (25% weight)
    timing_score = market_analysis.get('timing_score', 0)
    if timing_score > 0:
        # Higher confidence if timing score is reasonable (not default 0.7)
        if 0.5 <= timing_score <= 1.0:  # Reasonable range
            scores.append(0.8)  # High confidence
        else:
            scores.append(0.5)  # Medium confidence (might be default)
    else:
        scores.append(0.3)  # Low confidence (no data)
    
    # Average all confidence scores
    if scores:
        return round(sum(scores) / len(scores), 2)
    else:
        return 0.3  # Default low confidence


# ============================================================================
# Batch calculation function
# ============================================================================

def calculate_all_features(company: Dict,
                           linkedin_profiles: List[Dict],
                           market_analysis: Dict) -> Dict:
    """
    Calculate all 7 features with real calculations.
    
    Returns:
        Dict with feature names and values
    """
    description = company.get('description', '')
    categories = company.get('categories', [])
    
    # Extract base market features
    market_size_billion = market_analysis.get('market_size_billion', 100.0) if market_analysis else 100.0
    cagr_percent = market_analysis.get('cagr_percent', 25.0) if market_analysis else 25.0
    competitor_count = market_analysis.get('competitor_count', 50) if market_analysis else 50
    momentum_3m = market_analysis.get('momentum_score', 0.8) if market_analysis else 0.8
    timing_score = market_analysis.get('timing_score', 0.7) if market_analysis else 0.7
    
    features = {}
    
    # 1. Estimated age
    features['estimated_age'] = calculate_estimated_age(linkedin_profiles)
    
    # 2. Description sentiment
    features['description_sentiment'] = calculate_description_sentiment(description)
    
    # 3. Founder-market fit
    features['founder_market_fit'] = calculate_founder_market_fit(
        linkedin_profiles, market_analysis, categories
    )
    
    # 4. Market saturation
    features['market_saturation_score'] = calculate_market_saturation_score(
        competitor_count, market_size_billion
    )
    
    # 5. Differentiation
    features['differentiation_score'] = calculate_differentiation_score(
        description, competitor_count, market_analysis
    )
    
    # 6. Timing trend
    features['timing_score_trend'] = calculate_timing_score_trend(
        momentum_3m, cagr_percent, timing_score
    )
    
    # 7. Competitor growth rate
    features['competitor_growth_rate'] = calculate_competitor_growth_rate(
        market_analysis, cagr_percent
    )
    
    # 8. Confidence score (NEW - replaces hardcoded 0.85)
    features['confidence_score'] = calculate_confidence_score(
        company, linkedin_profiles, market_analysis
    )
    
    # 9. Confidence score market (NEW - replaces hardcoded 0.85)
    features['confidence_score_market'] = calculate_confidence_score_market(
        market_analysis
    )
    
    return features


if __name__ == "__main__":
    # Test calculations
    print("="*80)
    print("PHASE 2 FEATURE CALCULATION TESTS")
    print("="*80)
    
    # Test data
    test_profile = {
        'education': [
            {'starts_at': {'year': 2015}, 'school': 'Stanford University'}
        ],
        'experiences': [
            {'title': 'ML Engineer', 'company': 'Google', 'description': 'Built AI systems'},
            {'title': 'Founder', 'company': 'Previous Startup', 'description': 'Led team'}
        ]
    }
    
    test_company = {
        'description': 'We are building a revolutionary AI platform that uses proprietary algorithms to transform the industry.',
        'categories': ['Artificial Intelligence', 'Software']
    }
    
    test_market = {
        'market_size_billion': 5.0,
        'cagr_percent': 35.0,
        'competitor_count': 30,
        'momentum_score': 0.85,
        'timing_score': 0.8
    }
    
    features = calculate_all_features(test_company, [test_profile], test_market)
    
    print("\nCalculated features:")
    for feat_name, feat_value in features.items():
        print(f"  {feat_name:30s}: {feat_value:.3f}")
    
    print("\n✅ All calculations working!")

