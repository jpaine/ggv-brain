#!/usr/bin/env python3
"""
Phase 5 Week 3: Enhanced Pitch Quality Features
===============================================

Extracts 4 new features from company description/pitch:
1. pitch_clarity_score - How clearly problem/solution is articulated (0-1)
2. pitch_specificity_score - Concrete numbers vs vague claims (0-1)
3. vision_realism_score - Realistic vs overpromising (0-1)
4. narrative_coherence_score - Problem-solution-market alignment (0-1)

Data Source: Company description (fully available, no API calls needed)
"""

from typing import Dict, Optional
import re


def calculate_readability_score(text: str) -> float:
    """
    Calculate readability score using simple heuristics.
    Returns 0-1 (higher = more readable).
    """
    if not text or len(text.strip()) == 0:
        return 0.0
    
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        return 0.0
    
    words = text.split()
    avg_sentence_length = len(words) / len(sentences) if sentences else 0
    avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
    
    # Ideal: 15-20 words per sentence, 4-5 chars per word
    sentence_score = 1.0 - abs(avg_sentence_length - 17.5) / 30.0  # Penalize if too far from 17.5
    word_score = 1.0 - abs(avg_word_length - 4.5) / 5.0  # Penalize if too far from 4.5
    
    readability = (max(0.0, sentence_score) + max(0.0, word_score)) / 2.0
    return min(1.0, max(0.0, readability))


def extract_pitch_clarity_score(company_description: str) -> float:
    """
    Calculate how clearly the problem and solution are articulated.
    
    Factors:
    - Sentence structure and readability
    - Presence of clear problem statement
    - Presence of clear solution statement
    - Use of active voice vs passive voice
    """
    if not company_description:
        return 0.5  # Neutral default
    
    description_lower = company_description.lower()
    
    # 1. Readability score (40% weight)
    readability = calculate_readability_score(company_description)
    
    # 2. Problem statement clarity (30% weight)
    problem_keywords = [
        'problem', 'challenge', 'pain', 'issue', 'difficulty', 'struggle',
        'inefficient', 'ineffective', 'lack', 'missing', 'gap', 'barrier'
    ]
    has_problem = any(kw in description_lower for kw in problem_keywords)
    problem_clarity = 1.0 if has_problem else 0.3
    
    # 3. Solution statement clarity (30% weight)
    solution_keywords = [
        'solution', 'solve', 'address', 'provide', 'enable', 'help',
        'platform', 'tool', 'system', 'software', 'service', 'product'
    ]
    has_solution = any(kw in description_lower for kw in solution_keywords)
    solution_clarity = 1.0 if has_solution else 0.3
    
    # Weighted average
    clarity_score = (readability * 0.4) + (problem_clarity * 0.3) + (solution_clarity * 0.3)
    
    return min(1.0, max(0.0, clarity_score))


def extract_pitch_specificity_score(company_description: str) -> float:
    """
    Calculate how specific vs vague the pitch is.
    
    Factors:
    - Presence of numbers, metrics, percentages
    - Use of concrete examples vs abstract concepts
    - Specific technologies mentioned
    - Specific industries/customers mentioned
    """
    if not company_description:
        return 0.0
    
    description_lower = company_description.lower()
    
    # 1. Numbers and metrics (40% weight)
    # Find numbers with units (%, $, M, B, K, etc.)
    number_patterns = [
        r'\d+%',  # Percentages
        r'\$\d+[MBK]?',  # Dollar amounts
        r'\d+[MBK]',  # Millions/Billions/Thousands
        r'\d+\.\d+',  # Decimals
        r'\d+x',  # Multipliers
        r'\d+[,\d]+',  # Large numbers with commas
    ]
    
    numbers_found = []
    for pattern in number_patterns:
        numbers_found.extend(re.findall(pattern, company_description, re.IGNORECASE))
    
    # Count unique number patterns
    unique_numbers = len(set(numbers_found))
    number_score = min(unique_numbers / 5.0, 1.0)  # 5+ numbers = specific
    
    # 2. Specific technologies (20% weight)
    tech_keywords = [
        'ai', 'ml', 'machine learning', 'deep learning', 'neural network',
        'blockchain', 'crypto', 'api', 'saas', 'cloud', 'aws', 'azure',
        'python', 'javascript', 'react', 'node', 'kubernetes', 'docker'
    ]
    tech_count = sum(1 for kw in tech_keywords if kw in description_lower)
    tech_score = min(tech_count / 3.0, 1.0)  # 3+ technologies = specific
    
    # 3. Specific industries/customers (20% weight)
    industry_keywords = [
        'healthcare', 'finance', 'retail', 'manufacturing', 'education',
        'enterprise', 'sme', 'b2b', 'b2c', 'consumer', 'enterprise'
    ]
    industry_count = sum(1 for kw in industry_keywords if kw in description_lower)
    industry_score = min(industry_count / 2.0, 1.0)  # 2+ industries = specific
    
    # 4. Concrete examples (20% weight)
    example_keywords = [
        'for example', 'such as', 'including', 'like', 'e.g.', 'i.e.',
        'case study', 'use case', 'scenario'
    ]
    has_examples = any(kw in description_lower for kw in example_keywords)
    example_score = 1.0 if has_examples else 0.3
    
    # Weighted average
    specificity_score = (
        number_score * 0.4 +
        tech_score * 0.2 +
        industry_score * 0.2 +
        example_score * 0.2
    )
    
    return min(1.0, max(0.0, specificity_score))


def extract_vision_realism_score(company_description: str) -> float:
    """
    Calculate how realistic vs overpromising the vision is.
    
    Factors:
    - Avoids excessive hyperbole
    - Realistic timelines mentioned
    - Acknowledges challenges/limitations
    - Grounded in current capabilities
    """
    if not company_description:
        return 0.5  # Neutral default
    
    description_lower = company_description.lower()
    
    # 1. Hyperbole detection (40% weight) - negative signal
    hyperbolic_words = [
        'revolutionary', 'disruptive', 'game-changing', 'unprecedented',
        'breakthrough', 'revolution', 'transform', 'change the world',
        'perfect', 'ultimate', 'best ever', 'never before', 'completely new',
        'totally', 'absolutely', 'incredible', 'amazing', 'fantastic'
    ]
    
    hyperbole_count = sum(1 for word in hyperbolic_words if word in description_lower)
    hyperbole_penalty = min(hyperbole_count / 3.0, 1.0)  # 3+ hyperbole words = unrealistic
    
    # 2. Realistic qualifiers (30% weight) - positive signal
    realistic_keywords = [
        'currently', 'now', 'today', 'recently', 'initial', 'first',
        'pilot', 'beta', 'early', 'phase', 'version', 'stage',
        'aim', 'goal', 'target', 'plan', 'strategy'
    ]
    has_realistic_qualifiers = any(kw in description_lower for kw in realistic_keywords)
    realistic_bonus = 0.3 if has_realistic_qualifiers else 0.0
    
    # 3. Acknowledges challenges (20% weight) - positive signal
    challenge_keywords = [
        'challenge', 'difficulty', 'limitation', 'constraint', 'barrier',
        'however', 'although', 'while', 'despite', 'challenging'
    ]
    acknowledges_challenges = any(kw in description_lower for kw in challenge_keywords)
    challenge_bonus = 0.2 if acknowledges_challenges else 0.0
    
    # 4. Grounded in current tech (10% weight)
    current_tech_keywords = [
        'api', 'integration', 'existing', 'current', 'standard',
        'compatible', 'works with', 'built on', 'using'
    ]
    grounded_in_current = any(kw in description_lower for kw in current_tech_keywords)
    grounded_bonus = 0.1 if grounded_in_current else 0.0
    
    # Calculate score: start at 1.0, subtract hyperbole penalty, add bonuses
    realism_score = 1.0 - hyperbole_penalty + realistic_bonus + challenge_bonus + grounded_bonus
    
    return min(1.0, max(0.0, realism_score))


def extract_narrative_coherence_score(company_description: str) -> float:
    """
    Calculate how well problem-solution-market align in the narrative.
    
    Factors:
    - Problem clearly stated
    - Solution addresses the problem
    - Market/opportunity mentioned
    - Logical flow from problem → solution → market
    """
    if not company_description:
        return 0.0
    
    description_lower = company_description.lower()
    
    # 1. Problem statement present (25% weight)
    problem_keywords = [
        'problem', 'challenge', 'pain', 'issue', 'difficulty', 'struggle',
        'inefficient', 'ineffective', 'lack', 'missing', 'gap', 'barrier',
        'frustrated', 'struggling', 'difficult', 'hard', 'complex'
    ]
    has_problem = any(kw in description_lower for kw in problem_keywords)
    problem_score = 1.0 if has_problem else 0.0
    
    # 2. Solution statement present (25% weight)
    solution_keywords = [
        'solution', 'solve', 'address', 'provide', 'enable', 'help',
        'platform', 'tool', 'system', 'software', 'service', 'product',
        'offers', 'delivers', 'creates', 'builds', 'develops'
    ]
    has_solution = any(kw in description_lower for kw in solution_keywords)
    solution_score = 1.0 if has_solution else 0.0
    
    # 3. Market/opportunity mentioned (25% weight)
    market_keywords = [
        'market', 'customers', 'users', 'clients', 'businesses',
        'industry', 'sector', 'billion', 'million', 'opportunity',
        'demand', 'need', 'growth', 'size', 'tam', 'sam'
    ]
    has_market = any(kw in description_lower for kw in market_keywords)
    market_score = 1.0 if has_market else 0.0
    
    # 4. Logical flow indicators (25% weight)
    # Check if problem appears before solution (better narrative flow)
    problem_positions = [description_lower.find(kw) for kw in problem_keywords if kw in description_lower]
    solution_positions = [description_lower.find(kw) for kw in solution_keywords if kw in description_lower]
    
    flow_score = 0.0
    if problem_positions and solution_positions:
        avg_problem_pos = sum(problem_positions) / len(problem_positions)
        avg_solution_pos = sum(solution_positions) / len(solution_positions)
        
        # If problem mentioned before solution = good flow
        if avg_problem_pos < avg_solution_pos:
            flow_score = 1.0
        elif avg_problem_pos < avg_solution_pos + 100:  # Within 100 chars
            flow_score = 0.7
        else:
            flow_score = 0.3
    elif problem_positions or solution_positions:
        flow_score = 0.5  # Partial flow
    else:
        flow_score = 0.0
    
    # Average all components
    coherence_score = (
        problem_score * 0.25 +
        solution_score * 0.25 +
        market_score * 0.25 +
        flow_score * 0.25
    )
    
    return min(1.0, max(0.0, coherence_score))


def extract_pitch_quality_features(company_description: str) -> Dict:
    """
    Extract all 4 pitch quality features from company description.
    
    Args:
        company_description: Company description/pitch text
    
    Returns:
        Dictionary with 4 pitch quality features:
        - pitch_clarity_score
        - pitch_specificity_score
        - vision_realism_score
        - narrative_coherence_score
    """
    
    features = {}
    
    features['pitch_clarity_score'] = extract_pitch_clarity_score(company_description)
    features['pitch_specificity_score'] = extract_pitch_specificity_score(company_description)
    features['vision_realism_score'] = extract_vision_realism_score(company_description)
    features['narrative_coherence_score'] = extract_narrative_coherence_score(company_description)
    
    return features


if __name__ == "__main__":
    # Test with sample descriptions
    sample_descriptions = [
        "We are building a revolutionary AI platform that will change the world. Our solution uses machine learning to solve problems for businesses. The market is huge, worth $100B.",
        
        "AI-powered platform for automated software development. We help developers write code faster using machine learning. Current market size is $50B, growing at 20% annually. Our beta version has 500+ users.",
        
        "We solve the problem of inefficient code review. Our platform enables teams to review code 3x faster using AI. The developer tools market is $15B and growing. We're currently in beta with 10 enterprise customers."
    ]
    
    print("Testing Pitch Quality Features:")
    print("=" * 60)
    
    for i, desc in enumerate(sample_descriptions, 1):
        print(f"\nSample {i}:")
        print(f"Description: {desc[:100]}...")
        
        features = extract_pitch_quality_features(desc)
        
        print(f"  pitch_clarity_score: {features['pitch_clarity_score']:.3f}")
        print(f"  pitch_specificity_score: {features['pitch_specificity_score']:.3f}")
        print(f"  vision_realism_score: {features['vision_realism_score']:.3f}")
        print(f"  narrative_coherence_score: {features['narrative_coherence_score']:.3f}")

