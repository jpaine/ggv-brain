#!/usr/bin/env python3
"""
Workflow 1: Automated Daily Crunchbase Monitoring
=================================================

Daily cron job that:
1. Scans Crunchbase for new AI companies (founded 2025, USA, with founder LinkedIn)
2. Extracts 18 V11.6.1 features for each founder
3. Scores with V11.6.1 2025-optimized model (trained on 2021-2024, 95.5% test accuracy)
4. Saves all to PostgreSQL
5. Emails jeff@goldengate.vc for all founders (classified by score)

Deployment: Render (daily cron at 12pm UTC / 4am PST)
"""

import os
import sys
import pickle
import json
import requests
import psycopg2
from psycopg2.extras import Json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional
import logging
import asyncio
import aiohttp
import ssl
import certifi
import re

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Environment variables
CRUNCHBASE_API_KEY = os.getenv('CRUNCHBASE_API_KEY', '948f87e1734828b47f1232c4c4c9bcf9')
ENRICHLAYER_API_KEY = os.getenv('ENRICHLAYER_API_KEY', '-RbBxHU8AtMyH1dacibNKA')
RESEND_API_KEY = os.getenv('RESEND_API_KEY', 're_4D7gm8JB_4An2NRYKxWT1VuFXabqPnhD3')
PERPLEXITY_API_KEY = os.getenv('PERPLEXITY_API_KEY', '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')
RECIPIENT_EMAIL = os.getenv('RECIPIENT_EMAIL', 'jeff@goldengate.vc')
FROM_EMAIL = os.getenv('FROM_EMAIL', 'ggv-brain@goldengate.vc')
DATABASE_URL = os.getenv('DATABASE_URL')

# V11.6.1 feature order (18 features total - all using real calculations)
FEATURE_ORDER = [
    'l_level',
    'estimated_age',
    'founder_experience_score',
    'timing_score',
    'market_size_billion',
    'cagr_percent',
    'competitor_count',
    'market_maturity_stage',
    'confidence_score',
    'confidence_score_market',  # Real calculation from market data quality
    'geographic_advantage',
    'description_sentiment',
    'description_complexity',
    'about_quality',
    'sector_keyword_score',
    'founder_market_fit',
    'market_saturation_score',
    'differentiation_score'
]


class V11_6_1_Model:
    """V11.6.1 model loader and scorer - Fully Normalized (trained on 2021-2024 data with normalized features)."""
    
    def __init__(self, model_path='v11_6_1_fully_normalized_model_20251114_193248.pkl',
                 scaler_path='v11_6_1_fully_normalized_scaler_20251114_193248.pkl'):
        """Load V11.6.1 fully normalized model and scaler."""
        logger.info("Loading V11.6.1 fully normalized model (trained on 2021-2024)...")
        logger.info("  Features normalized to match current extraction pipeline:")
        logger.info("    - timing_score: 0-5 → 0-1 scale")
        logger.info("    - market_size_billion: TAM → SAM (÷19)")
        logger.info("    - competitor_count: broad → direct (÷4.6)")
        
        self.model = pickle.load(open(model_path, 'rb'))
        self.scaler = pickle.load(open(scaler_path, 'rb'))
        
        logger.info("✅ V11.6.1 fully normalized model loaded successfully")
    
    def score_founder(self, features: Dict) -> Dict:
        """
        Score a founder with V11.6.1 model.
        
        Returns:
            Dict with score, probability, and feature importance
        """
        # Prepare feature vector in correct order
        feature_vector = [features.get(feat, 0.0) for feat in FEATURE_ORDER]
        
        # Cap outliers at 99th percentile (from training)
        # For production, use conservative defaults if outliers detected
        feature_vector_capped = [min(val, 1.0) if val > 0 else val for val in feature_vector]
        
        # Normalize
        features_scaled = self.scaler.transform([feature_vector_capped])
        
        # Predict probability
        probability = self.model.predict_proba(features_scaled)[0, 1]
        
        # Convert to 0-10 score
        score = probability * 10
        
        # Get feature importance
        feature_importance = self.model.feature_importances_
        top_features = sorted(
            zip(FEATURE_ORDER, feature_importance, feature_vector),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        return {
            'score': float(score),
            'probability': float(probability),
            'top_features': [
                {'feature': f, 'importance': float(imp), 'value': float(val)}
                for f, imp, val in top_features
            ]
        }


class CrunchbaseScanner:
    """Scan Crunchbase for new AI companies."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.crunchbase.com/api/v4"
    
    def search_new_ai_companies(self, days_back: int = 30) -> List[Dict]:
        """
        Search for AI companies founded in 2025, headquartered in USA.
        
        Args:
            days_back: Number of days to look back (default 1 for daily scan)
        
        Returns:
            List of company dictionaries
        """
        logger.info(f"Searching Crunchbase for new AI companies (last {days_back} days)...")
        
        # Date range
        now = datetime.now()
        end_date = now.strftime('%Y-%m-%d')
        start_date = (now - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        search_url = f"{self.base_url}/searches/organizations"
        
        headers = {
            'accept': 'application/json',
            'Content-Type': 'application/json',
            'X-cb-user-key': self.api_key
        }
        
        # Search payload
        payload = {
            "field_ids": [
                "identifier", "name", "short_description", "description",
                "founded_on", "categories", "website", "location_identifiers",
                "linkedin", "founder_identifiers"
            ],
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
            ],
            "limit": 100
        }
        
        try:
            response = requests.post(search_url, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                entities = data.get('entities', [])
                
                companies = []
                for entity in entities:
                    props = entity.get('properties', {})
                    entity_uuid = entity.get('uuid', '')  # Get UUID from entity, not from props
                    
                    # Filter for USA and has founder LinkedIn
                    location = props.get('location_identifiers', [])
                    has_usa = any('united-states' in str(loc).lower() for loc in location) if location else False
                    
                    founder_identifiers = props.get('founder_identifiers', [])
                    has_founders = len(founder_identifiers) > 0 if founder_identifiers else False
                    
                    if has_usa and has_founders:
                        identifier = props.get('identifier')
                        if isinstance(identifier, dict):
                            identifier = identifier.get('value', '')
                        
                        website = props.get('website')
                        if isinstance(website, dict):
                            website = website.get('value', '')
                        
                        founded_on = props.get('founded_on', '')
                        if isinstance(founded_on, dict):
                            founded_on = founded_on.get('value', '')
                        
                        # Store UUID for later use in getting founder entity IDs
                        companies.append({
                            'name': props.get('name', ''),
                            'identifier': identifier or '',
                            'uuid': entity_uuid,  # Store UUID for entity endpoint access
                            'description': props.get('description', '') or props.get('short_description', ''),
                            'website': website or '',
                            'founded_on': founded_on or '',
                            'categories': props.get('categories', []),
                            'founder_identifiers': founder_identifiers  # Keep original for now
                        })
                
                logger.info(f"✅ Found {len(companies)} companies (filtered for USA + founders)")
                return companies
            else:
                logger.error(f"Crunchbase API error: {response.status_code} - {response.text[:200]}")
                return []
        
        except Exception as e:
            logger.error(f"Error searching Crunchbase: {e}")
            return []
    
    def get_founder_entity_ids_from_company(self, company_uuid: str) -> tuple[List[str], List[Dict]]:
        """
        Get founder entity IDs and full founder data from company entity using UUID.
        
        Args:
            company_uuid: Company entity UUID
        
        Returns:
            Tuple of (founder_entity_ids, founder_data_list)
            - founder_entity_ids: List of founder entity IDs (format: "person/{uuid}")
            - founder_data_list: List of founder dictionaries with LinkedIn URLs from founder card
        """
        if not company_uuid:
            return [], []
        
        try:
            url = f"{self.base_url}/entities/organizations/{company_uuid}"
            headers = {
                'accept': 'application/json',
                'X-cb-user-key': self.api_key
            }
            params = {'card_ids': 'founders'}
            
            response = requests.get(url, headers=headers, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                cards = data.get('cards', {})
                founders_list = cards.get('founders', [])
                
                founder_ids = []
                founder_data = []
                for founder in founders_list:
                    # Extract UUID from founder identifier
                    identifier_obj = founder.get('identifier', {})
                    if isinstance(identifier_obj, dict):
                        founder_uuid = identifier_obj.get('uuid', '')
                        founder_name = identifier_obj.get('value', 'Unknown')
                        if founder_uuid:
                            founder_ids.append(f"person/{founder_uuid}")
                            
                            # Extract LinkedIn URL from founder card
                            linkedin = founder.get('linkedin', {})
                            linkedin_url = linkedin.get('value', '') if isinstance(linkedin, dict) else ''
                            
                            founder_data.append({
                                'id': f"person/{founder_uuid}",
                                'name': founder_name,
                                'linkedin_url': linkedin_url,
                                'first_name': founder.get('first_name', ''),
                                'last_name': founder.get('last_name', ''),
                                'title': founder.get('short_description', '')
                            })
                
                return founder_ids, founder_data
            else:
                logger.warning(f"Could not fetch company {company_uuid}: {response.status_code}")
                return [], []
        
        except Exception as e:
            logger.error(f"Error fetching founders for company {company_uuid}: {e}")
            return [], []
    
    def get_founder_details(self, founder_identifiers: List[str]) -> List[Dict]:
        """
        Get detailed founder information including LinkedIn URLs from Crunchbase.
        
        Args:
            founder_identifiers: List of founder entity IDs from Crunchbase
        
        Returns:
            List of founder dictionaries with name, LinkedIn URL, etc.
        """
        if not founder_identifiers:
            return []
        
        founders = []
        headers = {
            'accept': 'application/json',
            'X-cb-user-key': self.api_key
        }
        
        for founder_id in founder_identifiers:
            try:
                # Get founder entity data
                # Handle both "person/{uuid}" format and just UUID
                if '/' in founder_id:
                    # Extract UUID from "person/{uuid}" format
                    founder_uuid = founder_id.split('/')[-1]
                else:
                    founder_uuid = founder_id
                
                founder_url = f"{self.base_url}/entities/people/{founder_uuid}"
                response = requests.get(founder_url, headers=headers, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    props = data.get('properties', {})
                    
                    # Extract LinkedIn URL
                    linkedin = props.get('linkedin', {})
                    linkedin_url = linkedin.get('value', '') if isinstance(linkedin, dict) else ''
                    
                    # Extract other founder details
                    identifier = props.get('identifier', {})
                    name = identifier.get('value', 'Unknown') if isinstance(identifier, dict) else props.get('name', 'Unknown')
                    
                    founders.append({
                        'id': founder_id,
                        'name': name,
                        'linkedin_url': linkedin_url,
                        'first_name': props.get('first_name', ''),
                        'last_name': props.get('last_name', ''),
                        'title': props.get('short_description', '')
                    })
                    logger.info(f"  ✅ Found founder: {name} (LinkedIn: {'Yes' if linkedin_url else 'No'})")
                else:
                    logger.warning(f"  ⚠️ Could not fetch founder {founder_id}: {response.status_code}")
            
            except Exception as e:
                logger.error(f"  ❌ Error fetching founder {founder_id}: {e}")
                continue
        
        return founders


class EnrichlayerService:
    """Get LinkedIn profile data using Enrichlayer API."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.api_endpoint = "https://enrichlayer.com/api/v2/profile"
    
    def get_linkedin_profile(self, linkedin_url: str) -> Optional[Dict]:
        """Get LinkedIn profile data."""
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            params = {
                'profile_url': linkedin_url,
                'extra': 'include',
                'skills': 'include',
                'use_cache': 'if-present',
                'fallback_to_cache': 'on-error'
            }
            
            response = requests.get(self.api_endpoint, params=params, headers=headers, timeout=30)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"Enrichlayer API error: {response.status_code}")
                return None
        
        except Exception as e:
            logger.error(f"Error getting LinkedIn profile: {e}")
            return None


class PerplexityMarketService:
    """Get market analysis using Perplexity API."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.api_url = "https://api.perplexity.ai/chat/completions"
    
    def is_generic_response(self, market_data: Dict) -> bool:
        """
        Detect if Perplexity returned generic/example values.
        
        Generic indicators:
        - Exact example values (2500M, 25% CAGR, 35 competitors)
        - Suspiciously round numbers (e.g., 2500, 25.0, 35)
        - Common default combinations
        """
        if not market_data:
            return False
        
        market_size = market_data.get('market_size_billion', 0) * 1000  # Convert to millions
        cagr = market_data.get('cagr_percent', 0)
        competitors = market_data.get('competitor_count', 0)
        
        # Check for exact example values
        if abs(market_size - 2500) < 0.1 and abs(cagr - 25.0) < 0.1:
            logger.warning(f"❌ Generic response detected: 2500M + 25% CAGR (example values)")
            return True
        
        # Check for suspiciously round numbers (all three are round)
        if (market_size % 500 == 0 and cagr % 5.0 == 0 and competitors % 5 == 0):
            logger.warning(f"❌ Generic response detected: All round numbers ({market_size}M, {cagr}%, {competitors} competitors)")
            return True
        
        # Check for common generic combinations
        generic_combos = [
            (2500, 25.0), (2000, 20.0), (3000, 30.0),
            (1000, 15.0), (1500, 20.0)
        ]
        for generic_size, generic_cagr in generic_combos:
            if abs(market_size - generic_size) < 0.1 and abs(cagr - generic_cagr) < 0.1:
                logger.warning(f"❌ Generic response detected: Common combo ({generic_size}M, {generic_cagr}%)")
                return True
        
        return False
    
    async def get_market_analysis(self, company_name: str, description: str, categories: List[str], max_retries: int = 2) -> Optional[Dict]:
        """
        Get market analysis from Perplexity API with retry logic for generic responses.
        
        Args:
            company_name: Company name
            description: Company description
            categories: List of category strings
            max_retries: Maximum number of retry attempts if generic response detected (default: 2)
        
        Returns:
            Dictionary with market_size_millions, cagr_percent, competitor_count, timing_score, momentum_score
            Returns None if API call fails (no defaults per requirement)
        """
        if not self.api_key:
            logger.error("Perplexity API key not set")
            return None
        
        category_str = ', '.join(categories) if categories else 'Technology'
        
        prompt = f"""
        Research and analyze the SPECIFIC market for this startup:
        
        Company: {company_name}
        Description: {description}
        Categories: {category_str}
        
        CRITICAL REQUIREMENTS:
        1. Research THIS SPECIFIC COMPANY and its direct competitors (name at least 3 competitors)
        2. Identify the NICHE market this company targets (not broad "AI" or "Software")
        3. Calculate SERVICEABLE ADDRESSABLE MARKET (SAM) - what THIS startup can realistically capture in 3-5 years
        4. Use PRECISE numbers with decimals (e.g., $847M, 18.3% CAGR, not round $2500M, 25%)
        5. Provide UNIQUE analysis - DO NOT use generic estimates or example values
        6. If precise data unavailable, estimate narrower niche market (smaller = more realistic)
        
        Market Analysis Focus:
        - Market size: $100M-$5B serviceable market (NOT total industry TAM)
        - CAGR: Realistic growth rate with decimal precision (e.g., 18.7%, not 25%)
        - Competitors: Exact count of direct competitors in THIS niche
        - Timing: How well-timed is entry based on market maturity (0-1 scale, use decimals)
        - Momentum: Current market momentum/hype (0-1 scale, use decimals)
        
        Return JSON with ACTUAL research data (NOT example values):
        {{
            "market_size_millions": <integer - precise estimate>,
            "cagr_percent": <float - use decimals like 18.3>,
            "competitor_count": <integer - exact count>,
            "timing_score": <float 0-1 - use decimals like 0.73>,
            "momentum_score": <float 0-1 - use decimals like 0.68>,
            "data_confidence": <"high"|"medium"|"low">
        }}
        
        ⚠️ IMPORTANT: Each company should have DIFFERENT values. Avoid returning 2500/25.0 repeatedly.
        """
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "sonar-pro",
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1
            }
            
            # Create SSL context with certifi certificates
            ssl_context = ssl.create_default_context(cafile=certifi.where())
            
            async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=ssl_context)) as session:
                async with session.post(self.api_url, headers=headers, json=payload, timeout=60) as response:
                    if response.status == 200:
                        data = await response.json()
                        content = data.get('choices', [{}])[0].get('message', {}).get('content', '')
                        
                        # Parse JSON from response (may have markdown formatting)
                        # Try to extract JSON object from response
                        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*"market_size_millions"[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content, re.DOTALL)
                        if not json_match:
                            # Try simpler pattern
                            json_match = re.search(r'\{.*?"market_size_millions".*?\}', content, re.DOTALL)
                        
                        if json_match:
                            try:
                                market_data = json.loads(json_match.group())
                                
                                # Convert market_size_millions to billions for feature
                                market_size_millions = market_data.get('market_size_millions', 0)
                                market_size_billions = market_size_millions / 1000.0 if market_size_millions > 0 else 0
                                
                                result = {
                                    'market_size_billion': market_size_billions,
                                    'cagr_percent': float(market_data.get('cagr_percent', 0)),
                                    'competitor_count': int(market_data.get('competitor_count', 0)),
                                    'timing_score': float(market_data.get('timing_score', 0)),
                                    'momentum_score': float(market_data.get('momentum_score', 0))
                                }
                                
                                # Check if response is generic
                                if self.is_generic_response(result):
                                    if max_retries > 0:
                                        logger.warning(f"⚠️ Retrying market analysis for {company_name} (attempts left: {max_retries})")
                                        await asyncio.sleep(1)  # Brief delay before retry
                                        return await self.get_market_analysis(company_name, description, categories, max_retries - 1)
                                    else:
                                        logger.warning(f"⚠️ All retries exhausted, accepting generic response for {company_name}")
                                
                                return result
                            except json.JSONDecodeError as e:
                                logger.error(f"JSON decode error: {e}")
                                logger.error(f"Content snippet: {content[:500]}")
                                return None
                        else:
                            logger.error(f"Could not find JSON in Perplexity response")
                            logger.error(f"Content snippet: {content[:500]}")
                            return None
                    else:
                        error_text = await response.text()
                        logger.error(f"Perplexity API error {response.status}: {error_text[:200]}")
                        return None
        
        except Exception as e:
            logger.error(f"Error calling Perplexity API: {e}")
            return None


class FeatureExtractor:
    """Extract all 18 V11.6.1 features from company data (all using real calculations)."""
    
    def extract_features(self, company: Dict, linkedin_profiles: Optional[List[Dict]] = None, market_analysis: Optional[Dict] = None) -> Dict:
        """
        Extract all 18 V11.6.1 features.
        
        Args:
            company: Crunchbase company data
            linkedin_profiles: List of Enrichlayer LinkedIn profile data (for multiple founders)
            market_analysis: Perplexity market analysis data (optional)
        
        Returns:
            Dictionary with all 18 features
        """
        features = {}
        
        # Extract V11.6.1 features (18 features - all using real calculations)
        features.update(self._extract_v11_4_features(company, linkedin_profiles, market_analysis))
        
        # Note: Phase 5 features removed (not in V11.6.1)
        # V11.6.1 uses only the 18 core features with real calculations
        
        return features
    
    def _extract_v11_4_features(self, company: Dict, linkedin_profiles: Optional[List[Dict]], market_analysis: Optional[Dict]) -> Dict:
        """Extract V11.4 features using real data from LinkedIn and Perplexity."""
        features = {}
        
        description = company.get('description', '')
        founded_year = self._parse_year(company.get('founded_on', ''))
        
        # Founder characteristics (aggregate across all founders)
        if linkedin_profiles and len(linkedin_profiles) > 0:
            l_levels = [self._estimate_l_level(profile) for profile in linkedin_profiles if profile]
            experience_scores = [self._calculate_experience_score(profile) for profile in linkedin_profiles if profile]
            
            # Average across founders
            features['l_level'] = sum(l_levels) / len(l_levels) if l_levels else 2.0
            features['founder_experience_score'] = sum(experience_scores) / len(experience_scores) if experience_scores else 0.5
        else:
            features['l_level'] = 2.0
            features['founder_experience_score'] = 0.5
        
        # V11.6.1: Use real calculations for all features (no hardcoded defaults)
        from phase2_real_feature_calculations import (
            calculate_estimated_age,
            calculate_description_sentiment,
            calculate_founder_market_fit,
            calculate_market_saturation_score,
            calculate_differentiation_score,
            calculate_confidence_score,
            calculate_confidence_score_market
        )
        
        # 1. estimated_age - Real calculation from LinkedIn
        features['estimated_age'] = calculate_estimated_age(linkedin_profiles) if linkedin_profiles else 35.0
        
        # Market characteristics from Perplexity API (REAL DATA, no defaults)
        if market_analysis:
            features['timing_score'] = market_analysis.get('timing_score', 0.7)
            features['market_size_billion'] = market_analysis.get('market_size_billion', 100.0)
            features['cagr_percent'] = market_analysis.get('cagr_percent', 25.0)
            features['competitor_count'] = market_analysis.get('competitor_count', 50)
            
            # Calculate market maturity stage from market size and CAGR
            market_size = market_analysis.get('market_size_billion', 100.0)
            if market_size > 5.0:
                features['market_maturity_stage'] = 0.8  # Mature market
            elif market_size > 1.0:
                features['market_maturity_stage'] = 0.6  # Growing market
            else:
                features['market_maturity_stage'] = 0.4  # Early market
            
            # Confidence scores based on data quality (REAL CALCULATION)
            features['confidence_score'] = calculate_confidence_score(company, linkedin_profiles or [], market_analysis)
            features['confidence_score_market'] = calculate_confidence_score_market(market_analysis)
        else:
            # If Perplexity fails, we skip the company (no defaults per requirement)
            raise ValueError("Market analysis is required - cannot use defaults")
        
        features['geographic_advantage'] = 0.8 if 'san francisco' in str(company.get('location', '')).lower() else 0.5
        
        # 2. description_sentiment - Real calculation (TextBlob)
        features['description_sentiment'] = calculate_description_sentiment(description)
        
        # 3. description_complexity - Real calculation (word count)
        features['description_complexity'] = len(description.split()) / 100.0 if description else 0.5
        
        # 4. about_quality - Real calculation (description quality metrics)
        # Quality based on: length, word count, sentence structure
        desc_len = len(description) if description else 0
        word_count = len(description.split()) if description else 0
        sentence_count = len([s for s in description.split('.') if s.strip()]) if description else 0
        
        # Normalize to 0-1 scale
        # Length score (0-0.4): 0-50 chars = 0.1, 50-100 = 0.2, 100-200 = 0.3, 200+ = 0.4
        length_score = min(0.4, desc_len / 500.0)
        
        # Word count score (0-0.3): more words = higher quality
        word_score = min(0.3, word_count / 100.0)
        
        # Sentence structure score (0-0.3): more sentences = better structure
        sentence_score = min(0.3, sentence_count / 10.0)
        
        features['about_quality'] = round(length_score + word_score + sentence_score, 3)
        
        # 5. sector_keyword_score - Real calculation
        features['sector_keyword_score'] = self._calculate_sector_score(description)
        
        # 6. founder_market_fit - Real calculation
        categories = company.get('categories', [])
        category_strings = [cat.get('value', '') if isinstance(cat, dict) else str(cat) for cat in categories]
        features['founder_market_fit'] = calculate_founder_market_fit(
            linkedin_profiles or [], market_analysis, category_strings
        ) if linkedin_profiles and market_analysis else 0.5
        
        # 7. market_saturation_score - Real calculation
        features['market_saturation_score'] = calculate_market_saturation_score(
            features.get('competitor_count', 50),
            features.get('market_size_billion', 100.0)
        )
        
        # 8. differentiation_score - Real calculation
        features['differentiation_score'] = calculate_differentiation_score(
            description,
            features.get('competitor_count', 50),
            market_analysis
        )
        
        # Note: Temporal features (momentum_3m, timing_score_trend, competitor_growth_rate) 
        # and Phase 5 features removed - not in V11.6.1
        
        return features
    
    def _extract_phase5_features(self, company: Dict, linkedin_profiles: Optional[List[Dict]]) -> Dict:
        """Extract Phase 5 features from multiple LinkedIn profiles."""
        features = {}
        
        description = company.get('description', '')
        
        # Week 1: Serial founder features (aggregate across all founders)
        if linkedin_profiles and len(linkedin_profiles) > 0:
            all_experience = []
            founder_keywords = ['founder', 'co-founder', 'cofounder', 'ceo']
            
            for profile in linkedin_profiles:
                if profile:
                    # Enrichlayer API returns 'experiences' (plural) not 'experience'
                    experience = profile.get('experiences', profile.get('experience', []))
                    all_experience.extend(experience)
            
            # Count founder roles across all profiles
            founder_count = sum(
                1 for exp in all_experience
                if any(kw in str(exp.get('title', '')).lower() for kw in founder_keywords)
            )
            
            features['is_serial_founder'] = 1 if founder_count > 1 else 0
            features['founder_experience_years'] = min(len(all_experience) * 2.5 / 20.0, 1.0) if all_experience else 0.0
            
            # Week 1: LinkedIn network estimates (aggregate across all founders)
            industry_keywords = ['ai', 'ml', 'software', 'technology', 'startup']
            industry_experience = sum(
                1 for exp in all_experience
                if any(kw in str(exp.get('title', '')).lower() + str(exp.get('company', '')).lower() 
                      for kw in industry_keywords)
            )
            
            if len(all_experience) > 0:
                features['linkedin_industry_connections_ratio'] = 0.2 + (industry_experience / len(all_experience)) * 0.5
            else:
                features['linkedin_industry_connections_ratio'] = 0.2
            
            features['linkedin_influencer_connections'] = 0.3 if founder_count > 0 else 0.1
        else:
            features['is_serial_founder'] = 0
            features['founder_experience_years'] = 0.0
            features['linkedin_industry_connections_ratio'] = 0.2
            features['linkedin_influencer_connections'] = 0.1
        
        # Week 2: Barrier to entry
        barrier_keywords = ['ai', 'ml', 'deep learning', 'proprietary', 'patent']
        features['barrier_to_entry_score'] = 0.5 if any(kw in description.lower() for kw in barrier_keywords) else 0.3
        
        # NOTE: Removed pitch quality features (4) - noise/not predictive
        # NOTE: Removed linkedin_2nd_degree_reach - not available from Enrichlayer API
        
        return features
    
    def _estimate_l_level(self, linkedin_profile: Dict) -> float:
        """Estimate founder L level from LinkedIn profile."""
        # Enrichlayer API returns 'experiences' (plural) not 'experience'
        experience = linkedin_profile.get('experiences', linkedin_profile.get('experience', []))
        education = linkedin_profile.get('education', [])
        
        # Simple heuristic
        score = 2.0  # Base L2
        
        # Check for founder experience
        founder_keywords = ['founder', 'co-founder', 'ceo']
        has_founder_exp = any(
            any(kw in str(exp.get('title', '')).lower() for kw in founder_keywords)
            for exp in experience
        )
        if has_founder_exp:
            score += 1.0  # L3
        
        # Check for top company experience
        top_companies = ['google', 'facebook', 'amazon', 'microsoft', 'apple', 'openai', 'anthropic']
        has_top_company = any(
            any(comp in str(exp.get('company', '')).lower() for comp in top_companies)
            for exp in experience
        )
        if has_top_company:
            score += 0.5
        
        # Check for education
        top_schools = ['stanford', 'mit', 'harvard', 'berkeley', 'carnegie mellon']
        has_top_school = any(
            any(school in str(edu.get('school', '')).lower() for school in top_schools)
            for edu in education
        )
        if has_top_school:
            score += 0.5
        
        return min(score, 7.0)  # Cap at L7
    
    def _calculate_experience_score(self, linkedin_profile: Dict) -> float:
        """Calculate founder experience score."""
        # Enrichlayer API returns 'experiences' (plural) not 'experience'
        experience = linkedin_profile.get('experiences', linkedin_profile.get('experience', []))
        
        if not experience:
            return 0.3
        
        # Years of experience
        total_years = len(experience) * 2.5  # Estimate 2.5 years per role
        experience_score = min(total_years / 15.0, 1.0)  # Cap at 15 years
        
        return experience_score
    
    def _calculate_sector_score(self, description: str) -> float:
        """Calculate sector keyword relevance."""
        if not description:
            return 0.0
        
        ai_keywords = ['ai', 'ml', 'machine learning', 'artificial intelligence', 
                      'deep learning', 'neural network', 'nlp', 'computer vision']
        
        desc_lower = description.lower()
        keyword_count = sum(1 for kw in ai_keywords if kw in desc_lower)
        
        return min(keyword_count / 5.0, 1.0)
    
    def _parse_year(self, date_str: str) -> int:
        """Parse year from date string."""
        try:
            if not date_str:
                return 2025
            return int(str(date_str)[:4])
        except:
            return 2025


class DatabaseService:
    """PostgreSQL database service for tracking scored companies."""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.conn = None
    
    def connect(self):
        """Connect to PostgreSQL."""
        try:
            self.conn = psycopg2.connect(self.database_url)
            logger.info("✅ Connected to PostgreSQL")
        except Exception as e:
            logger.error(f"Error connecting to database: {e}")
            raise
    
    def save_scored_company(self, company: Dict, score_result: Dict, linkedin_url: str = None, emailed: bool = False):
        """Save scored company to database."""
        if not self.conn:
            self.connect()
        
        try:
            cursor = self.conn.cursor()
            
            founded_on = company.get('founded_on')
            founded_year = None
            if founded_on:
                try:
                    founded_year = int(str(founded_on)[:4])
                except (ValueError, TypeError):
                    founded_year = None
            
            crunchbase_identifier = company.get('identifier', '')
            crunchbase_url = f"https://www.crunchbase.com/organization/{crunchbase_identifier}" if crunchbase_identifier else None
            
            query = """
                INSERT INTO scored_companies 
                (name, crunchbase_url, crunchbase_identifier, founded_year, description, website, linkedin_url,
                 v11_5_score, predicted_probability, features, top_features, emailed, email_sent_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (name) DO UPDATE SET
                    v11_5_score = EXCLUDED.v11_5_score,  # Keep column name for compatibility
                    predicted_probability = EXCLUDED.predicted_probability,
                    emailed = EXCLUDED.emailed,
                    email_sent_at = CASE WHEN EXCLUDED.emailed THEN EXCLUDED.email_sent_at ELSE scored_companies.email_sent_at END,
                    updated_at = NOW()
            """
            
            email_sent_at = datetime.now() if emailed else None
            
            cursor.execute(query, (
                company.get('name'),
                crunchbase_url,
                crunchbase_identifier,
                founded_year,
                company.get('description'),
                company.get('website'),
                linkedin_url,
                score_result.get('score'),
                score_result.get('probability'),
                Json({}),  # features dict
                Json(score_result.get('top_features', [])),
                emailed,
                email_sent_at
            ))
            
            self.conn.commit()
            cursor.close()
            
            logger.info(f"✅ Saved {company.get('name')} to database (score: {score_result.get('score'):.2f})")
            
        except Exception as e:
            logger.error(f"Error saving to database: {e}")
            if self.conn:
                self.conn.rollback()
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
    
    def log_scan_summary(
        self,
        scan_date: date,
        companies_found: int,
        companies_scored: int,
        high_scores_count: int,
        emails_sent: int,
        duration_seconds: int,
        status: str = 'success',
        error_message: str = None,
    ):
        """Log daily scan metrics to scan_history."""
        if not self.conn:
            self.connect()
        
        try:
            cursor = self.conn.cursor()
            query = """
                INSERT INTO scan_history
                (scan_date, companies_found, companies_scored, high_scores_count,
                 emails_sent, scan_duration_seconds, status, error_message)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (scan_date) DO UPDATE SET
                    companies_found = EXCLUDED.companies_found,
                    companies_scored = EXCLUDED.companies_scored,
                    high_scores_count = EXCLUDED.high_scores_count,
                    emails_sent = EXCLUDED.emails_sent,
                    scan_duration_seconds = EXCLUDED.scan_duration_seconds,
                    status = EXCLUDED.status,
                    error_message = EXCLUDED.error_message
            """
            cursor.execute(
                query,
                (
                    scan_date,
                    companies_found,
                    companies_scored,
                    high_scores_count,
                    emails_sent,
                    duration_seconds,
                    status,
                    error_message,
                ),
            )
            self.conn.commit()
            cursor.close()
        except Exception as e:
            logger.error(f"Error logging scan summary: {e}")
            if self.conn:
                self.conn.rollback()


class EmailAlertService:
    """Send email alerts via Resend API."""
    
    def __init__(self, api_key: str, from_email: str, recipient_email: str):
        self.api_key = api_key
        self.from_email = from_email
        self.recipient_email = recipient_email
        self.api_url = "https://api.resend.com/emails"
    
    def send_founder_alert(self, company: Dict, score_result: Dict, linkedin_url: str = None):
        """Send email alert for every founder, with appropriate classification based on score."""
        score = score_result.get('score', 0)
        
        try:
            # Use different subject/header based on score
            if score >= 8.0:
                subject = f"🚀 GGV Brain Alert: {company.get('name')} - Score {score:.1f}/10 (High Potential)"
            elif score >= 7.0:
                subject = f"📊 GGV Brain Alert: {company.get('name')} - Score {score:.1f}/10 (Strong)"
            elif score >= 5.0:
                subject = f"📊 GGV Brain Update: {company.get('name')} - Score {score:.1f}/10 (Average)"
            else:
                subject = f"📊 GGV Brain Update: {company.get('name')} - Score {score:.1f}/10 (Below Average)"
            
            html_content = self._generate_email_html(company, score_result, linkedin_url)
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "from": self.from_email,
                "to": [self.recipient_email],
                "subject": subject,
                "html": html_content
            }
            
            response = requests.post(self.api_url, headers=headers, json=payload)
            
            if response.status_code == 200:
                logger.info(f"✅ Email sent for {company.get('name')}")
                return True
            else:
                logger.error(f"Email send failed: {response.status_code} - {response.text}")
                return False
        
        except Exception as e:
            logger.error(f"Error sending email: {e}")
            return False
    
    def _generate_email_html(self, company: Dict, score_result: Dict, linkedin_url: str = None) -> str:
        """Generate HTML email content."""
        score = score_result.get('score', 0)
        
        top_features_html = ''.join([
            f"<li><strong>{f['feature']}</strong>: {f['value']:.3f} (importance: {f['importance']:.1%})</li>"
            for f in score_result.get('top_features', [])
        ])
        
        linkedin_html = f'<p><strong>Founder LinkedIn:</strong> <a href="{linkedin_url}">{linkedin_url}</a></p>' if linkedin_url else ''
        
        # Conditional header and color based on score
        if score >= 8.0:
            header = "🚀 High-Potential Founder Alert"
            score_color = "#1a73e8"  # Blue for high scores
            alert_note = "✅ Strong investment signal - Score meets 8.0+ threshold"
        elif score >= 7.0:
            header = "📊 Strong Founder Alert"
            score_color = "#34a853"  # Green for good scores
            alert_note = "⚠️ Worth deep dive - Score above 7.0"
        else:
            header = "📊 Founder Score Update"
            score_color = "#ea4335"  # Red for low scores
            alert_note = "❌ Below threshold - Score < 8.0 (not high potential)"
        
        html = f"""
        <html>
        <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
            <h2 style="color: {score_color};">{header}</h2>
            
            <div style="background: #f5f5f5; padding: 20px; border-radius: 8px; margin: 20px 0;">
                <h3 style="margin-top: 0;">{company.get('name')}</h3>
                <p><strong>V11.6.1 Score:</strong> <span style="font-size: 24px; color: {score_color}; font-weight: bold;">{score_result.get('score'):.1f}/10</span></p>
                <p style="color: {score_color}; font-weight: bold;">{alert_note}</p>
                <p><strong>Probability:</strong> {score_result.get('probability'):.1%}</p>
                <p><strong>Founded:</strong> {company.get('founded_on', 'N/A')}</p>
                <p><strong>Website:</strong> <a href="{company.get('website', '#')}">{company.get('website', 'N/A')}</a></p>
                {linkedin_html}
                <p><strong>Crunchbase:</strong> <a href="https://www.crunchbase.com/organization/{company.get('identifier', '')}">View Profile</a></p>
            </div>
            
            <div style="margin: 20px 0;">
                <h4>Description:</h4>
                <p style="color: #666;">{company.get('description', 'N/A')}</p>
            </div>
            
            <div style="margin: 20px 0;">
                <h4>Top Contributing Features:</h4>
                <ul>{top_features_html}</ul>
            </div>
            
            <hr style="border: none; border-top: 1px solid #ddd; margin: 30px 0;">
            
            <p style="color: #999; font-size: 12px;">
                This alert was generated by GGV Brain V11.6.1 (66.7% accuracy on historical data).
                Model predicts funding efficiency based on 18 founding-time features (all using real calculations).
            </p>
        </body>
        </html>
        """
        
        return html


def main():
    """Main workflow execution."""
    run_start = datetime.now()
    logger.info("="*80)
    logger.info("WORKFLOW 1: CRUNCHBASE DAILY MONITOR")
    logger.info("="*80)
    logger.info(f"Started: {run_start.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("")
    
    # Initialize services
    model = V11_6_1_Model()
    crunchbase = CrunchbaseScanner(CRUNCHBASE_API_KEY)
    enrichlayer = EnrichlayerService(ENRICHLAYER_API_KEY)
    perplexity = PerplexityMarketService(PERPLEXITY_API_KEY)
    feature_extractor = FeatureExtractor()
    email_service = EmailAlertService(RESEND_API_KEY, FROM_EMAIL, RECIPIENT_EMAIL)
    
    # Connect to database
    if DATABASE_URL:
        db = DatabaseService(DATABASE_URL)
        db.connect()
    else:
        logger.warning("⚠️ No DATABASE_URL provided, skipping database saves")
        db = None
    
    # Search for new companies (last 30 days)
    companies = crunchbase.search_new_ai_companies(days_back=30)
    
    if not companies:
        logger.info("No new companies found")
        return
    
    logger.info(f"Processing {len(companies)} companies...")
    logger.info("")
    
    # Process each company
    high_score_count = 0
    companies_scored = 0
    
    for i, company in enumerate(companies, 1):
        logger.info(f"{i}/{len(companies)}: {company.get('name')}")
        
        linkedin_urls = []  # Initialize outside try block
        features = None
        
        try:
            # Step 1: Get founder details from Crunchbase (including LinkedIn URLs)
            # First, get founder entity IDs and data from company entity using UUID
            company_uuid = company.get('uuid', '')
            founders = []
            founder_identifiers = []
            
            if company_uuid:
                # Get founder entity IDs and data from company entity endpoint
                founder_identifiers, founders_from_card = crunchbase.get_founder_entity_ids_from_company(company_uuid)
                if founders_from_card:
                    # Use founder data from card (includes LinkedIn URLs)
                    founders = founders_from_card
                elif founder_identifiers:
                    # Fallback: get full details using entity IDs
                    founders = crunchbase.get_founder_details(founder_identifiers)
            
            # Fallback: use founder_identifiers from search if UUID method failed
            if not founders:
                founder_identifiers = company.get('founder_identifiers', [])
                founders = crunchbase.get_founder_details(founder_identifiers) if founder_identifiers else []
            
            if not founders:
                logger.warning(f"  ⚠️ No founder data available - skipping {company.get('name')}")
                continue
            
            # Step 2: Get LinkedIn profiles for all founders via Enrichlayer
            linkedin_profiles = []
            for founder in founders:
                linkedin_url = founder.get('linkedin_url')
                if linkedin_url:
                    linkedin_urls.append(linkedin_url)
                    profile = enrichlayer.get_linkedin_profile(linkedin_url)
                    if profile:
                        linkedin_profiles.append(profile)
                    else:
                        logger.warning(f"  ⚠️ Could not fetch LinkedIn profile for {founder.get('name')}")
            
            if not linkedin_profiles:
                logger.warning(f"  ⚠️ No LinkedIn profiles available - skipping {company.get('name')}")
                continue
            
            # Step 3: Get market analysis from Perplexity (async)
            categories = company.get('categories', [])
            category_strings = [cat.get('value', '') if isinstance(cat, dict) else str(cat) for cat in categories]
            
            market_analysis = asyncio.run(perplexity.get_market_analysis(
                company.get('name', ''),
                company.get('description', ''),
                category_strings
            ))
            
            if not market_analysis:
                logger.error(f"  ❌ Market analysis failed - skipping {company.get('name')} (no defaults per requirement)")
                continue
            
            logger.info(f"  ✅ Market analysis: ${market_analysis.get('market_size_billion', 0):.2f}B, {market_analysis.get('cagr_percent', 0):.1f}% CAGR")
            
            # Step 4: Extract features with real data
            features = feature_extractor.extract_features(company, linkedin_profiles, market_analysis)
            
        except ValueError as e:
            # Market analysis required error
            logger.error(f"  ❌ {e} - skipping {company.get('name')}")
            continue
        except Exception as e:
            logger.error(f"  ❌ Error processing {company.get('name')}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            continue
        
        if features is None:
            logger.warning(f"  ⚠️ Features not extracted - skipping {company.get('name')}")
            continue
        
        # Score with V11.6.1
        score_result = model.score_founder(features)
        score = score_result.get('score', 0)
        
        logger.info(f"  Score: {score:.2f}/10 (probability: {score_result.get('probability'):.1%})")
        
        # Send email alert for every founder (regardless of score)
        # Email content will be classified appropriately based on score
        # Use first LinkedIn URL for email (or concatenate if multiple)
        primary_linkedin_url = linkedin_urls[0] if linkedin_urls else None
        email_sent = email_service.send_founder_alert(company, score_result, primary_linkedin_url)
        if email_sent:
            if score >= 8.0:
                logger.info("  📧 Email alert sent (high potential)")
                high_score_count += 1
            elif score >= 7.0:
                logger.info("  📧 Email alert sent (strong)")
            elif score >= 5.0:
                logger.info("  📧 Email alert sent (average)")
            else:
                logger.info("  📧 Email alert sent (below average)")
        else:
            logger.warning("  ⚠️ Email alert failed")
        
        # Save to database (mark as emailed if email was sent)
        if db:
            primary_linkedin_url = linkedin_urls[0] if linkedin_urls else None
            db.save_scored_company(company, score_result, primary_linkedin_url, emailed=email_sent)
        
        companies_scored += 1
        
        logger.info("")
    
    # Summary
    logger.info("="*80)
    logger.info("DAILY SCAN SUMMARY")
    logger.info("="*80)
    logger.info(f"Companies scanned: {len(companies)}")
    logger.info(f"Companies scored: {companies_scored}")
    logger.info(f"High-potential founders (>= 8.0): {high_score_count}")
    logger.info(f"Total email alerts sent: {companies_scored}")
    run_end = datetime.now()
    logger.info(f"Completed: {run_end.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Close database
    if db:
        try:
            db.log_scan_summary(
                scan_date=run_start.date(),
                companies_found=len(companies),
                companies_scored=companies_scored,
                high_scores_count=high_score_count,
                emails_sent=high_score_count,
                duration_seconds=int((run_end - run_start).total_seconds()),
            )
        except Exception as e:
            logger.error(f"Failed to log scan summary: {e}")
        db.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Workflow failed: {e}")
        sys.exit(1)

