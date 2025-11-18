#!/usr/bin/env python3
"""
Scoring Service Wrapper
=======================
Wrapper function to process a company from Crunchbase URL through the full scoring pipeline.
"""

import os
import sys
import asyncio
import re
import requests
from typing import Dict, Optional
from datetime import datetime

# Add parent directory to path to import workflow modules
# This allows importing from the main BRAIN project directory
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_file_dir, '..', '..')
sys.path.insert(0, project_root)

from workflow_1_crunchbase_daily_monitor import (
    V11_7_1_Model, CrunchbaseScanner, EnrichlayerService,
    PerplexityMarketService, FeatureExtractor, EmailAlertService
)

# Environment variables
CRUNCHBASE_API_KEY = os.getenv('CRUNCHBASE_API_KEY')
ENRICHLAYER_API_KEY = os.getenv('ENRICHLAYER_API_KEY')
PERPLEXITY_API_KEY = os.getenv('PERPLEXITY_API_KEY')
RESEND_API_KEY = os.getenv('RESEND_API_KEY')
RECIPIENT_EMAIL = os.getenv('RECIPIENT_EMAIL', 'jeffrey.paine@gmail.com')
FROM_EMAIL = os.getenv('FROM_EMAIL', 'onboarding@resend.dev')


def extract_identifier_from_url(crunchbase_url: str) -> Optional[str]:
    """
    Extract company identifier from Crunchbase URL.
    
    Examples:
    - https://www.crunchbase.com/organization/company-name -> company-name
    - https://www.crunchbase.com/organization/company-name?query=... -> company-name
    """
    match = re.search(r'/organization/([^/?]+)', crunchbase_url)
    if match:
        return match.group(1)
    return None


def get_company_by_identifier(identifier: str, scanner: CrunchbaseScanner) -> Optional[Dict]:
    """
    Get company data by identifier using Crunchbase API.
    
    Args:
        identifier: Company identifier (slug)
        scanner: CrunchbaseScanner instance
    
    Returns:
        Company dictionary or None if not found
    """
    try:
        # Use entity endpoint to get company by identifier
        entity_url = f"{scanner.base_url}/entities/organizations/{identifier}"
        headers = {
            'accept': 'application/json',
            'X-cb-user-key': scanner.api_key
        }
        
        response = requests.get(entity_url, headers=headers, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            entity = data.get('properties', {})
            entity_uuid = data.get('uuid', '')
            
            # Extract company data
            identifier_value = entity.get('identifier', {})
            if isinstance(identifier_value, dict):
                identifier_value = identifier_value.get('value', '')
            
            website = entity.get('website', {})
            if isinstance(website, dict):
                website = website.get('value', '')
            
            founded_on = entity.get('founded_on', {})
            if isinstance(founded_on, dict):
                founded_on = founded_on.get('value', '')
            
            return {
                'name': entity.get('name', ''),
                'identifier': identifier_value or identifier,
                'uuid': entity_uuid,
                'description': entity.get('description', '') or entity.get('short_description', ''),
                'website': website or '',
                'founded_on': founded_on or '',
                'categories': entity.get('categories', []),
                'founder_identifiers': entity.get('founder_identifiers', []),
                'location_identifiers': entity.get('location_identifiers', [])
            }
        else:
            return None
    except Exception as e:
        print(f"Error fetching company: {e}")
        return None


async def process_company_from_url(crunchbase_url: str) -> Dict:
    """
    Process a company from Crunchbase URL through the full scoring pipeline.
    
    Args:
        crunchbase_url: Full Crunchbase URL (e.g., https://www.crunchbase.com/organization/company-name)
    
    Returns:
        Dictionary with:
        - success: bool
        - company_name: str
        - score: float
        - probability: float
        - top_features: list
        - error: str (if failed)
        - email_sent: bool
    """
    result = {
        'success': False,
        'company_name': '',
        'score': 0.0,
        'probability': 0.0,
        'top_features': [],
        'error': None,
        'email_sent': False
    }
    
    try:
        # Step 1: Extract identifier from URL
        identifier = extract_identifier_from_url(crunchbase_url)
        if not identifier:
            result['error'] = 'Invalid Crunchbase URL format. Expected: https://www.crunchbase.com/organization/company-name'
            return result
        
        # Step 2: Initialize services
        # Model paths - adjust based on deployment location
        # In Vercel, models should be in the same directory or accessible via absolute path
        model_dir = os.path.join(os.path.dirname(__file__), '..', 'public', 'models')
        model_path = os.path.join(model_dir, 'v11_7_1_fixed_distribution_model_20251114_214215.pkl')
        scaler_path = os.path.join(model_dir, 'v11_7_1_fixed_distribution_scaler_20251114_214215.pkl')
        
        # Fallback to root directory if not found
        if not os.path.exists(model_path):
            model_path = 'v11_7_1_fixed_distribution_model_20251114_214215.pkl'
            scaler_path = 'v11_7_1_fixed_distribution_scaler_20251114_214215.pkl'
        
        model = V11_7_1_Model(model_path=model_path, scaler_path=scaler_path)
        scanner = CrunchbaseScanner(CRUNCHBASE_API_KEY)
        enrichlayer = EnrichlayerService(ENRICHLAYER_API_KEY)
        perplexity = PerplexityMarketService(PERPLEXITY_API_KEY)
        feature_extractor = FeatureExtractor()
        email_service = EmailAlertService(RESEND_API_KEY, FROM_EMAIL, RECIPIENT_EMAIL)
        
        # Step 3: Get company data
        company = get_company_by_identifier(identifier, scanner)
        if not company:
            result['error'] = f'Company not found: {identifier}'
            return result
        
        result['company_name'] = company.get('name', '')
        
        # Step 4: Get founder details
        company_uuid = company.get('uuid', '')
        founders = []
        linkedin_urls = []
        
        if company_uuid:
            founder_identifiers, founders_from_card = scanner.get_founder_entity_ids_from_company(company_uuid)
            if founders_from_card:
                founders = founders_from_card
            elif founder_identifiers:
                founders = scanner.get_founder_details(founder_identifiers)
        
        if not founders:
            founder_identifiers = company.get('founder_identifiers', [])
            founders = scanner.get_founder_details(founder_identifiers) if founder_identifiers else []
        
        if not founders:
            result['error'] = 'No founders found for this company'
            return result
        
        # Step 5: Get LinkedIn profiles
        linkedin_profiles = []
        for founder in founders:
            linkedin_url = founder.get('linkedin_url')
            if linkedin_url:
                linkedin_urls.append(linkedin_url)
                profile = enrichlayer.get_linkedin_profile(linkedin_url)
                if profile:
                    linkedin_profiles.append(profile)
        
        if not linkedin_profiles:
            result['error'] = 'No LinkedIn profiles found for founders'
            return result
        
        # Step 6: Get market analysis
        categories = company.get('categories', [])
        category_strings = [cat.get('value', '') if isinstance(cat, dict) else str(cat) for cat in categories]
        
        market_analysis = await perplexity.get_market_analysis(
            company.get('name', ''),
            company.get('description', ''),
            category_strings
        )
        
        if not market_analysis:
            result['error'] = 'Market analysis failed'
            return result
        
        # Verify required fields
        required_fields = ['timing_score', 'market_size_billion', 'cagr_percent', 'competitor_count']
        missing = [f for f in required_fields if f not in market_analysis or market_analysis.get(f) is None]
        if missing:
            result['error'] = f'Market analysis missing required fields: {missing}'
            return result
        
        # Step 7: Extract features
        try:
            features = feature_extractor.extract_features(company, linkedin_profiles, market_analysis)
        except ValueError as e:
            result['error'] = f'Feature extraction failed: {str(e)}'
            return result
        
        # Step 8: Score
        score_result = model.score_founder(features)
        
        result['score'] = score_result.get('score', 0.0)
        result['probability'] = score_result.get('probability', 0.0)
        result['top_features'] = score_result.get('top_features', [])
        
        # Step 9: Send email
        primary_linkedin_url = linkedin_urls[0] if linkedin_urls else None
        email_sent = email_service.send_founder_alert(company, score_result, primary_linkedin_url)
        result['email_sent'] = email_sent
        
        result['success'] = True
        return result
        
    except Exception as e:
        result['error'] = f'Processing error: {str(e)}'
        import traceback
        result['error'] += f'\n{traceback.format_exc()}'
        return result

