#!/usr/bin/env python3
"""
Test Single Company Workflow
=============================
Find 1 USA-headquartered AI company founded in last 3 months,
verify it has founders with LinkedIn data, then run full extraction and scoring.
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
PERPLEXITY_API_KEY = os.getenv('PERPLEXITY_API_KEY', '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')
RESEND_API_KEY = os.getenv('RESEND_API_KEY', 're_4D7gm8JB_4An2NRYKxWT1VuFXabqPnhD3')
RECIPIENT_EMAIL = os.getenv('RECIPIENT_EMAIL', 'jeffrey.paine@gmail.com')
FROM_EMAIL = os.getenv('FROM_EMAIL', 'onboarding@resend.dev')
DATABASE_URL = os.getenv('DATABASE_URL')

# Import classes from workflow
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from workflow_1_crunchbase_daily_monitor import (
    V11_7_1_Model, CrunchbaseScanner, EnrichlayerService, 
    PerplexityMarketService, FeatureExtractor, EmailAlertService
)

def check_company_has_founders_with_linkedin(company: Dict, crunchbase: CrunchbaseScanner) -> tuple[bool, List[Dict], List[str]]:
    """
    Check if company has founders with LinkedIn data.
    
    Returns:
        (has_founders, founders_list, linkedin_urls)
    """
    logger.info(f"  Checking founders for {company.get('name')}...")
    
    company_uuid = company.get('uuid', '')
    founders = []
    founder_identifiers = []
    linkedin_urls = []
    
    if company_uuid:
        # Get founder entity IDs and data from company entity endpoint
        founder_identifiers, founders_from_card = crunchbase.get_founder_entity_ids_from_company(company_uuid)
        if founders_from_card:
            founders = founders_from_card
        elif founder_identifiers:
            founders = crunchbase.get_founder_details(founder_identifiers)
    
    # Fallback: use founder_identifiers from search if UUID method failed
    if not founders:
        founder_identifiers = company.get('founder_identifiers', [])
        founders = crunchbase.get_founder_details(founder_identifiers) if founder_identifiers else []
    
    if not founders:
        logger.warning(f"    ❌ No founders found")
        return False, [], []
    
    logger.info(f"    ✅ Found {len(founders)} founder(s)")
    
    # Check for LinkedIn URLs
    for founder in founders:
        linkedin_url = founder.get('linkedin_url')
        if linkedin_url:
            linkedin_urls.append(linkedin_url)
            logger.info(f"    ✅ Founder {founder.get('name', 'Unknown')}: {linkedin_url}")
        else:
            logger.warning(f"    ⚠️  Founder {founder.get('name', 'Unknown')}: No LinkedIn URL")
    
    if not linkedin_urls:
        logger.warning(f"    ❌ No founders with LinkedIn URLs")
        return False, founders, []
    
    logger.info(f"    ✅ Found {len(linkedin_urls)} founder(s) with LinkedIn URLs")
    return True, founders, linkedin_urls

def test_single_company():
    """Test workflow on a single company."""
    logger.info("="*80)
    logger.info("TEST SINGLE COMPANY WORKFLOW")
    logger.info("="*80)
    logger.info("")
    
    # Initialize services
    logger.info("Initializing services...")
    model = V11_7_1_Model()
    crunchbase = CrunchbaseScanner(CRUNCHBASE_API_KEY)
    enrichlayer = EnrichlayerService(ENRICHLAYER_API_KEY)
    perplexity = PerplexityMarketService(PERPLEXITY_API_KEY)
    feature_extractor = FeatureExtractor()
    email_service = EmailAlertService(RESEND_API_KEY, FROM_EMAIL, RECIPIENT_EMAIL)
    logger.info("✅ Services initialized")
    logger.info("")
    
    # Search for companies founded in last 3 months
    logger.info("Searching for USA-headquartered AI companies (last 3 months)...")
    companies = crunchbase.search_new_ai_companies(days_back=90)  # 3 months
    
    if not companies:
        logger.error("❌ No companies found")
        return
    
    logger.info(f"✅ Found {len(companies)} companies")
    logger.info("")
    
    # Check for founders with LinkedIn (companies are already filtered for USA)
    logger.info("Checking for companies with founders and LinkedIn data...")
    logger.info("(Companies are already filtered for USA headquarters)")
    logger.info("-" * 80)
    
    selected_company = None
    selected_founders = []
    selected_linkedin_urls = []
    
    for i, company in enumerate(companies, 1):
        logger.info(f"\n{i}/{len(companies)}: {company.get('name')}")
        logger.info(f"  Founded: {company.get('founded_on', 'N/A')}")
        logger.info(f"  ✅ USA headquarters (pre-filtered)")
        
        # Check for founders with LinkedIn
        has_founders, founders, linkedin_urls = check_company_has_founders_with_linkedin(company, crunchbase)
        
        if has_founders and linkedin_urls:
            selected_company = company
            selected_founders = founders
            selected_linkedin_urls = linkedin_urls
            logger.info("")
            logger.info("="*80)
            logger.info(f"✅ SELECTED COMPANY: {company.get('name')}")
            logger.info("="*80)
            break
        else:
            logger.warning(f"  ❌ Skipping - no founders with LinkedIn")
    
    if not selected_company:
        logger.error("❌ No suitable company found with founders and LinkedIn data")
        return
    
    # Now process the selected company
    logger.info("")
    logger.info("="*80)
    logger.info("PROCESSING SELECTED COMPANY")
    logger.info("="*80)
    logger.info("")
    
    try:
        # Step 1: Get LinkedIn profiles
        logger.info("Step 1: Fetching LinkedIn profiles...")
        linkedin_profiles = []
        for linkedin_url in selected_linkedin_urls:
            logger.info(f"  Fetching: {linkedin_url}")
            profile = enrichlayer.get_linkedin_profile(linkedin_url)
            if profile:
                linkedin_profiles.append(profile)
                logger.info(f"    ✅ Profile fetched")
            else:
                logger.warning(f"    ⚠️  Could not fetch profile")
        
        if not linkedin_profiles:
            logger.error("❌ No LinkedIn profiles fetched - cannot proceed")
            return
        
        logger.info(f"✅ Fetched {len(linkedin_profiles)} LinkedIn profile(s)")
        logger.info("")
        
        # Step 2: Get market analysis
        logger.info("Step 2: Fetching market analysis...")
        categories = selected_company.get('categories', [])
        category_strings = [cat.get('value', '') if isinstance(cat, dict) else str(cat) for cat in categories]
        
        market_analysis = asyncio.run(perplexity.get_market_analysis(
            selected_company.get('name', ''),
            selected_company.get('description', ''),
            category_strings
        ))
        
        if not market_analysis:
            logger.error("❌ Market analysis failed - cannot proceed")
            return
        
        logger.info(f"✅ Market analysis: ${market_analysis.get('market_size_billion', 0):.2f}B, {market_analysis.get('cagr_percent', 0):.1f}% CAGR")
        logger.info("")
        
        # Step 3: Extract features
        logger.info("Step 3: Extracting features...")
        features = feature_extractor.extract_features(selected_company, linkedin_profiles, market_analysis)
        
        if not features:
            logger.error("❌ Feature extraction failed - cannot proceed")
            return
        
        logger.info(f"✅ Extracted {len(features)} features")
        logger.info("")
        
        # Step 4: Score
        logger.info("Step 4: Scoring with V11.7.1 model...")
        score_result = model.score_founder(features)
        score = score_result.get('score', 0)
        
        logger.info(f"✅ Score: {score:.2f}/10 (probability: {score_result.get('probability'):.1%})")
        logger.info("")
        
        # Step 5: Send email
        logger.info("Step 5: Sending email alert...")
        primary_linkedin_url = selected_linkedin_urls[0] if selected_linkedin_urls else None
        email_sent = email_service.send_founder_alert(selected_company, score_result, primary_linkedin_url)
        
        if email_sent:
            logger.info("✅ Email sent successfully!")
        else:
            logger.warning("⚠️  Email send failed")
        logger.info("")
        
        # Step 6: Save to database (if available)
        if DATABASE_URL:
            logger.info("Step 6: Saving to database...")
            try:
                from workflow_1_crunchbase_daily_monitor import DatabaseService
                db = DatabaseService(DATABASE_URL)
                db.connect()
                db.save_scored_company(selected_company, score_result, primary_linkedin_url, emailed=email_sent)
                db.close()
                logger.info("✅ Saved to database")
            except Exception as e:
                logger.warning(f"⚠️  Database save failed: {e}")
        else:
            logger.info("Step 6: Skipping database (DATABASE_URL not set)")
        logger.info("")
        
        # Summary
        logger.info("="*80)
        logger.info("TEST COMPLETE")
        logger.info("="*80)
        logger.info(f"Company: {selected_company.get('name')}")
        logger.info(f"Score: {score:.2f}/10")
        logger.info(f"Probability: {score_result.get('probability'):.1%}")
        logger.info(f"Email sent: {'Yes' if email_sent else 'No'}")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"❌ Error processing company: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    test_single_company()

