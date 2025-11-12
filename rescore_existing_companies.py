#!/usr/bin/env python3
"""
Re-score Existing Companies with Real Data
==========================================
Re-score all companies in the database using real LinkedIn and Perplexity data
to compare old scores (with defaults) vs new scores (with real data).
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
from datetime import datetime
from typing import Dict, List, Optional
import logging
import asyncio
import aiohttp
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
DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://ggv_brain_user:OiKg0vH65qr6Mhjht8gxz950TmLzCyO2@dpg-d47dss6r433s739f3lo0-a.oregon-postgres.render.com/ggv_brain')

# Import classes from workflow
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from workflow_1_crunchbase_daily_monitor import (
    V11_5_Model, CrunchbaseScanner, EnrichlayerService, 
    PerplexityMarketService, FeatureExtractor
)

# V11.5 feature order (29 features total)
FEATURE_ORDER = [
    'l_level', 'estimated_age', 'founder_experience_score',
    'timing_score', 'market_size_billion', 'cagr_percent', 
    'competitor_count', 'market_maturity_stage',
    'confidence_score', 'confidence_score_market', 'geographic_advantage',
    'description_sentiment', 'description_complexity', 'about_quality',
    'sector_keyword_score', 'founder_market_fit',
    'market_saturation_score', 'differentiation_score',
    'momentum_3m', 'timing_score_trend', 'competitor_growth_rate',
    'is_serial_founder', 'founder_experience_years',
    'linkedin_industry_connections_ratio', 'linkedin_influencer_connections',
    'linkedin_2nd_degree_reach', 'barrier_to_entry_score',
    'pitch_clarity_score', 'pitch_specificity_score',
    'vision_realism_score', 'narrative_coherence_score'
]


def load_companies_from_database():
    """Load all companies from the database."""
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        
        query = """
            SELECT name, crunchbase_identifier, description, website, founded_year, v11_5_score
            FROM scored_companies
            ORDER BY created_at DESC
        """
        
        cursor.execute(query)
        rows = cursor.fetchall()
        
        companies = []
        for row in rows:
            companies.append({
                'name': row[0],
                'identifier': row[1] or '',
                'description': row[2] or '',
                'website': row[3] or '',
                'founded_year': row[4],
                'old_score': float(row[5]) if row[5] else 0.0
            })
        
        cursor.close()
        conn.close()
        
        logger.info(f"✅ Loaded {len(companies)} companies from database")
        return companies
    
    except Exception as e:
        logger.error(f"Error loading companies from database: {e}")
        return []


def get_company_founders_from_crunchbase(identifier: str) -> List[str]:
    """Get founder identifiers for a company from Crunchbase."""
    if not identifier:
        return []
    
    try:
        base_url = "https://api.crunchbase.com/api/v4"
        url = f"{base_url}/entities/organizations/{identifier}"
        
        headers = {
            'accept': 'application/json',
            'X-cb-user-key': CRUNCHBASE_API_KEY
        }
        
        # Request founder card
        params = {
            'card_ids': 'founders'
        }
        
        response = requests.get(url, headers=headers, params=params, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            cards = data.get('cards', {})
            founders_list = cards.get('founders', [])
            
            founder_ids = []
            for founder in founders_list:
                identifier_obj = founder.get('identifier', {})
                founder_id = identifier_obj.get('value', '') if isinstance(identifier_obj, dict) else ''
                if founder_id:
                    founder_ids.append(founder_id)
            
            return founder_ids
        else:
            logger.warning(f"Could not fetch company {identifier}: {response.status_code}")
            return []
    
    except Exception as e:
        logger.error(f"Error fetching founders for {identifier}: {e}")
        return []


def main():
    """Main re-scoring workflow."""
    logger.info("="*80)
    logger.info("RE-SCORING EXISTING COMPANIES WITH REAL DATA")
    logger.info("="*80)
    logger.info("")
    
    # Initialize services
    model = V11_5_Model()
    crunchbase = CrunchbaseScanner(CRUNCHBASE_API_KEY)
    enrichlayer = EnrichlayerService(ENRICHLAYER_API_KEY)
    perplexity = PerplexityMarketService(PERPLEXITY_API_KEY)
    feature_extractor = FeatureExtractor()
    
    # Load companies from database
    companies = load_companies_from_database()
    
    if not companies:
        logger.error("No companies found in database")
        return
    
    logger.info(f"Re-scoring {len(companies)} companies...")
    logger.info("")
    
    results = []
    successful = 0
    failed = 0
    
    for i, company in enumerate(companies, 1):
        logger.info(f"{i}/{len(companies)}: {company.get('name')}")
        logger.info(f"  Old Score: {company.get('old_score', 0):.2f}/10")
        
        try:
            # Step 1: Get founder identifiers from company
            identifier = company.get('identifier', '')
            if not identifier:
                logger.warning(f"  ⚠️ No Crunchbase identifier - skipping")
                failed += 1
                continue
            
            founder_identifiers = get_company_founders_from_crunchbase(identifier)
            
            if not founder_identifiers:
                logger.warning(f"  ⚠️ No founders found - skipping")
                failed += 1
                continue
            
            # Step 2: Get founder details (including LinkedIn URLs)
            founders = crunchbase.get_founder_details(founder_identifiers)
            
            if not founders:
                logger.warning(f"  ⚠️ No founder details available - skipping")
                failed += 1
                continue
            
            # Step 3: Get LinkedIn profiles
            linkedin_profiles = []
            for founder in founders:
                linkedin_url = founder.get('linkedin_url')
                if linkedin_url:
                    profile = enrichlayer.get_linkedin_profile(linkedin_url)
                    if profile:
                        linkedin_profiles.append(profile)
            
            if not linkedin_profiles:
                logger.warning(f"  ⚠️ No LinkedIn profiles available - skipping")
                failed += 1
                continue
            
            logger.info(f"  ✅ Found {len(linkedin_profiles)} LinkedIn profile(s)")
            
            # Step 4: Get market analysis from Perplexity
            market_analysis = asyncio.run(perplexity.get_market_analysis(
                company.get('name', ''),
                company.get('description', ''),
                ['Technology']  # Default category
            ))
            
            if not market_analysis:
                logger.error(f"  ❌ Market analysis failed - skipping (no defaults)")
                failed += 1
                continue
            
            logger.info(f"  ✅ Market: ${market_analysis.get('market_size_billion', 0):.2f}B, {market_analysis.get('cagr_percent', 0):.1f}% CAGR")
            
            # Step 5: Extract features with real data
            features = feature_extractor.extract_features(company, linkedin_profiles, market_analysis)
            
            # Step 6: Re-score
            score_result = model.score_founder(features)
            new_score = score_result.get('score', 0)
            old_score = company.get('old_score', 0)
            
            score_change = new_score - old_score
            score_change_pct = (score_change / old_score * 100) if old_score > 0 else 0
            
            logger.info(f"  New Score: {new_score:.2f}/10")
            logger.info(f"  Change: {score_change:+.2f} ({score_change_pct:+.1f}%)")
            
            results.append({
                'name': company.get('name'),
                'old_score': old_score,
                'new_score': new_score,
                'score_change': score_change,
                'score_change_pct': score_change_pct,
                'probability': score_result.get('probability', 0),
                'market_size_billion': market_analysis.get('market_size_billion', 0),
                'cagr_percent': market_analysis.get('cagr_percent', 0),
                'founders_count': len(linkedin_profiles)
            })
            
            successful += 1
            
        except Exception as e:
            logger.error(f"  ❌ Error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            failed += 1
            continue
        
        logger.info("")
    
    # Summary
    logger.info("="*80)
    logger.info("RE-SCORING SUMMARY")
    logger.info("="*80)
    logger.info(f"Total companies: {len(companies)}")
    logger.info(f"Successfully re-scored: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info("")
    
    if results:
        df = pd.DataFrame(results)
        
        logger.info("Score Changes:")
        logger.info(f"  Average old score: {df['old_score'].mean():.2f}/10")
        logger.info(f"  Average new score: {df['new_score'].mean():.2f}/10")
        logger.info(f"  Average change: {df['score_change'].mean():+.2f}")
        logger.info(f"  Max improvement: {df['score_change'].max():+.2f}")
        logger.info(f"  Max decline: {df['score_change'].min():+.2f}")
        logger.info("")
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f'rescore_results_{timestamp}.json'
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"✅ Results saved to {output_file}")
        
        # Save CSV
        csv_file = f'rescore_results_{timestamp}.csv'
        df.to_csv(csv_file, index=False)
        logger.info(f"✅ CSV saved to {csv_file}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

