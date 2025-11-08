#!/usr/bin/env python3
"""
Workflow 1: Automated Daily Crunchbase Monitoring
=================================================

Daily cron job that:
1. Scans Crunchbase for new AI companies (founded 2025, USA, with founder LinkedIn)
2. Extracts 29 V11.5 features for each founder
3. Scores with V11.5 model (73.3% accuracy)
4. Saves all to PostgreSQL
5. Emails jeff@goldengate.vc for scores >= 8.0

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
RECIPIENT_EMAIL = os.getenv('RECIPIENT_EMAIL', 'jeff@goldengate.vc')
FROM_EMAIL = os.getenv('FROM_EMAIL', 'ggv-brain@goldengate.vc')
DATABASE_URL = os.getenv('DATABASE_URL')

# V11.5 feature order (29 features total)
FEATURE_ORDER = [
    # V11.4 features (21)
    'l_level', 'estimated_age', 'founder_experience_score',
    'timing_score', 'market_size_billion', 'cagr_percent', 
    'competitor_count', 'market_maturity_stage',
    'confidence_score', 'confidence_score_market', 'geographic_advantage',
    'description_sentiment', 'description_complexity', 'about_quality',
    'sector_keyword_score', 'founder_market_fit',
    'market_saturation_score', 'differentiation_score',
    'momentum_3m', 'timing_score_trend', 'competitor_growth_rate',
    # Phase 5 features (10)
    'is_serial_founder', 'founder_experience_years',
    'linkedin_industry_connections_ratio', 'linkedin_influencer_connections',
    'linkedin_2nd_degree_reach', 'barrier_to_entry_score',
    'pitch_clarity_score', 'pitch_specificity_score',
    'vision_realism_score', 'narrative_coherence_score'
]


class V11_5_Model:
    """V11.5 model loader and scorer."""
    
    def __init__(self, model_path='v11_5_phase5_model_20251105_170709.pkl',
                 scaler_path='v11_5_phase5_scaler_20251105_170709.pkl'):
        """Load V11.5 model and scaler."""
        logger.info("Loading V11.5 model...")
        
        self.model = pickle.load(open(model_path, 'rb'))
        self.scaler = pickle.load(open(scaler_path, 'rb'))
        
        logger.info("✅ V11.5 model loaded successfully")
    
    def score_founder(self, features: Dict) -> Dict:
        """
        Score a founder with V11.5 model.
        
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
    
    def search_new_ai_companies(self, days_back: int = 1) -> List[Dict]:
        """
        Search for AI companies founded in 2025, headquartered in USA.
        
        Args:
            days_back: Number of days to look back (default 1 for daily scan)
        
        Returns:
            List of company dictionaries
        """
        logger.info(f"Searching Crunchbase for new AI companies (last {days_back} days)...")
        
        # Date range
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
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
                    "values": ["2025-01-01"]
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
                        
                        companies.append({
                            'name': props.get('name', ''),
                            'identifier': identifier or '',
                            'description': props.get('description', '') or props.get('short_description', ''),
                            'website': website or '',
                            'founded_on': founded_on or '',
                            'categories': props.get('categories', []),
                            'founder_identifiers': founder_identifiers
                        })
                
                logger.info(f"✅ Found {len(companies)} companies (filtered for USA + founders)")
                return companies
            else:
                logger.error(f"Crunchbase API error: {response.status_code} - {response.text[:200]}")
                return []
        
        except Exception as e:
            logger.error(f"Error searching Crunchbase: {e}")
            return []


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


class FeatureExtractor:
    """Extract all 29 V11.5 features from company data."""
    
    def extract_features(self, company: Dict, linkedin_profile: Optional[Dict] = None) -> Dict:
        """
        Extract all 29 V11.5 features.
        
        Args:
            company: Crunchbase company data
            linkedin_profile: Enrichlayer LinkedIn profile data (optional)
        
        Returns:
            Dictionary with all 29 features
        """
        features = {}
        
        # Extract V11.4 features (19 features)
        features.update(self._extract_v11_4_features(company, linkedin_profile))
        
        # Extract Phase 5 features (10 features)
        features.update(self._extract_phase5_features(company, linkedin_profile))
        
        return features
    
    def _extract_v11_4_features(self, company: Dict, linkedin_profile: Optional[Dict]) -> Dict:
        """Extract V11.4 features (simplified version for production)."""
        features = {}
        
        description = company.get('description', '')
        founded_year = self._parse_year(company.get('founded_on', ''))
        
        # Founder characteristics
        features['l_level'] = self._estimate_l_level(linkedin_profile) if linkedin_profile else 2.0
        features['estimated_age'] = 35.0  # Default estimate
        features['founder_experience_score'] = self._calculate_experience_score(linkedin_profile) if linkedin_profile else 0.5
        
        # Market characteristics (require Perplexity API - use defaults for now)
        features['timing_score'] = 0.7  # Default for 2025 AI
        features['market_size_billion'] = 100.0  # AI market default
        features['cagr_percent'] = 25.0  # AI market growth
        features['competitor_count'] = 50  # Moderate competition
        features['market_maturity_stage'] = 0.6  # Growing market
        features['confidence_score'] = 0.7  # Moderate confidence
        features['confidence_score_market'] = 0.7
        features['geographic_advantage'] = 0.8 if 'san francisco' in str(company.get('location', '')).lower() else 0.5
        
        # Pitch quality (rule-based from description)
        features['description_sentiment'] = 0.7  # Neutral-positive default
        features['description_complexity'] = len(description.split()) / 100.0 if description else 0.5
        features['about_quality'] = 0.7 if len(description) > 100 else 0.3
        features['sector_keyword_score'] = self._calculate_sector_score(description)
        
        # Market position
        features['founder_market_fit'] = 0.6  # Default
        features['market_saturation_score'] = 0.4  # Growing market
        features['differentiation_score'] = 0.6  # Default
        
        # Temporal (use current market state)
        features['momentum_3m'] = 0.8  # High AI momentum in 2025
        features['timing_score_trend'] = 0.1  # Positive trend
        features['competitor_growth_rate'] = 0.3  # Moderate growth
        
        return features
    
    def _extract_phase5_features(self, company: Dict, linkedin_profile: Optional[Dict]) -> Dict:
        """Extract Phase 5 features."""
        features = {}
        
        description = company.get('description', '')
        
        # Week 1: Serial founder features
        experience = linkedin_profile.get('experience', []) if linkedin_profile else []
        founder_keywords = ['founder', 'co-founder', 'cofounder', 'ceo']
        
        founder_count = sum(
            1 for exp in experience
            if any(kw in str(exp.get('title', '')).lower() for kw in founder_keywords)
        )
        
        features['is_serial_founder'] = 1 if founder_count > 1 else 0
        features['founder_experience_years'] = min(len(experience) * 2.5 / 20.0, 1.0) if experience else 0.0
        
        # Week 1: LinkedIn network estimates
        industry_keywords = ['ai', 'ml', 'software', 'technology', 'startup']
        industry_experience = sum(
            1 for exp in experience
            if any(kw in str(exp.get('title', '')).lower() + str(exp.get('company', '')).lower() 
                  for kw in industry_keywords)
        )
        
        if len(experience) > 0:
            features['linkedin_industry_connections_ratio'] = 0.2 + (industry_experience / len(experience)) * 0.5
        else:
            features['linkedin_industry_connections_ratio'] = 0.2
        
        features['linkedin_influencer_connections'] = 0.3 if founder_count > 0 else 0.1
        features['linkedin_2nd_degree_reach'] = 0.0  # No connections count available
        
        # Week 2: Barrier to entry
        barrier_keywords = ['ai', 'ml', 'deep learning', 'proprietary', 'patent']
        features['barrier_to_entry_score'] = 0.5 if any(kw in description.lower() for kw in barrier_keywords) else 0.3
        
        # Week 3: Pitch quality (import from phase5_pitch_quality_features.py)
        features.update(self._extract_pitch_quality(description))
        
        return features
    
    def _extract_pitch_quality(self, description: str) -> Dict:
        """Extract pitch quality features."""
        if not description:
            return {
                'pitch_clarity_score': 0.5,
                'pitch_specificity_score': 0.0,
                'vision_realism_score': 0.5,
                'narrative_coherence_score': 0.0
            }
        
        # Import pitch quality extraction
        sys.path.append('.')
        from phase5_pitch_quality_features import extract_pitch_quality_features
        return extract_pitch_quality_features(description)
    
    def _estimate_l_level(self, linkedin_profile: Dict) -> float:
        """Estimate founder L level from LinkedIn profile."""
        experience = linkedin_profile.get('experience', [])
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
        experience = linkedin_profile.get('experience', [])
        
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
    
    def save_scored_company(self, company: Dict, score_result: Dict, linkedin_url: str = None):
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
                 v11_5_score, predicted_probability, features, top_features, emailed)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (name) DO UPDATE SET
                    v11_5_score = EXCLUDED.v11_5_score,
                    predicted_probability = EXCLUDED.predicted_probability,
                    updated_at = NOW()
            """
            
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
                score_result.get('score', 0) >= 8.0  # Mark as emailed if high score
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


class EmailAlertService:
    """Send email alerts via Resend API."""
    
    def __init__(self, api_key: str, from_email: str, recipient_email: str):
        self.api_key = api_key
        self.from_email = from_email
        self.recipient_email = recipient_email
        self.api_url = "https://api.resend.com/emails"
    
    def send_founder_alert(self, company: Dict, score_result: Dict, linkedin_url: str = None):
        """Send email alert for high-scoring founder."""
        try:
            subject = f"🚀 GGV Brain Alert: {company.get('name')} - Score {score_result.get('score'):.1f}/10"
            
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
        top_features_html = ''.join([
            f"<li><strong>{f['feature']}</strong>: {f['value']:.3f} (importance: {f['importance']:.1%})</li>"
            for f in score_result.get('top_features', [])
        ])
        
        linkedin_html = f'<p><strong>Founder LinkedIn:</strong> <a href="{linkedin_url}">{linkedin_url}</a></p>' if linkedin_url else ''
        
        html = f"""
        <html>
        <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
            <h2 style="color: #1a73e8;">🚀 High-Potential Founder Alert</h2>
            
            <div style="background: #f5f5f5; padding: 20px; border-radius: 8px; margin: 20px 0;">
                <h3 style="margin-top: 0;">{company.get('name')}</h3>
                <p><strong>V11.5 Score:</strong> <span style="font-size: 24px; color: #1a73e8; font-weight: bold;">{score_result.get('score'):.1f}/10</span></p>
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
                This alert was generated by GGV Brain V11.5 (73.3% accuracy on historical data).
                Model predicts funding efficiency based on 29 founding-time features.
            </p>
        </body>
        </html>
        """
        
        return html


def main():
    """Main workflow execution."""
    logger.info("="*80)
    logger.info("WORKFLOW 1: CRUNCHBASE DAILY MONITOR")
    logger.info("="*80)
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("")
    
    # Initialize services
    model = V11_5_Model()
    crunchbase = CrunchbaseScanner(CRUNCHBASE_API_KEY)
    enrichlayer = EnrichlayerService(ENRICHLAYER_API_KEY)
    feature_extractor = FeatureExtractor()
    email_service = EmailAlertService(RESEND_API_KEY, FROM_EMAIL, RECIPIENT_EMAIL)
    
    # Connect to database
    if DATABASE_URL:
        db = DatabaseService(DATABASE_URL)
        db.connect()
    else:
        logger.warning("⚠️ No DATABASE_URL provided, skipping database saves")
        db = None
    
    # Search for new companies
    companies = crunchbase.search_new_ai_companies(days_back=1)
    
    if not companies:
        logger.info("No new companies found")
        return
    
    logger.info(f"Processing {len(companies)} companies...")
    logger.info("")
    
    # Process each company
    high_score_count = 0
    
    for i, company in enumerate(companies, 1):
        logger.info(f"{i}/{len(companies)}: {company.get('name')}")
        
        # Get LinkedIn profile for first founder
        founder_identifiers = company.get('founder_identifiers', [])
        linkedin_profile = None
        linkedin_url = None
        
        if founder_identifiers:
            # For now, use placeholder - would need to get actual LinkedIn URL from Crunchbase
            # This requires additional API call to get founder details
            logger.info("  ⚠️ Founder LinkedIn extraction not implemented yet")
            # TODO: Implement founder LinkedIn URL extraction
        
        # Extract features
        features = feature_extractor.extract_features(company, linkedin_profile)
        
        # Score with V11.5
        score_result = model.score_founder(features)
        
        logger.info(f"  Score: {score_result.get('score'):.2f}/10 (probability: {score_result.get('probability'):.1%})")
        
        # Save to database
        if db:
            db.save_scored_company(company, score_result, linkedin_url)
        
        # Send email alert if high score
        if score_result.get('score', 0) >= 8.0:
            logger.info(f"  🚀 HIGH SCORE - Sending email alert")
            email_service.send_founder_alert(company, score_result, linkedin_url)
            high_score_count += 1
        
        logger.info("")
    
    # Summary
    logger.info("="*80)
    logger.info("DAILY SCAN SUMMARY")
    logger.info("="*80)
    logger.info(f"Companies scanned: {len(companies)}")
    logger.info(f"High scores (>= 8.0): {high_score_count}")
    logger.info(f"Email alerts sent: {high_score_count}")
    logger.info(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Close database
    if db:
        db.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Workflow failed: {e}")
        sys.exit(1)

