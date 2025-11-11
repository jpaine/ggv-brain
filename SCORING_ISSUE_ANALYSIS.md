# Scoring Issue Analysis

## Problem Identified

**Issue:** All companies are scoring very low (1.7-2.2/10) when they should potentially score higher.

**Root Cause:** The feature extraction is using **too many default/placeholder values** instead of real data.

## Current Feature Extraction Issues

### 1. Missing LinkedIn Data
- **Problem:** `linkedin_profile = None` for all companies
- **Impact:** Missing critical founder features:
  - `l_level` defaults to 2.0 (should be calculated from LinkedIn)
  - `founder_experience_score` defaults to 0.5 (should be calculated)
  - Phase 5 features (serial founder, LinkedIn network) all default to 0

### 2. Missing Market Data
- **Problem:** Using hardcoded defaults instead of real market analysis
- **Defaults being used:**
  - `timing_score = 0.7` (should be calculated from market timing)
  - `market_size_billion = 100.0` (should be actual TAM)
  - `cagr_percent = 25.0` (should be actual growth rate)
  - `competitor_count = 50` (should be actual count)

### 3. Missing Perplexity API Integration
- **Problem:** Comment says "require Perplexity API - use defaults for now"
- **Impact:** All market characteristics are generic defaults

## Why Scores Are Low

With mostly default values:
- `l_level = 2.0` (low founder level)
- Generic market scores (0.6-0.7)
- No serial founder signals
- No LinkedIn network data

The model correctly identifies these as **low-potential** because the features indicate:
- Early-stage founder (L2)
- Generic market positioning
- No prior experience signals
- No network advantages

## Solutions

### Option 1: Fix LinkedIn Extraction (HIGH PRIORITY)
- Currently: `linkedin_profile = None` for all companies
- Need: Extract actual LinkedIn URLs from Crunchbase founder data
- Impact: Will improve `l_level`, `founder_experience_score`, and Phase 5 features

### Option 2: Add Market Analysis API
- Integrate Perplexity API for real market data
- Or use Crunchbase API for competitor/market data
- Impact: Will improve market-related features

### Option 3: Accept Low Scores (If Companies Are Genuinely Low)
- If companies are truly early-stage with minimal signals, low scores are correct
- Model is working as designed - it's identifying low-potential founders
- Only email when score >= 8.0 (which we've now fixed)

## Current Workflow Status

✅ **Fixed:** Only emails for scores >= 8.0  
✅ **Fixed:** Email template conditional based on score  
⚠️ **Issue:** Feature extraction using too many defaults  
⚠️ **Issue:** LinkedIn profile extraction not implemented  

## Next Steps

1. **Implement LinkedIn extraction** from Crunchbase founder data
2. **Add market analysis** (Perplexity API or Crunchbase)
3. **Test with real data** to see if scores improve
4. **Monitor** - if scores remain low, companies may genuinely be low-potential

