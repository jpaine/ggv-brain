# Deployment Guide for Vercel Frontend

## Quick Start

1. **Install dependencies:**
   ```bash
   cd vercel-frontend
   npm install
   ```

2. **Set environment variables in Vercel:**
   - Go to Vercel Dashboard → Your Project → Settings → Environment Variables
   - Add all variables from `.env.example`

3. **Deploy:**
   ```bash
   vercel --prod
   ```

## Important Notes

### Model Files Location

The model files need to be accessible to the Python serverless function. In Vercel:

- **Option 1 (Recommended)**: Store models in Vercel Blob Storage or S3 and download on cold start
- **Option 2**: Include models in deployment (may hit size limits)
- **Option 3**: Use a separate model service/API

Current setup expects models at:
- `public/models/v11_7_1_fixed_distribution_model_20251114_214215.pkl`
- `public/models/v11_7_1_fixed_distribution_scaler_20251114_214215.pkl`

### Python Dependencies

The `requirements.txt` file lists Python dependencies. Vercel will automatically install these for the Python serverless function.

### Timeout Limits

- **Hobby Plan**: 10 seconds (insufficient for processing)
- **Pro Plan**: 60 seconds (sufficient for most cases)
- **Enterprise**: Up to 300 seconds

**Recommendation**: Use Vercel Pro plan for this application.

### Import Paths

The Python code imports from the main BRAIN project:
- `workflow_1_crunchbase_daily_monitor.py` - Main workflow
- `phase2_real_feature_calculations.py` - Feature calculations

These files must be accessible. Options:
1. Copy necessary files to `vercel-frontend/` directory
2. Use a monorepo structure
3. Package as a Python module

## Troubleshooting

### Import Errors

If you see import errors for `workflow_1_crunchbase_daily_monitor` or `phase2_real_feature_calculations`:

1. Ensure these files are in the project root or accessible via `sys.path`
2. Check that the path resolution in `scoring-service.py` is correct
3. Consider copying these files to `vercel-frontend/lib/` if needed

### Model Not Found

If models aren't found:
1. Check that model files are in `public/models/`
2. Verify file paths in `scoring-service.py`
3. Consider using absolute paths or environment variables for model location

### Timeout Errors

If processing times out:
1. Upgrade to Vercel Pro (60s timeout)
2. Optimize API calls (parallelize where possible)
3. Consider moving to async processing with webhooks

## Next Steps

1. Test locally with `npm run dev`
2. Deploy to Vercel staging
3. Test with a real Crunchbase URL
4. Monitor logs in Vercel dashboard
5. Deploy to production

