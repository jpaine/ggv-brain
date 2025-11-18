# GGV Brain Company Scorer - Vercel Frontend

A Next.js web application deployed on Vercel that allows users to input a Crunchbase URL and receive a detailed company/founder score via email and on-page display.

## Features

- **Simple Web Interface**: Enter a Crunchbase URL to analyze a company
- **Real-time Processing**: Full pipeline execution (30-60 seconds)
- **Email Notifications**: Automatic email with detailed results
- **Score Display**: On-page results showing score, probability, and top features

## Architecture

- **Frontend**: Next.js 14 with TypeScript and Tailwind CSS
- **Backend**: Python serverless function on Vercel
- **Processing**: Reuses existing `workflow_1_crunchbase_daily_monitor.py` logic
- **Model**: V11.7.1 early-stage founder scoring model

## Setup Instructions

### 1. Install Dependencies

```bash
cd vercel-frontend
npm install
```

### 2. Environment Variables

Create a `.env.local` file (not committed) or set in Vercel dashboard:

```
CRUNCHBASE_API_KEY=your_crunchbase_api_key
ENRICHLAYER_API_KEY=your_enrichlayer_api_key
PERPLEXITY_API_KEY=your_perplexity_api_key
RESEND_API_KEY=your_resend_api_key
RECIPIENT_EMAIL=jeffrey.paine@gmail.com
FROM_EMAIL=onboarding@resend.dev
```

### 3. Model Files

Model files are already copied to `public/models/`:
- `v11_7_1_fixed_distribution_model_20251114_214215.pkl`
- `v11_7_1_fixed_distribution_scaler_20251114_214215.pkl`

### 4. Local Development

```bash
npm run dev
```

Visit `http://localhost:3000`

### 5. Deploy to Vercel

#### Option A: Using Vercel CLI

```bash
# Install Vercel CLI
npm i -g vercel

# Login
vercel login

# Deploy
cd vercel-frontend
vercel --prod
```

#### Option B: Using Vercel Dashboard

1. Push code to GitHub
2. Import project in Vercel dashboard
3. Configure environment variables in Vercel dashboard
4. Deploy

### 6. Configure Environment Variables in Vercel

In Vercel dashboard → Project Settings → Environment Variables:

- `CRUNCHBASE_API_KEY`
- `ENRICHLAYER_API_KEY`
- `PERPLEXITY_API_KEY`
- `RESEND_API_KEY`
- `RECIPIENT_EMAIL`
- `FROM_EMAIL`

## Project Structure

```
vercel-frontend/
├── app/
│   ├── page.tsx              # Main page
│   ├── layout.tsx            # Root layout
│   ├── globals.css           # Global styles
│   └── api/
│       └── process/
│           └── route.py      # Python API endpoint
├── components/
│   └── CompanyForm.tsx       # Form component
├── lib/
│   └── scoring-service.py    # Scoring wrapper
├── public/
│   └── models/               # Model files
├── package.json
├── vercel.json              # Vercel configuration
└── requirements.txt          # Python dependencies
```

## API Endpoint

**POST** `/api/process`

Request body:
```json
{
  "url": "https://www.crunchbase.com/organization/company-name"
}
```

Response:
```json
{
  "success": true,
  "company_name": "Company Name",
  "score": 7.5,
  "probability": 0.75,
  "top_features": [...],
  "email_sent": true
}
```

## Notes

- Processing takes 30-60 seconds (multiple API calls)
- Vercel Pro plan required for 60s timeout (Hobby plan has 10s limit)
- Model files must be accessible to Python serverless function
- All features use real data (no mock/defaults)

## Troubleshooting

1. **Model not found**: Ensure model files are in `public/models/` or adjust paths in `scoring-service.py`
2. **Import errors**: Ensure `workflow_1_crunchbase_daily_monitor.py` and dependencies are accessible
3. **Timeout errors**: Upgrade to Vercel Pro for 60s timeout
4. **CORS errors**: CORS headers are already configured in the API route

