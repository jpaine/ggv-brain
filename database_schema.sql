-- GGV Brain Workflow 1 Database Schema
-- PostgreSQL schema for tracking scored companies

CREATE TABLE IF NOT EXISTS scored_companies (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL UNIQUE,
    crunchbase_url VARCHAR(500),
    crunchbase_identifier VARCHAR(255),
    founded_year INTEGER,
    description TEXT,
    website VARCHAR(500),
    linkedin_url VARCHAR(500),
    v11_5_score FLOAT NOT NULL,
    predicted_probability FLOAT,
    features JSONB,
    top_features JSONB,
    scan_date DATE DEFAULT CURRENT_DATE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    emailed BOOLEAN DEFAULT FALSE,
    email_sent_at TIMESTAMP
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_score ON scored_companies(v11_5_score DESC);
CREATE INDEX IF NOT EXISTS idx_created_at ON scored_companies(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_emailed ON scored_companies(emailed);
CREATE INDEX IF NOT EXISTS idx_scan_date ON scored_companies(scan_date DESC);
CREATE INDEX IF NOT EXISTS idx_founded_year ON scored_companies(founded_year DESC);

-- Table for tracking daily scan history
CREATE TABLE IF NOT EXISTS scan_history (
    id SERIAL PRIMARY KEY,
    scan_date DATE NOT NULL UNIQUE,
    companies_found INTEGER DEFAULT 0,
    companies_scored INTEGER DEFAULT 0,
    high_scores_count INTEGER DEFAULT 0,
    emails_sent INTEGER DEFAULT 0,
    scan_duration_seconds INTEGER,
    status VARCHAR(50) DEFAULT 'success',
    error_message TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_scan_history_date ON scan_history(scan_date DESC);

-- Comments for documentation
COMMENT ON TABLE scored_companies IS 'All companies scored by V11.5 model from Crunchbase monitoring';
COMMENT ON COLUMN scored_companies.v11_5_score IS 'V11.5 model score (0-10), threshold >= 8.0 for alerts';
COMMENT ON COLUMN scored_companies.predicted_probability IS 'Raw probability (0-1) from model';
COMMENT ON COLUMN scored_companies.top_features IS 'Top 5 features contributing to score';
COMMENT ON COLUMN scored_companies.emailed IS 'Whether email alert was sent for this company';

COMMENT ON TABLE scan_history IS 'Daily scan execution history and metrics';

