'use client';

import { useState } from 'react';

interface ProcessingResult {
  success: boolean;
  company_name?: string;
  score?: number;
  probability?: number;
  top_features?: Array<{
    feature: string;
    value: number;
    importance: number;
  }>;
  error?: string;
  email_sent?: boolean;
}

export default function CompanyForm() {
  const [url, setUrl] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ProcessingResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const validateUrl = (url: string): boolean => {
    return url.includes('crunchbase.com/organization/');
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setResult(null);
    
    if (!url.trim()) {
      setError('Please enter a Crunchbase URL');
      return;
    }
    
    if (!validateUrl(url)) {
      setError('Invalid Crunchbase URL format. Expected: https://www.crunchbase.com/organization/company-name');
      return;
    }
    
    setLoading(true);
    
    try {
      const response = await fetch('/api/process', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ url: url.trim() }),
      });
      
      const data: ProcessingResult = await response.json();
      
      if (data.success) {
        setResult(data);
      } else {
        setError(data.error || 'Processing failed');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Network error occurred');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-xl p-8">
      <form onSubmit={handleSubmit} className="space-y-6">
        <div>
          <label htmlFor="url" className="block text-sm font-medium text-gray-700 mb-2">
            Crunchbase URL
          </label>
          <input
            type="text"
            id="url"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            placeholder="https://www.crunchbase.com/organization/company-name"
            className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            disabled={loading}
          />
        </div>
        
        <button
          type="submit"
          disabled={loading || !url.trim()}
          className="w-full bg-blue-600 text-white py-3 px-6 rounded-lg font-semibold hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
        >
          {loading ? (
            <span className="flex items-center justify-center">
              <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              Processing...
            </span>
          ) : (
            'Process Company'
          )}
        </button>
      </form>

      {error && (
        <div className="mt-6 p-4 bg-red-50 border border-red-200 rounded-lg">
          <div className="flex">
            <div className="flex-shrink-0">
              <svg className="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
              </svg>
            </div>
            <div className="ml-3">
              <h3 className="text-sm font-medium text-red-800">Error</h3>
              <p className="mt-1 text-sm text-red-700">{error}</p>
            </div>
          </div>
        </div>
      )}

      {result && result.success && (
        <div className="mt-6 space-y-4">
          <div className="p-6 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg border border-blue-200">
            <h2 className="text-2xl font-bold text-gray-900 mb-4">{result.company_name}</h2>
            
            <div className="grid grid-cols-2 gap-4 mb-6">
              <div className="bg-white p-4 rounded-lg shadow">
                <div className="text-sm text-gray-600 mb-1">Score</div>
                <div className="text-3xl font-bold text-blue-600">
                  {result.score?.toFixed(1)}/10
                </div>
              </div>
              <div className="bg-white p-4 rounded-lg shadow">
                <div className="text-sm text-gray-600 mb-1">Probability</div>
                <div className="text-3xl font-bold text-indigo-600">
                  {(result.probability! * 100).toFixed(1)}%
                </div>
              </div>
            </div>

            {result.email_sent && (
              <div className="mb-4 p-3 bg-green-50 border border-green-200 rounded-lg">
                <p className="text-sm text-green-800">
                  ✓ Email sent successfully to configured recipient
                </p>
              </div>
            )}

            {result.top_features && result.top_features.length > 0 && (
              <div>
                <h3 className="text-lg font-semibold text-gray-900 mb-3">Top Contributing Features</h3>
                <div className="space-y-2">
                  {result.top_features.map((feature, idx) => (
                    <div key={idx} className="bg-white p-3 rounded-lg shadow-sm">
                      <div className="flex justify-between items-center">
                        <span className="font-medium text-gray-700">{feature.feature}</span>
                        <span className="text-sm text-gray-500">
                          {feature.value.toFixed(3)} ({(feature.importance * 100).toFixed(1)}% importance)
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

