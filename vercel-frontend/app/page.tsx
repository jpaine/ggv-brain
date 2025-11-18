'use client';

import { useState } from 'react';
import CompanyForm from '@/components/CompanyForm';

export default function Home() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-4xl mx-auto">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">
            GGV Brain Company Scorer
          </h1>
          <p className="text-lg text-gray-600">
            Enter a Crunchbase URL to analyze a company and receive a detailed score
          </p>
        </div>
        
        <CompanyForm />
      </div>
    </div>
  );
}

