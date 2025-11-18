#!/usr/bin/env python3
"""
Vercel API Route for Processing Companies
=========================================
Serverless function endpoint that processes a company from Crunchbase URL.
"""

from http.server import BaseHTTPRequestHandler
import os
import sys
import json
import asyncio

# Add parent directories to path
# This allows importing from the main BRAIN project directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up from api/process.py to vercel-frontend, then to BRAIN root
vercel_frontend_dir = os.path.join(current_dir, '..')
project_root = os.path.join(vercel_frontend_dir, '..')
sys.path.insert(0, project_root)
sys.path.insert(0, vercel_frontend_dir)

from lib.scoring_service import process_company_from_url


class handler(BaseHTTPRequestHandler):
    """Vercel serverless function handler."""
    
    def do_OPTIONS(self):
        """Handle CORS preflight requests."""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def do_POST(self):
        """Handle POST requests to process a company."""
        try:
            # Read request body
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode('utf-8'))
            
            crunchbase_url = data.get('url', '').strip()
            
            if not crunchbase_url:
                self._send_error(400, 'Crunchbase URL is required')
                return
            
            # Validate URL format
            if 'crunchbase.com/organization/' not in crunchbase_url:
                self._send_error(400, 'Invalid Crunchbase URL format. Expected: https://www.crunchbase.com/organization/company-name')
                return
            
            # Process company (async)
            result = asyncio.run(process_company_from_url(crunchbase_url))
            
            # Send response
            status_code = 200 if result.get('success') else 400
            self.send_response(status_code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())
            
        except json.JSONDecodeError:
            self._send_error(400, 'Invalid JSON in request body')
        except Exception as e:
            self._send_error(500, f'Server error: {str(e)}')
    
    def _send_error(self, status_code: int, error_message: str):
        """Helper to send error responses."""
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps({
            'success': False,
            'error': error_message
        }).encode())

