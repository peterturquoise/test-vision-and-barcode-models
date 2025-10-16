#!/usr/bin/env python3
"""
Simple web server for ZXing Barcode Scanner
Serves the HTML interface and proxies API requests
"""

from http.server import HTTPServer, SimpleHTTPRequestHandler
import urllib.request
import urllib.parse
import json
import os
from pathlib import Path

class CORSRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        if self.path == '/':
            self.path = '/web_scanner.html'
        return super().do_GET()

    def do_POST(self):
        if self.path.startswith('/api/'):
            # Proxy API requests to the ZXing container
            self.proxy_api_request()
        else:
            self.send_error(404)

    def proxy_api_request(self):
        try:
            # Extract the API path
            api_path = self.path[4:]  # Remove '/api' prefix
            
            # Read request body
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            # Forward request to ZXing API
            url = f'http://localhost:8001{api_path}'
            req = urllib.request.Request(url, data=post_data, method='POST')
            
            # Copy headers
            for header, value in self.headers.items():
                if header.lower() not in ['host', 'content-length']:
                    req.add_header(header, value)
            
            # Make request
            with urllib.request.urlopen(req) as response:
                response_data = response.read()
                
                # Send response
                self.send_response(response.status)
                self.send_header('Content-Type', response.headers.get('Content-Type', 'application/json'))
                self.end_headers()
                self.wfile.write(response_data)
                
        except Exception as e:
            self.send_error(500, f"Proxy error: {str(e)}")

def main():
    port = 8080
    server_address = ('', port)
    
    print(f"🚀 Starting ZXing Barcode Scanner Web Server")
    print(f"📱 Open your browser to: http://localhost:{port}")
    print(f"🔗 Make sure ZXing API is running on http://localhost:8001")
    print(f"⏹️  Press Ctrl+C to stop")
    
    httpd = HTTPServer(server_address, CORSRequestHandler)
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print(f"\n🛑 Server stopped")
        httpd.shutdown()

if __name__ == "__main__":
    main()
