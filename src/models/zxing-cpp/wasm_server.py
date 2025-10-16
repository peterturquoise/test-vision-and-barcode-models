#!/usr/bin/env python3
"""
Simple HTTP server to serve the ZXing-CPP WASM barcode scanner.
This eliminates the need for Docker containers and provides direct browser access.
"""

import http.server
import socketserver
import webbrowser
import os
import sys
from pathlib import Path

class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Add CORS headers for WASM loading
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        super().end_headers()

def main():
    # Get the directory containing this script
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Check if the WASM scanner HTML file exists
    wasm_scanner_path = script_dir / "wasm_scanner.html"
    if not wasm_scanner_path.exists():
        print("❌ Error: wasm_scanner.html not found!")
        print(f"Expected location: {wasm_scanner_path}")
        sys.exit(1)
    
    # Choose port
    PORT = 8080
    
    # Try to find an available port
    for port in range(8080, 8090):
        try:
            with socketserver.TCPServer(("", port), CustomHTTPRequestHandler) as httpd:
                print(f"🚀 ZXing-CPP WASM Scanner Server")
                print(f"📁 Serving from: {script_dir}")
                print(f"🌐 Server running at: http://localhost:{port}")
                print(f"🔍 Scanner URL: http://localhost:{port}/wasm_scanner.html")
                print(f"📱 Mobile-friendly scanner ready!")
                print()
                print("✨ Benefits of WASM vs Docker:")
                print("   • No Docker installation required")
                print("   • Runs directly in browser")
                print("   • Works on mobile devices")
                print("   • Near-native performance")
                print("   • Real-time camera scanning")
                print("   • Pattern matching (e.g., 3232 for BPost)")
                print()
                print("Press Ctrl+C to stop the server")
                
                # Open browser automatically
                try:
                    webbrowser.open(f"http://localhost:{port}/wasm_scanner.html")
                except:
                    pass
                
                httpd.serve_forever()
                break
        except OSError:
            continue
    else:
        print("❌ Error: Could not find an available port (8080-8089)")
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Server stopped. Thanks for using ZXing-CPP WASM Scanner!")
