#!/usr/bin/env python3
"""
Simple load test script for zxing-cpp barcode detection API
Uses only standard library (no aiohttp dependency)
"""

import requests
import time
import statistics
import threading
from concurrent.futures import ThreadPoolExecutor
import json

class SimpleLoadTester:
    def __init__(self, base_url="http://localhost:8001", test_image_path=None):
        self.base_url = base_url
        self.test_image_path = test_image_path or "/Users/peterbrooke/dev/cursor/test-7-vision-models/test_images/Complete Labels/3_J18CBEP8CCN070812400095N.jpeg"
        self.results = []
        
    def single_request(self, request_id):
        """Make a single barcode detection request"""
        start_time = time.time()
        
        try:
            with open(self.test_image_path, 'rb') as f:
                files = {'file': ('test_image.jpeg', f, 'image/jpeg')}
                response = requests.post(f"{self.base_url}/detect_barcodes", files=files, timeout=30)
                
                if response.status_code == 200:
                    result = response.json()
                    processing_time = result.get('processing_time', 0)
                    end_time = time.time()
                    total_time = end_time - start_time
                    
                    return {
                        'request_id': request_id,
                        'status': 'success',
                        'processing_time': processing_time,
                        'total_time': total_time,
                        'barcodes_found': len(result.get('barcodes', [])),
                        'status_code': response.status_code
                    }
                else:
                    return {
                        'request_id': request_id,
                        'status': 'error',
                        'status_code': response.status_code,
                        'error': response.text
                    }
        except Exception as e:
            return {
                'request_id': request_id,
                'status': 'exception',
                'error': str(e)
            }
    
    def run_load_test(self, duration_seconds=60, max_concurrent=5):
        """Run load test for specified duration"""
        print(f"🚀 Starting load test for {duration_seconds} seconds...")
        print(f"📊 Max concurrent requests: {max_concurrent}")
        print(f"🖼️  Test image: {self.test_image_path}")
        print()
        
        start_time = time.time()
        request_counter = 0
        
        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            futures = []
            
            while time.time() - start_time < duration_seconds:
                # Start new requests up to max_concurrent
                while len(futures) < max_concurrent and time.time() - start_time < duration_seconds:
                    request_counter += 1
                    future = executor.submit(self.single_request, request_counter)
                    futures.append(future)
                
                # Check for completed requests
                completed_futures = []
                for future in futures:
                    if future.done():
                        result = future.result()
                        self.results.append(result)
                        if result['status'] == 'success':
                            print(f"✅ Request {result['request_id']}: {result['processing_time']:.3f}s")
                        else:
                            print(f"❌ Request {result['request_id']}: {result['status']}")
                        completed_futures.append(future)
                
                # Remove completed futures
                for future in completed_futures:
                    futures.remove(future)
                
                # Small delay to prevent overwhelming
                time.sleep(0.1)
            
            # Wait for remaining requests
            for future in futures:
                result = future.result()
                self.results.append(result)
                if result['status'] == 'success':
                    print(f"✅ Request {result['request_id']}: {result['processing_time']:.3f}s")
                else:
                    print(f"❌ Request {result['request_id']}: {result['status']}")
        
        return self.analyze_results()
    
    def analyze_results(self):
        """Analyze load test results"""
        if not self.results:
            return {"error": "No results to analyze"}
        
        successful_requests = [r for r in self.results if r['status'] == 'success']
        failed_requests = [r for r in self.results if r['status'] != 'success']
        
        if not successful_requests:
            return {
                "total_requests": len(self.results),
                "successful_requests": 0,
                "failed_requests": len(failed_requests),
                "success_rate": 0.0,
                "error": "No successful requests"
            }
        
        processing_times = [r['processing_time'] for r in successful_requests]
        total_times = [r['total_time'] for r in successful_requests]
        
        analysis = {
            "total_requests": len(self.results),
            "successful_requests": len(successful_requests),
            "failed_requests": len(failed_requests),
            "success_rate": len(successful_requests) / len(self.results) * 100,
            "processing_time_stats": {
                "min": min(processing_times),
                "max": max(processing_times),
                "mean": statistics.mean(processing_times),
                "median": statistics.median(processing_times),
                "std_dev": statistics.stdev(processing_times) if len(processing_times) > 1 else 0
            },
            "total_time_stats": {
                "min": min(total_times),
                "max": max(total_times),
                "mean": statistics.mean(total_times),
                "median": statistics.median(total_times)
            },
            "throughput": {
                "requests_per_second": len(successful_requests) / 60,  # Assuming 60 second test
                "packages_per_day": len(successful_requests) / 60 * 3600 * 24
            }
        }
        
        return analysis
    
    def print_results(self, analysis):
        """Print formatted results"""
        print("\n" + "="*60)
        print("📊 LOAD TEST RESULTS")
        print("="*60)
        
        print(f"📈 Total Requests: {analysis['total_requests']}")
        print(f"✅ Successful: {analysis['successful_requests']}")
        print(f"❌ Failed: {analysis['failed_requests']}")
        print(f"📊 Success Rate: {analysis['success_rate']:.1f}%")
        
        print(f"\n⏱️  Processing Time Stats:")
        stats = analysis['processing_time_stats']
        print(f"   Min: {stats['min']:.3f}s")
        print(f"   Max: {stats['max']:.3f}s")
        print(f"   Mean: {stats['mean']:.3f}s")
        print(f"   Median: {stats['median']:.3f}s")
        print(f"   Std Dev: {stats['std_dev']:.3f}s")
        
        print(f"\n🚀 Throughput:")
        throughput = analysis['throughput']
        print(f"   Requests/second: {throughput['requests_per_second']:.2f}")
        print(f"   Packages/day: {throughput['packages_per_day']:,.0f}")
        
        print(f"\n🎯 TARGET ANALYSIS:")
        target_rps = 11.6  # Required for 1M packages/day
        current_rps = throughput['requests_per_second']
        multiplier_needed = target_rps / current_rps if current_rps > 0 else float('inf')
        
        print(f"   Required: {target_rps:.1f} requests/second")
        print(f"   Current: {current_rps:.2f} requests/second")
        print(f"   Gap: {multiplier_needed:.1f}x capacity needed")
        
        if multiplier_needed <= 8:
            print(f"   ✅ Can achieve with {int(multiplier_needed)} workers")
        else:
            print(f"   ⚠️  Need {int(multiplier_needed)}+ workers or optimization")

def main():
    """Main function to run load test"""
    tester = SimpleLoadTester()
    
    try:
        analysis = await tester.run_load_test(duration_seconds=60, max_concurrent=5)
        tester.print_results(analysis)
        
        # Save results to file
        with open('load_test_results.json', 'w') as f:
            json.dump({
                'analysis': analysis,
                'raw_results': tester.results
            }, f, indent=2)
        print(f"\n💾 Results saved to load_test_results.json")
        
    except Exception as e:
        print(f"❌ Load test failed: {e}")

if __name__ == "__main__":
    main()
