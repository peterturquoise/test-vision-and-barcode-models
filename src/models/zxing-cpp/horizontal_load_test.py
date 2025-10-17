#!/usr/bin/env python3
"""
Simple load balancer for horizontal scaling test
Distributes requests across 4 single-worker containers
"""

import asyncio
import aiohttp
import time
import statistics
from pathlib import Path
import json
import random

class HorizontalLoadTester:
    def __init__(self, base_urls=None, test_image_path=None):
        # 4 single-worker containers on different ports
        self.base_urls = base_urls or [
            "http://localhost:8001",
            "http://localhost:8002", 
            "http://localhost:8003",
            "http://localhost:8004"
        ]
        self.test_image_path = test_image_path or "/Users/peterbrooke/dev/cursor/test-7-vision-models/test_images/Complete Labels/3_J18CBEP8CCN070812400095N.jpeg"
        self.results = []
        
    def get_random_url(self):
        """Get a random container URL for load balancing"""
        return random.choice(self.base_urls)
        
    async def single_request(self, session, request_id):
        """Make a single barcode detection request to a random container"""
        start_time = time.time()
        base_url = self.get_random_url()
        
        try:
            with open(self.test_image_path, 'rb') as f:
                data = aiohttp.FormData()
                data.add_field('file', f, filename='test_image.jpeg', content_type='image/jpeg')
                
                async with session.post(f"{base_url}/detect_barcodes", data=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        processing_time = result.get('processing_time', 0)
                        end_time = time.time()
                        total_time = end_time - start_time
                        
                        return {
                            'request_id': request_id,
                            'status': 'success',
                            'processing_time': processing_time,
                            'total_time': total_time,
                            'barcodes_found': len(result.get('barcodes', [])),
                            'status_code': response.status,
                            'container_url': base_url
                        }
                    else:
                        return {
                            'request_id': request_id,
                            'status': 'error',
                            'status_code': response.status,
                            'error': await response.text(),
                            'container_url': base_url
                        }
        except Exception as e:
            return {
                'request_id': request_id,
                'status': 'exception',
                'error': str(e),
                'container_url': base_url
            }
    
    async def run_load_test(self, duration_seconds=60, max_concurrent=20):
        """Run load test for specified duration across multiple containers"""
        print(f"🚀 Starting horizontal load test for {duration_seconds} seconds...")
        print(f"📊 Max concurrent requests: {max_concurrent}")
        print(f"🖼️  Test image: {self.test_image_path}")
        print(f"🌐 Containers: {len(self.base_urls)} single-worker containers")
        print(f"📍 URLs: {', '.join(self.base_urls)}")
        print(f"🔄 Load Balancing: Random Selection")
        print()
        
        start_time = time.time()
        request_counter = 0
        
        async with aiohttp.ClientSession() as session:
            # Create semaphore to limit concurrent requests
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def bounded_request():
                nonlocal request_counter
                async with semaphore:
                    request_counter += 1
                    return await self.single_request(session, request_counter)
            
            # Start load test
            tasks = []
            while time.time() - start_time < duration_seconds:
                # Start new requests up to max_concurrent
                while len(tasks) < max_concurrent and time.time() - start_time < duration_seconds:
                    task = asyncio.create_task(bounded_request())
                    tasks.append(task)
                
                # Wait for at least one task to complete
                if tasks:
                    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                    
                    # Process completed tasks
                    for task in done:
                        result = await task
                        self.results.append(result)
                        if result['status'] == 'success':
                            print(f"✅ Request {result['request_id']}: {result['processing_time']:.3f}s ({result['container_url'].split(':')[-1]})")
                        else:
                            print(f"❌ Request {result['request_id']}: {result['status']} ({result['container_url'].split(':')[-1]})")
                    
                    # Update tasks list
                    tasks = list(pending)
                
                # Small delay to prevent overwhelming
                await asyncio.sleep(0.1)
            
            # Wait for remaining tasks
            if tasks:
                remaining_results = await asyncio.gather(*tasks, return_exceptions=True)
                for result in remaining_results:
                    if isinstance(result, dict):
                        self.results.append(result)
        
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
        
        # Analyze per-container performance
        container_stats = {}
        for result in successful_requests:
            container = result['container_url']
            if container not in container_stats:
                container_stats[container] = []
            container_stats[container].append(result['processing_time'])
        
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
            },
            "container_performance": {
                container: {
                    "requests": len(times),
                    "avg_processing_time": statistics.mean(times),
                    "min_processing_time": min(times),
                    "max_processing_time": max(times)
                }
                for container, times in container_stats.items()
            }
        }
        
        return analysis
    
    def print_results(self, analysis):
        """Print formatted results"""
        print("\n" + "="*60)
        print("📊 HORIZONTAL SCALING LOAD TEST RESULTS")
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
        
        print(f"\n🌐 Container Performance:")
        for container, stats in analysis['container_performance'].items():
            port = container.split(':')[-1]
            print(f"   Port {port}: {stats['requests']} requests, avg {stats['avg_processing_time']:.3f}s")
        
        print(f"\n🎯 TARGET ANALYSIS:")
        target_rps = 11.6  # Required for 1M packages/day
        current_rps = throughput['requests_per_second']
        multiplier_needed = target_rps / current_rps if current_rps > 0 else float('inf')
        
        print(f"   Required: {target_rps:.1f} requests/second")
        print(f"   Current: {current_rps:.2f} requests/second")
        print(f"   Gap: {multiplier_needed:.1f}x capacity needed")
        
        if multiplier_needed <= 4:
            print(f"   ✅ Can achieve with {int(multiplier_needed)} containers")
        else:
            print(f"   ⚠️  Need {int(multiplier_needed)}+ containers or optimization")

async def main():
    """Main function to run horizontal load test"""
    tester = HorizontalLoadTester()
    
    try:
        analysis = await tester.run_load_test(duration_seconds=60, max_concurrent=12)
        tester.print_results(analysis)
        
        # Save results to file
        with open('horizontal_load_test_results.json', 'w') as f:
            json.dump({
                'analysis': analysis,
                'raw_results': tester.results
            }, f, indent=2)
        print(f"\n💾 Results saved to horizontal_load_test_results.json")
        
    except Exception as e:
        print(f"❌ Horizontal load test failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
