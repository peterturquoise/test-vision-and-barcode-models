#!/usr/bin/env python3
import asyncio
import aiohttp
import time
import os

async def send_request(session, url, image_path, request_id):
    """Send a single request"""
    try:
        with open(image_path, 'rb') as f:
            data = aiohttp.FormData()
            data.add_field('file', f, filename='test.jpg', content_type='image/jpeg')
            
            start_time = time.time()
            async with session.post(url, data=data) as response:
                result = await response.json()
                processing_time = time.time() - start_time
                print(f"Request {request_id}: {processing_time:.3f}s - {result.get('processing_time', 'N/A')}s processing")
                return result
    except Exception as e:
        print(f"Request {request_id} failed: {e}")
        return None

async def test_concurrent_requests():
    """Test multiple concurrent requests"""
    image_path = "/Users/peterbrooke/dev/cursor/test-7-vision-models/test_images/Complete Labels/3_J18CBEP8CCN070812400095N.jpeg"
    url = "http://localhost:8001/detect_barcodes"
    
    print(f"🚀 Testing concurrent requests to {url}")
    print(f"📁 Using image: {os.path.basename(image_path)}")
    print()
    
    async with aiohttp.ClientSession() as session:
        # Test with 3 concurrent requests
        tasks = []
        for i in range(3):
            task = send_request(session, url, image_path, i+1)
            tasks.append(task)
        
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        print(f"\n📊 Results:")
        print(f"Total time for 3 concurrent requests: {total_time:.3f}s")
        successful = sum(1 for r in results if r is not None)
        print(f"Successful requests: {successful}/3")

if __name__ == "__main__":
    asyncio.run(test_concurrent_requests())
