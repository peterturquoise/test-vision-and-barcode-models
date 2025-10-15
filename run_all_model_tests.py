#!/usr/bin/env python3
"""
Run All Model Tests
Executes tests for all available vision models and provides comprehensive analysis
"""

import subprocess
import sys
from pathlib import Path
import time

def run_model_test(model_name, model_dir):
    """Run test for a specific model"""
    print(f"\n{'='*60}")
    print(f"🧪 Testing {model_name.upper()}")
    print(f"{'='*60}")
    
    test_file = model_dir / "test.py"
    if not test_file.exists():
        print(f"❌ No test.py found for {model_name}")
        return False
    
    try:
        print(f"📁 Running test from: {model_dir}")
        result = subprocess.run([sys.executable, "test.py"], 
                              cwd=model_dir, 
                              capture_output=True, 
                              text=True, 
                              timeout=300)  # 5 minute timeout
        
        if result.returncode == 0:
            print(f"✅ {model_name} test completed successfully")
            print("📄 Output:")
            print(result.stdout)
            return True
        else:
            print(f"❌ {model_name} test failed")
            print("📄 Error output:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {model_name} test timed out (5 minutes)")
        return False
    except Exception as e:
        print(f"❌ {model_name} test error: {e}")
        return False

def run_comprehensive_analysis():
    """Run comprehensive analysis of all results"""
    print(f"\n{'='*60}")
    print("📊 COMPREHENSIVE ANALYSIS")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run([sys.executable, "analyze_results.py"], 
                              capture_output=True, 
                              text=True, 
                              timeout=60)
        
        if result.returncode == 0:
            print("✅ Analysis completed successfully")
            print("📄 Analysis Results:")
            print(result.stdout)
            return True
        else:
            print("❌ Analysis failed")
            print("📄 Error output:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ Analysis timed out")
        return False
    except Exception as e:
        print(f"❌ Analysis error: {e}")
        return False

def main():
    """Main function to run all model tests"""
    print("🚀 VISION MODELS COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    print("This script will:")
    print("1. Test all available vision models")
    print("2. Run comprehensive analysis")
    print("3. Provide performance comparison")
    print("=" * 80)
    
    # Find all model directories
    models_dir = Path("src/models")
    if not models_dir.exists():
        print("❌ No src/models directory found")
        return False
    
    model_dirs = [d for d in models_dir.iterdir() if d.is_dir()]
    
    if not model_dirs:
        print("❌ No model directories found")
        return False
    
    print(f"\n📋 Found {len(model_dirs)} model directories:")
    for model_dir in model_dirs:
        print(f"   - {model_dir.name}")
    
    # Run tests for each model
    successful_tests = []
    failed_tests = []
    
    start_time = time.time()
    
    for model_dir in model_dirs:
        model_name = model_dir.name
        success = run_model_test(model_name, model_dir)
        
        if success:
            successful_tests.append(model_name)
        else:
            failed_tests.append(model_name)
    
    total_time = time.time() - start_time
    
    # Summary
    print(f"\n{'='*80}")
    print("📊 TEST SUMMARY")
    print(f"{'='*80}")
    print(f"✅ Successful tests: {len(successful_tests)}")
    for model in successful_tests:
        print(f"   - {model}")
    
    print(f"\n❌ Failed tests: {len(failed_tests)}")
    for model in failed_tests:
        print(f"   - {model}")
    
    print(f"\n⏱️ Total execution time: {total_time:.1f} seconds")
    
    # Run comprehensive analysis if we have successful tests
    if successful_tests:
        print(f"\n🔍 Running comprehensive analysis...")
        analysis_success = run_comprehensive_analysis()
        
        if analysis_success:
            print("\n🎉 All tests and analysis completed!")
        else:
            print("\n⚠️ Tests completed but analysis failed")
    else:
        print("\n❌ No successful tests to analyze")
    
    return len(successful_tests) > 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)






