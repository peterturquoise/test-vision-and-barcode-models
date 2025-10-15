"""
Comprehensive Analysis of Vision Models Test Results
Analyzes performance across all models using JSON-based expected results
"""

import json
import os
import sys
from pathlib import Path
from collections import defaultdict

# Add src/lib to path to import expected_results
sys.path.append(str(Path(__file__).parent / "src" / "lib"))

# Import the core utility functions
from expected_results import load_test_results, get_expected_for_image

def analyze_results():
    """Analyze all model results and provide comprehensive comparison"""
    
    # Look for results in both old and new locations
    results_locations = [
        Path("results"),  # Old location
        Path("src/models")  # New per-model location
    ]
    
    print("="*80)
    print("COMPREHENSIVE VISION MODELS ANALYSIS")
    print("="*80)
    
    # Load expected results from JSON files using utility function
    expected_results = load_all_expected_results()
    
    # Load model results
    model_results = {}
    
    for results_dir in results_locations:
        if results_dir.exists():
            # Look for result files
            result_files = list(results_dir.glob("**/*_results.json"))
            for result_file in result_files:
                model_name = result_file.stem.replace("_results", "")
                try:
                    with open(result_file, 'r') as f:
                        model_results[model_name] = json.load(f)
                    print(f"✓ Loaded {model_name} results from {result_file}")
                except Exception as e:
                    print(f"✗ Failed to load {model_name}: {e}")
    
    print(f"\nLoaded {len(model_results)} model result files")
    
    if not model_results:
        print("No model results found. Run some model tests first.")
        return
    
    # Analyze each model
    analysis = {}
    
    for model_name, results in model_results.items():
        print(f"\n{'='*60}")
        print(f"ANALYZING {model_name.upper()}")
        print(f"{'='*60}")
        
        total_images = len(results)
        successful_images = sum(1 for r in results if r.get('error') is None)
        failed_images = total_images - successful_images
        
        # Count detections
        total_text_detections = sum(len(r.get('text_detections', [])) for r in results)
        total_barcode_detections = sum(len(r.get('barcode_detections', [])) for r in results)
        
        # Calculate average processing time
        processing_times = [r.get('processing_time', 0) for r in results if r.get('processing_time')]
        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
        
        # Analyze against expected results
        accuracy_analysis = analyze_accuracy(results, expected_results)
        
        # Analyze text quality
        text_quality = analyze_text_quality(results)
        
        # Analyze barcode quality
        barcode_quality = analyze_barcode_quality(results)
        
        # Check for errors
        errors = analyze_errors(results)
        
        analysis[model_name] = {
            'total_images': total_images,
            'successful_images': successful_images,
            'failed_images': failed_images,
            'success_rate': (successful_images / total_images * 100) if total_images > 0 else 0,
            'total_text_detections': total_text_detections,
            'total_barcode_detections': total_barcode_detections,
            'avg_processing_time': avg_processing_time,
            'accuracy_analysis': accuracy_analysis,
            'text_quality': text_quality,
            'barcode_quality': barcode_quality,
            'errors': errors
        }
        
        # Print model summary
        print(f"Images Processed: {successful_images}/{total_images} ({analysis[model_name]['success_rate']:.1f}%)")
        print(f"Text Detections: {total_text_detections}")
        print(f"Barcode Detections: {total_barcode_detections}")
        print(f"Avg Processing Time: {avg_processing_time:.2f}s")
        print(f"Text Quality Score: {text_quality['score']:.2f}/10")
        print(f"Barcode Quality Score: {barcode_quality['score']:.2f}/10")
        print(f"Overall Accuracy: {accuracy_analysis['overall_score']:.1f}%")
        
        if errors:
            print(f"Errors: {len(errors)} unique error types")
            for error_type, count in errors.items():
                print(f"  - {error_type}: {count} occurrences")
    
    # Overall comparison
    print(f"\n{'='*80}")
    print("OVERALL COMPARISON")
    print(f"{'='*80}")
    
    # Find best performers
    best_text_model = max(analysis.items(), key=lambda x: x[1]['text_quality']['score'])
    best_barcode_model = max(analysis.items(), key=lambda x: x[1]['barcode_quality']['score'])
    fastest_model = min(analysis.items(), key=lambda x: x[1]['avg_processing_time'])
    most_reliable_model = max(analysis.items(), key=lambda x: x[1]['success_rate'])
    most_accurate_model = max(analysis.items(), key=lambda x: x[1]['accuracy_analysis']['overall_score'])
    
    print(f"🏆 BEST TEXT DETECTION: {best_text_model[0]} (Score: {best_text_model[1]['text_quality']['score']:.2f}/10)")
    print(f"🏆 BEST BARCODE DETECTION: {best_barcode_model[0]} (Score: {best_barcode_model[1]['barcode_quality']['score']:.2f}/10)")
    print(f"🏆 FASTEST MODEL: {fastest_model[0]} ({fastest_model[1]['avg_processing_time']:.2f}s avg)")
    print(f"🏆 MOST RELIABLE: {most_reliable_model[0]} ({most_reliable_model[1]['success_rate']:.1f}% success rate)")
    print(f"🏆 MOST ACCURATE: {most_accurate_model[0]} ({most_accurate_model[1]['accuracy_analysis']['overall_score']:.1f}% accuracy)")
    
    # Detailed comparison table
    print(f"\n{'='*140}")
    print("DETAILED COMPARISON TABLE")
    print(f"{'='*140}")
    print(f"{'Model':<20} {'Success%':<10} {'Text Det':<10} {'Barcode Det':<12} {'Avg Time':<10} {'Text Score':<12} {'Barcode Score':<15} {'Accuracy%':<12}")
    print(f"{'-'*140}")
    
    for model_name, data in analysis.items():
        print(f"{model_name:<20} {data['success_rate']:<10.1f} {data['total_text_detections']:<10} {data['total_barcode_detections']:<12} {data['avg_processing_time']:<10.2f} {data['text_quality']['score']:<12.2f} {data['barcode_quality']['score']:<15.2f} {data['accuracy_analysis']['overall_score']:<12.1f}")
    
    # Issue analysis
    print(f"\n{'='*80}")
    print("ISSUE ANALYSIS & RECOMMENDATIONS")
    print(f"{'='*80}")
    
    for model_name, data in analysis.items():
        if data['errors']:
            print(f"\n🔧 {model_name.upper()} ISSUES:")
            for error_type, count in data['errors'].items():
                print(f"  - {error_type} ({count} occurrences)")
                print(f"    Recommendation: {get_error_recommendation(error_type)}")

def load_all_expected_results():
    """Load all expected results from .testresults.json files"""
    expected_results = {}
    
    test_images_dir = Path("test_images")
    if not test_images_dir.exists():
        print("⚠️ No test_images directory found")
        return expected_results
    
    # Scan for .testresults.json files
    json_files = list(test_images_dir.glob("**/*.testresults.json"))
    
    for json_file in json_files:
        try:
            # Use the utility function to load individual results
            results = load_test_results(str(json_file))
            if results:
                # Extract image name
                image_name = json_file.with_suffix('').with_suffix('').name
                expected_results[image_name] = results
                
        except Exception as e:
            print(f"⚠️ Error loading {json_file}: {e}")
    
    print(f"✓ Loaded expected results for {len(expected_results)} images")
    return expected_results

def analyze_accuracy(results, expected_results):
    """Analyze accuracy against expected results"""
    total_expected_text = 0
    total_expected_barcodes = 0
    total_found_text = 0
    total_found_barcodes = 0
    
    for result in results:
        # Find matching expected results
        image_name = result.get('image_name', '')
        if image_name in expected_results:
            expected = expected_results[image_name]
            
            # Count expected elements
            total_expected_text += len(expected.get('expected_text', []))
            total_expected_barcodes += len(expected.get('expected_barcodes', []))
            
            # Count found elements
            found_texts = [det.get('text', '') for det in result.get('text_detections', [])]
            found_barcodes = [det.get('value', '') for det in result.get('barcode_detections', [])]
            
            # Check matches
            for expected_text in expected.get('expected_text', []):
                if any(expected_text.lower() in text.lower() for text in found_texts):
                    total_found_text += 1
            
            for expected_barcode in expected.get('expected_barcodes', []):
                barcode_value = expected_barcode.get('value', expected_barcode) if isinstance(expected_barcode, dict) else expected_barcode
                if barcode_value in found_barcodes:
                    total_found_barcodes += 1
    
    # Calculate accuracy
    text_accuracy = (total_found_text / total_expected_text * 100) if total_expected_text > 0 else 0
    barcode_accuracy = (total_found_barcodes / total_expected_barcodes * 100) if total_expected_barcodes > 0 else 0
    overall_accuracy = (total_found_text + total_found_barcodes) / (total_expected_text + total_expected_barcodes) * 100 if (total_expected_text + total_expected_barcodes) > 0 else 0
    
    return {
        'text_accuracy': text_accuracy,
        'barcode_accuracy': barcode_accuracy,
        'overall_score': overall_accuracy,
        'expected_text': total_expected_text,
        'expected_barcodes': total_expected_barcodes,
        'found_text': total_found_text,
        'found_barcodes': total_found_barcodes
    }

def analyze_text_quality(results):
    """Analyze text detection quality"""
    quality_score = 0
    total_detections = 0
    high_confidence_detections = 0
    meaningful_text_count = 0
    
    for result in results:
        text_detections = result.get('text_detections', [])
        total_detections += len(text_detections)
        
        for detection in text_detections:
            confidence = detection.get('confidence', 0)
            text = detection.get('text', '').strip()
            
            if confidence > 0.7:
                high_confidence_detections += 1
            
            # Check if text is meaningful (not single characters, numbers, or gibberish)
            if len(text) > 2 and not text.isdigit() and any(c.isalpha() for c in text):
                meaningful_text_count += 1
    
    # Calculate quality score (0-10)
    if total_detections > 0:
        confidence_score = (high_confidence_detections / total_detections) * 5
        meaningful_score = (meaningful_text_count / total_detections) * 5
        quality_score = confidence_score + meaningful_score
    
    return {
        'score': quality_score,
        'total_detections': total_detections,
        'high_confidence': high_confidence_detections,
        'meaningful_text': meaningful_text_count
    }

def analyze_barcode_quality(results):
    """Analyze barcode detection quality"""
    quality_score = 0
    total_detections = 0
    successful_decodes = 0
    
    for result in results:
        barcode_detections = result.get('barcode_detections', [])
        total_detections += len(barcode_detections)
        
        for detection in barcode_detections:
            value = detection.get('value', '')
            if value and value.strip():
                successful_decodes += 1
    
    # Calculate quality score (0-10)
    if total_detections > 0:
        decode_rate = successful_decodes / total_detections
        quality_score = decode_rate * 10
    
    return {
        'score': quality_score,
        'total_detections': total_detections,
        'successful_decodes': successful_decodes
    }

def analyze_errors(results):
    """Analyze error patterns"""
    error_counts = defaultdict(int)
    
    for result in results:
        error = result.get('error')
        if error:
            # Extract main error type
            if 'not available' in error.lower():
                error_counts['Package not available'] += 1
            elif 'not a valid model identifier' in error.lower():
                error_counts['Invalid model identifier'] += 1
            elif 'incorrect image source' in error.lower():
                error_counts['Image processing error'] += 1
            elif 'no attribute' in error.lower():
                error_counts['API compatibility error'] += 1
            elif 'accelerate' in error.lower():
                error_counts['Missing accelerate package'] += 1
            else:
                error_counts['Other error'] += 1
    
    return dict(error_counts)

def get_error_recommendation(error_type):
    """Get recommendations for fixing errors"""
    recommendations = {
        'Package not available': 'Install missing packages: pip install segment-anything minigpt4',
        'Invalid model identifier': 'Update model names to correct Hugging Face identifiers',
        'Image processing error': 'Fix image preprocessing pipeline for vision-language models',
        'API compatibility error': 'Update model API calls to match current library versions',
        'Missing accelerate package': 'Install accelerate: pip install accelerate',
        'Other error': 'Check model-specific documentation for setup requirements'
    }
    return recommendations.get(error_type, 'Check model documentation')

if __name__ == "__main__":
    analyze_results()


