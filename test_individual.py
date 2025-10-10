"""
Individual Model Test Scripts
Run tests for specific models
"""

import sys
from pathlib import Path

# Add models directory to path
sys.path.append(str(Path(__file__).parent / "models"))

def test_yolov9():
    """Test YOLOv9 model"""
    from yolov9_model import test_yolov9
    test_yolov9()

def test_mobilesam():
    """Test MobileSAM model"""
    from mobilesam_model import test_mobilesam
    test_mobilesam()

def test_llava():
    """Test LLaVA-1.5 model"""
    from llava15_model import test_llava15
    test_llava15()

def test_minigpt4():
    """Test MiniGPT-4 model"""
    from minigpt4_model import test_minigpt4
    test_minigpt4()

def test_qwen_vl():
    """Test Qwen-VL model"""
    from qwen_vl_model import test_qwen_vl
    test_qwen_vl()

def test_cogvlm():
    """Test CogVLM model"""
    from cogvlm_model import test_cogvlm
    test_cogvlm()

def test_mobile_llava():
    """Test Mobile-tuned LLaVA model"""
    from mobile_llava_model import test_mobile_llava
    test_mobile_llava()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        model_name = sys.argv[1].lower()
        
        if model_name == "yolov9":
            test_yolov9()
        elif model_name == "mobilesam":
            test_mobilesam()
        elif model_name == "llava":
            test_llava()
        elif model_name == "minigpt4":
            test_minigpt4()
        elif model_name == "qwen-vl":
            test_qwen_vl()
        elif model_name == "cogvlm":
            test_cogvlm()
        elif model_name == "mobile-llava":
            test_mobile_llava()
        else:
            print(f"Unknown model: {model_name}")
            print("Available models: yolov9, mobilesam, llava, minigpt4, qwen-vl, cogvlm, mobile-llava")
    else:
        print("Usage: python test_individual.py <model_name>")
        print("Available models: yolov9, mobilesam, llava, minigpt4, qwen-vl, cogvlm, mobile-llava")
