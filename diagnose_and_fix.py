#!/usr/bin/env python3
# diagnose_and_fix.py - Diagnostic script to find the best solution

import subprocess
import sys
import os
from pathlib import Path

def print_header(text):
    """Print a formatted header"""
    print("\n" + "="*60)
    print(f" {text} ")
    print("="*60)

def test_import(module_name):
    """Test if a module can be imported"""
    try:
        __import__(module_name)
        return True
    except Exception as e:
        print(f"✗ {module_name}: {str(e)[:50]}...")
        return False

def check_current_environment():
    """Check the current Python environment"""
    print_header("Current Environment Check")
    
    print(f"Python Version: {sys.version}")
    print(f"Python Executable: {sys.executable}")
    
    # Check key packages
    packages = {
        'torch': 'PyTorch',
        'transformers': 'Transformers',
        'sentence_transformers': 'Sentence Transformers',
        'streamlit': 'Streamlit',
        'faiss': 'FAISS',
        'numpy': 'NumPy',
        'pandas': 'Pandas'
    }
    
    working = []
    broken = []
    
    for pkg, name in packages.items():
        if test_import(pkg):
            working.append(name)
        else:
            broken.append(name)
    
    print(f"\n✓ Working: {', '.join(working)}")
    print(f"✗ Broken: {', '.join(broken)}")
    
    return len(broken) == 0

def test_sentence_transformers():
    """Test if sentence-transformers actually works"""
    print_header("Testing Sentence Transformers")
    
    test_code = """
import sys
try:
    from sentence_transformers import SentenceTransformer
    print("✓ Import successful")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✓ Model loaded")
    embedding = model.encode("Test")
    print(f"✓ Encoding successful! Shape: {embedding.shape}")
    sys.exit(0)
except Exception as e:
    print(f"✗ Failed: {e}")
    sys.exit(1)
"""
    
    result = subprocess.run([sys.executable, '-c', test_code], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    
    return result.returncode == 0

def test_alternative_implementation():
    """Test the alternative implementation"""
    print_header("Testing Alternative Implementation")
    
    test_code = """
from transformers import AutoTokenizer, AutoModel
import torch

try:
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    print("✓ Model loaded without sentence-transformers")
    
    # Test encoding
    inputs = tokenizer("Test", return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    print("✓ Encoding successful!")
    print("Alternative implementation works!")
except Exception as e:
    print(f"✗ Alternative failed: {e}")
"""
    
    result = subprocess.run([sys.executable, '-c', test_code], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    
    return result.returncode == 0

def suggest_solution():
    """Suggest the best solution based on tests"""
    print_header("Recommended Solution")
    
    # Check if we're on Mac
    is_mac = sys.platform == 'darwin'
    python_version = sys.version_info
    
    if is_mac:
        print("You're on macOS - this is likely causing the segmentation fault.")
    
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version >= (3, 11):
        print("\n⚠️  You're using Python 3.11+, which has known issues with sentence-transformers.")
        print("Recommendation: Use Python 3.10")
    
    print("\n" + "-"*60)
    print("IMMEDIATE SOLUTIONS (in order of preference):")
    print("-"*60)
    
    print("\n1. Quick Fix with Alternative Implementation:")
    print("   - Copy 'alternative_recommender.py' to your project")
    print("   - In your app, replace:")
    print("     from manthan_core.recommender.intelligent_recommender import IntelligentRecommender")
    print("   - With:")
    print("     from alternative_recommender import AlternativeRecommender as IntelligentRecommender")
    
    print("\n2. Fix Dependencies (if you have time):")
    print("   - Run: chmod +x quick_fix.sh && ./quick_fix.sh")
    print("   - This will create a new environment with compatible versions")
    
    print("\n3. Use Lightweight Cloud Alternative:")
    print("   - Use Google Colab or GitHub Codespaces")
    print("   - These provide Linux environments where sentence-transformers works reliably")

def main():
    """Main diagnostic function"""
    print_header("Manthan Diagnostic Tool")
    
    # Check environment
    env_ok = check_current_environment()
    
    if env_ok:
        # Test sentence-transformers
        st_ok = test_sentence_transformers()
        
        if st_ok:
            print("\n✅ Everything seems to be working!")
            print("You can run your app with: streamlit run intelligent_app.py")
        else:
            print("\n❌ Sentence-transformers is failing (segmentation fault)")
            # Test alternative
            alt_ok = test_alternative_implementation()
            if alt_ok:
                print("\n✅ Good news! The alternative implementation works.")
                suggest_solution()
    else:
        print("\n❌ Some dependencies are missing or broken")
        suggest_solution()
    
    print("\n" + "="*60)
    print("For detailed instructions, see: manthan-fixes.md")
    print("="*60)

if __name__ == "__main__":
    main()