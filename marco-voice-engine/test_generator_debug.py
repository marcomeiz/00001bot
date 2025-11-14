#!/usr/bin/env python3
"""
Debug script to test the generator directly with custom prompts.
"""

import sys
import os
sys.path.append('/Users/marcomeipersonal/Desktop/MMEI/Proyectos Personales/00xandbot/repos/00001bot/marco-voice-engine/src')

from marco_voice_engine.generator import Generator

def test_generator():
    print("🧪 Testing Generator with Custom Prompt")
    print("=" * 50)
    
    # Test with custom prompt
    custom_prompt = "Generate tweets about the topic. Include the word PERRO somewhere in the text."
    topic = "business automation"
    
    print(f"Topic: {topic}")
    print(f"Custom prompt: {custom_prompt}")
    print()
    
    generator = Generator(mode="ops")
    
    try:
        # Test with custom prompt
        print("Testing with custom prompt...")
        variants = generator.generate_variants(topic, n_expected=2, custom_prompt=custom_prompt)
        
        print(f"Generated {len(variants)} variants:")
        for i, variant in enumerate(variants):
            print(f"Variant {i+1}: {variant}")
            if 'PERRO' in variant:
                print(f"✅ PERRO found in variant {i+1}!")
            else:
                print(f"❌ PERRO not found in variant {i+1}")
        
        print()
        
        # Test without custom prompt (default)
        print("Testing with default prompt...")
        variants_default = generator.generate_variants(topic, n_expected=1)
        
        print(f"Generated {len(variants_default)} variants with default:")
        for i, variant in enumerate(variants_default):
            print(f"Default variant {i+1}: {variant}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_generator()