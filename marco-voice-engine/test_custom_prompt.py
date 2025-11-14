#!/usr/bin/env python3
"""
Test script to verify that Marco Voice Engine accepts custom prompts
"""

import requests
import json

def test_custom_prompt():
    """Test that custom prompts are accepted and used"""
    
    # Test with custom prompt
    custom_prompt = """
    You are a social media expert. Generate tweets that MUST include the word "PERRO" in the text.
    The tweets should be engaging and relevant to the topic provided.
    """
    
    test_data = {
        "mode": "ops",
        "prompt": custom_prompt,
        "topic": "La inteligencia artificial en el mundo moderno"
    }
    
    print("Testing custom prompt injection...")
    print(f"Custom prompt: {custom_prompt[:100]}...")
    
    try:
        response = requests.post(
            "https://marco-voice-engine.fly.dev/generate",
            json=test_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Request successful!")
            print(f"Received {len(result.get('variants', []))} variants")
            
            # Check if any variant contains "PERRO"
            variants_with_perro = []
            for i, variant in enumerate(result.get('variants', [])):
                text = variant.get('text', '')
                if 'perro' in text.lower():
                    variants_with_perro.append((i, text))
                print(f"Variant {i+1}: {text}")
                print(f"Score: {variant.get('score', 'N/A')}")
                print("---")
            
            if variants_with_perro:
                print(f"🎉 SUCCESS! Found 'PERRO' in {len(variants_with_perro)} variants:")
                for idx, text in variants_with_perro:
                    print(f"  Variant {idx+1}: {text}")
            else:
                print("❌ No variants contained 'PERRO' - custom prompt may not be working")
                
        else:
            print(f"❌ Request failed with status {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error testing custom prompt: {e}")

def test_default_prompt():
    """Test that default prompts still work"""
    
    test_data = {
        "mode": "ops",
        "topic": "La inteligencia artificial en el mundo moderno"
        # No custom prompt - should use default
    }
    
    print("\nTesting default prompt (no custom prompt)...")
    
    try:
        response = requests.post(
            "https://marco-voice-engine.fly.dev/generate",
            json=test_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Default prompt request successful!")
            print(f"Received {len(result.get('variants', []))} variants")
            
            for i, variant in enumerate(result.get('variants', [])):
                print(f"Variant {i+1}: {variant.get('text', '')}")
                print(f"Score: {variant.get('score', 'N/A')}")
                print("---")
        else:
            print(f"❌ Default prompt request failed with status {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error testing default prompt: {e}")

if __name__ == "__main__":
    test_custom_prompt()
    test_default_prompt()