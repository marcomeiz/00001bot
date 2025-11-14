#!/usr/bin/env python3
"""
Test the API directly with custom prompts.
"""

import requests
import json

def test_api_custom_prompt():
    print("🧪 Testing API with Custom Prompt")
    print("=" * 50)
    
    # Test with custom prompt
    custom_prompt = "Generate tweets about the topic. Include the word PERRO somewhere in the text."
    
    print(f"Custom prompt: {custom_prompt}")
    print()
    
    try:
        response = requests.post(
            "http://localhost:8000/generate",
            json={"mode": "ops", "prompt": custom_prompt},
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Generation successful!")
            print(f"Topic: {data.get('topic')}")
            print(f"Received {len(data.get('variants', []))} variants:")
            
            perro_found = False
            for i, variant in enumerate(data.get('variants', [])):
                text = variant.get('text', '')
                score = variant.get('score', 0)
                print(f"Variant {i+1} (score: {score}): {text}")
                if 'PERRO' in text:
                    perro_found = True
                    print(f"✅ PERRO found in variant {i+1}!")
            
            if not perro_found:
                print("❌ PERRO not found in any variant")
                return False
            else:
                return True
        else:
            print(f"❌ API error: {response.status_code}")
            print(response.text)
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_api_custom_prompt()
    if success:
        print("\n🎉 SUCCESS: Custom prompt injection working!")
    else:
        print("\n❌ FAILED: Custom prompt injection not working")