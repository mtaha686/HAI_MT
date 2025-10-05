#!/usr/bin/env python3
"""
Example script to test the herbal medicine chatbot
"""

import requests
import json
import time

API_URL = "http://localhost:8000"

def test_api_health():
    """Test if API is running"""
    try:
        response = requests.get(f"{API_URL}/health")
        if response.status_code == 200:
            print("✅ API is healthy")
            return True
        else:
            print(f"❌ API health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Make sure it's running on http://localhost:8000")
        return False

def ask_question(question):
    """Ask a question to the chatbot"""
    try:
        response = requests.post(
            f"{API_URL}/chat",
            json={"message": question},
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            data = response.json()
            return data
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Error asking question: {e}")
        return None

def main():
    """Run example interactions"""
    print("🌿 Herbal Medicine Chatbot - Example Usage")
    print("=" * 50)
    
    # Check API health
    if not test_api_health():
        print("\n💡 Start the API server first:")
        print("   python src/api.py")
        return
    
    # Example questions
    questions = [
        "What are the uses of Sokhrus?",
        "How do you prepare Chamomile?",
        "What are the side effects of Astragalus psilocentros?",
        "Tell me about Rhodiola imbricata",
        "Which parts of Equisetum arvense are used medicinally?",
        "What family does Capparis spinosa belong to?",
        "Is Nepeta erecta safe to use?",
        "How is Tussilago farfara prepared?",
        "What are the benefits of Swertia cordata?",
        "Where is Primula macrophylla found?"
    ]
    
    print(f"\n🤖 Testing {len(questions)} example questions...\n")
    
    for i, question in enumerate(questions, 1):
        print(f"📝 Question {i}: {question}")
        
        start_time = time.time()
        result = ask_question(question)
        
        if result:
            print(f"🤖 Response: {result['response']}")
            # Handle optional fields that might not be in the response
            if 'confidence' in result:
                print(f"📊 Confidence: {result['confidence']:.2f}")
            if 'response_time' in result:
                print(f"⏱️ Response time: {result['response_time']:.2f}s")
        else:
            print("❌ Failed to get response")
        
        print("-" * 50)
        
        # Small delay between questions
        time.sleep(1)
    
    print("\n✅ Example completed!")
    print("\n💡 Try the web interface at: http://localhost:3000")

if __name__ == "__main__":
    main()