# test_simple_request.py
import requests
import json
from dotenv import load_dotenv
import os

load_dotenv()

def test_simple_ebay_request():
    print("🔍 Testing simple eBay API request...")
    
    app_id = os.getenv('EBAY_APP_ID')
    
    if not app_id:
        print("❌ No EBAY_APP_ID found in environment")
        return
    
    # Try a simple Finding API call
    url = "https://svcs.sandbox.ebay.com/services/search/FindingService/v1"
    
    headers = {
        'X-EBAY-SOA-SECURITY-APPNAME': app_id,
        'X-EBAY-SOA-OPERATION-NAME': 'findItemsByKeywords',
        'X-EBAY-SOA-SERVICE-VERSION': '1.0.0',
        'X-EBAY-SOA-GLOBAL-ID': 'EBAY-US',
        'X-EBAY-SOA-RESPONSE-DATA-FORMAT': 'JSON'
    }
    
    data = {
        'keywords': 'iphone'
    }
    
    try:
        response = requests.post(url, headers=headers, data=data, timeout=15)
        print(f"✅ Response status: {response.status_code}")
        if response.status_code == 200:
            print("✅ Simple eBay API request successful!")
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"❌ Request failed with status: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Request failed: {e}")

if __name__ == "__main__":
    test_simple_ebay_request()