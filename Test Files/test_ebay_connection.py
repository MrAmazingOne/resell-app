# test_ebay_connection.py
import os
import requests
from dotenv import load_dotenv

load_dotenv()

def test_ebay_connection():
    print("🔍 Testing eBay API connection with current credentials...")
    
    # Check all required environment variables
    required_vars = ['EBAY_APP_ID', 'EBAY_CERT_ID', 'EBAY_DEV_ID']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {missing_vars}")
        return False
    
    app_id = os.getenv('EBAY_APP_ID')
    cert_id = os.getenv('EBAY_CERT_ID')
    dev_id = os.getenv('EBAY_DEV_ID')
    
    print(f"Using App ID: {app_id[:10]}...{app_id[-10:]}")
    
    # Test 1: Basic API connectivity
    print("\n1. Testing basic API connectivity...")
    try:
        response = requests.get(
            'https://api.sandbox.ebay.com/ws/api.dll',
            headers={'X-EBAY-API-IAF-TOKEN': f'Bearer {app_id}'},
            timeout=10
        )
        print(f"✅ API endpoint reachable (Status: {response.status_code})")
    except Exception as e:
        print(f"❌ API endpoint unreachable: {e}")
        return False
    
    # Test 2: Try a simple Finding API call
    print("\n2. Testing Finding API...")
    try:
        headers = {
            'X-EBAY-SOA-SECURITY-APPNAME': app_id,
            'X-EBAY-SOA-OPERATION-NAME': 'findItemsByKeywords',
            'X-EBAY-SOA-SERVICE-VERSION': '1.0.0',
            'X-EBAY-SOA-GLOBAL-ID': 'EBAY-US',
            'X-EBAY-SOA-RESPONSE-DATA-FORMAT': 'JSON',
            'Content-Type': 'text/xml'
        }
        
        # Simple XML request
        xml_request = '''<?xml version="1.0" encoding="UTF-8"?>
        <findItemsByKeywordsRequest xmlns="http://www.ebay.com/marketplace/search/v1/services">
            <keywords>iphone</keywords>
            <paginationInput>
                <entriesPerPage>1</entriesPerPage>
            </paginationInput>
        </findItemsByKeywordsRequest>'''
        
        response = requests.post(
            'https://svcs.sandbox.ebay.com/services/search/FindingService/v1',
            headers=headers,
            data=xml_request,
            timeout=15
        )
        
        print(f"✅ Finding API response: {response.status_code}")
        if response.status_code == 200:
            print("✅ Finding API test successful!")
        else:
            print(f"❌ Finding API error: {response.text[:200]}...")
            
    except Exception as e:
        print(f"❌ Finding API test failed: {e}")
        return False
    
    # Test 3: Check if credentials might be invalid
    print("\n3. Testing credential validity...")
    try:
        # Try to get eBay official time (simpler than Trading API)
        headers = {
            'X-EBAY-API-CALL-NAME': 'GeteBayOfficialTime',
            'X-EBAY-API-APP-NAME': app_id,
            'X-EBAY-API-CERT-NAME': cert_id,
            'X-EBAY-API-DEV-NAME': dev_id,
            'X-EBAY-API-SITEID': '0',
            'X-EBAY-API-COMPATIBILITY-LEVEL': '967',
            'Content-Type': 'text/xml'
        }
        
        xml_request = '''<?xml version="1.0" encoding="UTF-8"?>
        <GeteBayOfficialTimeRequest xmlns="urn:ebay:apis:eBLBaseComponents">
            <RequesterCredentials>
                <eBayAuthToken>AuthToken</eBayAuthToken>
            </RequesterCredentials>
        </GeteBayOfficialTimeRequest>'''
        
        response = requests.post(
            'https://api.sandbox.ebay.com/ws/api.dll',
            headers=headers,
            data=xml_request,
            timeout=15
        )
        
        print(f"✅ Credential test response: {response.status_code}")
        if response.status_code == 200:
            print("✅ Credentials appear valid!")
        elif response.status_code == 401:
            print("❌ Credentials rejected (401 Unauthorized)")
            return False
        else:
            print(f"⚠️  Unexpected response: {response.text[:200]}...")
            
    except Exception as e:
        print(f"❌ Credential test failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    test_ebay_connection()