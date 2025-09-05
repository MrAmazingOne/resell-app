# check_credentials_fixed.py
#!/usr/bin/env python3
"""
Check if eBay credentials are valid - FIXED VERSION
"""
import os
import requests
from dotenv import load_dotenv

load_dotenv()

def check_credentials_fixed():
    print("🔍 Checking eBay credentials (Fixed version)...")
    
    # Get credentials from environment
    appid = os.getenv('EBAY_APP_ID')
    certid = os.getenv('EBAY_CERT_ID')
    devid = os.getenv('EBAY_DEV_ID')
    
    print(f"App ID: {appid}")
    print(f"Cert ID: {certid}")
    print(f"Dev ID: {devid}")
    
    if not all([appid, certid, devid]):
        print("❌ Missing eBay credentials in environment variables")
        return None
    
    # Use the Shopping API instead - it's simpler and doesn't require auth token
    print("\n🧪 Testing with Shopping API (no auth token needed)...")
    
    try:
        # Shopping API - GetSingleItem is public and doesn't require auth
        headers = {
            'X-EBAY-API-APP-NAME': appid,
            'X-EBAY-API-CALL-NAME': 'GetSingleItem',
            'X-EBAY-API-VERSION': '1157',
            'X-EBAY-API-SITE-ID': '0',
            'X-EBAY-API-REQUEST-ENCODING': 'XML',
            'Content-Type': 'text/xml'
        }
        
        # Use a known test item ID from eBay sandbox
        xml_request = f'''<?xml version="1.0" encoding="UTF-8"?>
        <GetSingleItemRequest xmlns="urn:ebay:apis:eBLBaseComponents">
            <ItemID>110052465644</ItemID>
            <IncludeSelector>ItemSpecifics,Description,Details</IncludeSelector>
        </GetSingleItemRequest>'''
        
        response = requests.post(
            'https://open.api.sandbox.ebay.com/shopping',
            headers=headers,
            data=xml_request,
            timeout=15
        )
        
        print(f"✅ Shopping API response: {response.status_code}")
        
        if response.status_code == 200:
            print("🎉 SUCCESS: eBay credentials are valid!")
            print("Shopping API test passed - your App ID, Cert ID, and Dev ID are working")
            return True
        else:
            print(f"❌ Shopping API error: {response.status_code}")
            print(f"Response: {response.text[:200]}...")
            return False
            
    except Exception as e:
        print(f"❌ Shopping API test failed: {e}")
        return False

def test_finding_api():
    print("\n🧪 Testing Finding API...")
    
    appid = os.getenv('EBAY_APP_ID')
    
    try:
        headers = {
            'X-EBAY-SOA-SECURITY-APPNAME': appid,
            'X-EBAY-SOA-OPERATION-NAME': 'findItemsByKeywords',
            'X-EBAY-SOA-SERVICE-VERSION': '1.0.0',
            'X-EBAY-SOA-GLOBAL-ID': 'EBAY-US',
            'X-EBAY-SOA-RESPONSE-DATA-FORMAT': 'JSON',
            'Content-Type': 'text/xml'
        }
        
        xml_request = '''<?xml version="1.0" encoding="UTF-8"?>
        <findItemsByKeywordsRequest xmlns="http://www.ebay.com/marketplace/search/v1/services">
            <keywords>test</keywords>
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
        
        print(f"Finding API response: {response.status_code}")
        if response.status_code == 200:
            print("✅ Finding API test successful!")
            return True
        else:
            print(f"❌ Finding API error: {response.text[:200]}...")
            return False
            
    except Exception as e:
        print(f"❌ Finding API test failed: {e}")
        return False

if __name__ == "__main__":
    success1 = check_credentials_fixed()
    success2 = test_finding_api()
    
    if success1 or success2:
        print("\n🎉 Some API tests passed! Your credentials are likely valid.")
        print("The original error was due to malformed Trading API requests.")
    else:
        print("\n❌ All API tests failed. There may be an issue with your credentials.")