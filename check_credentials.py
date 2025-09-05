#!/usr/bin/env python3
"""
Check if eBay credentials are valid
"""
import os
from ebaysdk.trading import Connection as TradingConnection
from ebaysdk.exception import ConnectionError
from dotenv import load_dotenv

load_dotenv()

def check_credentials():
    print("🔍 Checking eBay credentials...")
    
    # Get credentials from environment
    appid = os.getenv('EBAY_APP_ID')
    certid = os.getenv('EBAY_CERT_ID')
    devid = os.getenv('EBAY_DEV_ID')
    
    if not all([appid, certid, devid]):
        print("❌ Missing eBay credentials in environment variables")
        return None
    
    # Test each credential combination
    test_cases = [
        {
            'name': 'Your current credentials',
            'appid': appid,
            'certid': certid,
            'devid': devid
        }
    ]
    
    for i, creds in enumerate(test_cases):
        print(f"\n🧪 Test {i+1}: {creds['name']}")
        try:
            api = TradingConnection(
                config_file=None,  # ← CRITICAL: Added config_file=None
                appid=creds['appid'],
                certid=creds['certid'], 
                devid=creds['devid'],
                domain='api.sandbox.ebay.com',
                warnings=True
            )
            
            # Simple API call to test credentials
            response = api.execute('GeteBayOfficialTime', {})
            print(f"✅ SUCCESS: Credentials work!")
            print(f"   eBay Time: {response.reply.Timestamp}")
            return creds
            
        except ConnectionError as e:
            print(f"❌ FAILED: {e}")
            # Print more details about the error
            if hasattr(e, 'response'):
                print(f"   Response: {e.response.reply}")
    
    return None

if __name__ == "__main__":
    check_credentials()