#!/usr/bin/env python3
"""
Check if eBay credentials are valid
"""
import os
from ebaysdk.trading import Connection as TradingConnection
from ebaysdk.exception import ConnectionError

def check_credentials():
    print("🔍 Checking eBay credentials...")
    
    # Test each credential combination
    test_cases = [
        {
            'name': 'Your current credentials',
            'appid': 'JustinHa-ReReSell-SBX-e11823bc1-b28e2f71',
            'certid': 'SBX-l1823bcla4a9-aa79-4e52-8a77-735e',
            'devid': 'a3376eb0-d27c-46a6-9a04-e6fdfed5e5fc'
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