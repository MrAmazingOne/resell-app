#!/usr/bin/env python3
"""
Test OAuth Token Access to eBay APIs
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

def test_oauth_token():
    """Test OAuth token directly"""
    print("=" * 60)
    print("eBay OAuth Token Test")
    print("=" * 60)
    
    token = os.getenv('EBAY_AUTH_TOKEN')
    if not token:
        print("❌ EBAY_AUTH_TOKEN not set")
        return False
    
    print(f"Token preview: {token[:50]}...")
    
    # Test 1: Basic Browse API
    print("\n🔍 Test 1: Browse API (basic search)...")
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json',
        'X-EBAY-C-MARKETPLACE-ID': 'EBAY_US'
    }
    
    try:
        response = requests.get(
            'https://api.ebay.com/buy/browse/v1/item_summary/search',
            headers=headers,
            params={'q': 'iphone', 'limit': '1', 'filter': 'soldItems:true'},
            timeout=10
        )
        
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            print("   ✅ Browse API ACCESS GRANTED")
            data = response.json()
            print(f"   Found {len(data.get('itemSummaries', []))} items")
            return True
        elif response.status_code == 401:
            print("   ❌ OAuth token INVALID or EXPIRED")
            print("   Response:", response.text[:200])
        elif response.status_code == 403:
            print("   ❌ OAuth token missing Browse API scope")
        else:
            print(f"   ⚠️ Unexpected: {response.status_code}")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    return False

def test_taxonomy_access():
    """Test Taxonomy API access"""
    print("\n🔍 Test 2: Taxonomy API...")
    
    token = os.getenv('EBAY_AUTH_TOKEN')
    if not token:
        return False
    
    headers = {'Authorization': f'Bearer {token}'}
    
    try:
        # Get category tree
        response = requests.get(
            'https://api.ebay.com/commerce/taxonomy/v1_beta/get_default_category_tree_id',
            headers=headers,
            timeout=10
        )
        
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            print("   ✅ Taxonomy API ACCESS GRANTED")
            tree_data = response.json()
            print(f"   Tree ID: {tree_data.get('categoryTreeId', 'Unknown')}")
            return True
        elif response.status_code == 401:
            print("   ❌ OAuth token invalid for Taxonomy API")
        elif response.status_code == 403:
            print("   ❌ Missing scope: commerce.taxonomy")
        else:
            print(f"   ⚠️ Status: {response.status_code}")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    return False

def test_marketing_access():
    """Test Marketing API access"""
    print("\n🔍 Test 3: Marketing API...")
    
    token = os.getenv('EBAY_AUTH_TOKEN')
    if not token:
        return False
    
    headers = {'Authorization': f'Bearer {token}'}
    
    try:
        response = requests.get(
            'https://api.ebay.com/sell/marketing/v1/ad_campaign',
            headers=headers,
            timeout=10
        )
        
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            print("   ✅ Marketing API ACCESS GRANTED")
            data = response.json()
            campaigns = data.get('campaigns', [])
            print(f"   Found {len(campaigns)} campaigns")
            return True
        elif response.status_code == 401:
            print("   ❌ OAuth token invalid for Marketing API")
        elif response.status_code == 403:
            print("   ❌ Missing scope: sell.marketing.readonly")
        else:
            print(f"   ⚠️ Status: {response.status_code}")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    return False

if __name__ == "__main__":
    # Test all APIs
    browse_ok = test_oauth_token()
    taxonomy_ok = test_taxonomy_access()
    marketing_ok = test_marketing_access()
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(f"  Browse API:   {'✅ WORKING' if browse_ok else '❌ FAILED'}")
    print(f"  Taxonomy API: {'✅ WORKING' if taxonomy_ok else '❌ FAILED'}")
    print(f"  Marketing API: {'✅ WORKING' if marketing_ok else '❌ FAILED'}")
    
    if not browse_ok:
        print("\n🚨 URGENT: OAuth token is INVALID")
        print("   Get new token from: /ebay/oauth/start")
        print("   Update EBAY_AUTH_TOKEN in both .env and Render")
    
    if browse_ok and not taxonomy_ok:
        print("\n⚠️ Missing Taxonomy API scope")
        print("   When getting new token, ensure 'commerce.taxonomy' scope is checked")
    
    if browse_ok and not marketing_ok:
        print("\n⚠️ Missing Marketing API scope")
        print("   When getting new token, ensure 'sell.marketing.readonly' scope is checked")
    
    print("=" * 60)