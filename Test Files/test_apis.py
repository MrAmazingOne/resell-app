#!/usr/bin/env python3
"""
Test eBay Taxonomy and Marketing APIs
"""

import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

def test_taxonomy_api():
    """Test Taxonomy API access"""
    print("\n🔍 Testing eBay Taxonomy API...")
    
    auth_token = os.getenv('EBAY_AUTH_TOKEN')
    if not auth_token:
        print("❌ No OAuth token available")
        return False
    
    try:
        headers = {
            'Authorization': f'Bearer {auth_token}',
            'Content-Type': 'application/json'
        }
        
        # Test 1: Get default category tree ID
        print("  1. Getting default category tree ID...")
        tree_url = "https://api.ebay.com/commerce/taxonomy/v1_beta/get_default_category_tree_id"
        response = requests.get(tree_url, headers=headers, timeout=10)
        
        print(f"    Status: {response.status_code}")
        
        if response.status_code == 200:
            tree_data = response.json()
            tree_id = tree_data.get('categoryTreeId')
            print(f"    ✅ Success! Tree ID: {tree_id}")
            
            # Test 2: Get category suggestions
            print("\n  2. Testing category suggestions...")
            suggest_url = f"https://api.ebay.com/commerce/taxonomy/v1_beta/category_tree/{tree_id}/get_category_suggestions"
            params = {'q': 'iphone 14 pro'}
            
            suggest_response = requests.get(suggest_url, headers=headers, params=params, timeout=10)
            print(f"    Status: {suggest_response.status_code}")
            
            if suggest_response.status_code == 200:
                suggestions = suggest_response.json().get('categorySuggestions', [])
                print(f"    ✅ Found {len(suggestions)} category suggestions")
                if suggestions:
                    print(f"    Top suggestion: {suggestions[0]['category']['categoryName']} (ID: {suggestions[0]['category']['categoryId']})")
                return True
            else:
                print(f"    ❌ Category suggestions failed: {suggest_response.text[:200]}")
                return False
        else:
            print(f"    ❌ Failed to get tree ID: {response.text[:200]}")
            return False
            
    except Exception as e:
        print(f"❌ Taxonomy API test error: {e}")
        return False

def test_marketing_api():
    """Test Marketing API access"""
    print("\n🔍 Testing eBay Marketing API...")
    
    auth_token = os.getenv('EBAY_AUTH_TOKEN')
    if not auth_token:
        print("❌ No OAuth token available")
        return False
    
    try:
        headers = {
            'Authorization': f'Bearer {auth_token}',
            'Content-Type': 'application/json'
        }
        
        # First, check if we need to create a test campaign
        print("  1. Checking for existing campaigns...")
        campaigns_url = "https://api.ebay.com/sell/marketing/v1/ad_campaign"
        response = requests.get(campaigns_url, headers=headers, timeout=10)
        
        print(f"    Status: {response.status_code}")
        
        if response.status_code == 200:
            campaigns = response.json().get('campaigns', [])
            if campaigns:
                campaign_id = campaigns[0]['campaignId']
                print(f"    ✅ Found campaign: {campaign_id}")
                
                # Get ad groups for this campaign
                ad_groups_url = f"https://api.ebay.com/sell/marketing/v1/ad_campaign/{campaign_id}/ad_group"
                ad_response = requests.get(ad_groups_url, headers=headers, timeout=10)
                
                if ad_response.status_code == 200:
                    ad_groups = ad_response.json().get('adGroups', [])
                    if ad_groups:
                        ad_group_id = ad_groups[0]['adGroupId']
                        print(f"    ✅ Found ad group: {ad_group_id}")
                        
                        # Test keyword suggestions
                        print("\n  2. Testing keyword suggestions...")
                        keywords_url = f"https://api.ebay.com/sell/marketing/v1/ad_campaign/{campaign_id}/ad_group/{ad_group_id}/suggest_keywords"
                        
                        keyword_data = {
                            "keywords": ["iphone", "smartphone"],
                            "maxNumOfKeywords": 10
                        }
                        
                        keyword_response = requests.post(keywords_url, headers=headers, json=keyword_data, timeout=10)
                        print(f"    Status: {keyword_response.status_code}")
                        
                        if keyword_response.status_code == 200:
                            suggestions = keyword_response.json()
                            print(f"    ✅ Keyword suggestions working!")
                            return True
                        else:
                            print(f"    ❌ Keyword suggestions failed: {keyword_response.text[:200]}")
                            return False
                else:
                    print(f"    ❌ No ad groups found: {ad_response.text[:200]}")
            else:
                print("    ⚠️ No campaigns found. Marketing API may require campaign setup.")
                return False
        else:
            print(f"    ❌ Campaign check failed: {response.text[:200]}")
            return False
            
    except Exception as e:
        print(f"❌ Marketing API test error: {e}")
        return False

def test_scopes():
    """Check what scopes our token has"""
    print("\n🔍 Checking OAuth token scopes...")
    
    auth_token = os.getenv('EBAY_AUTH_TOKEN')
    if not auth_token:
        print("❌ No OAuth token available")
        return
    
    try:
        # The token itself contains scope information
        # We can decode it (it's a JWT)
        import jwt
        
        # Try to decode without verification first
        try:
            decoded = jwt.decode(auth_token, options={"verify_signature": False})
            scopes = decoded.get('scope', '').split()
            print(f"✅ Token scopes found:")
            for scope in scopes:
                print(f"   - {scope}")
            
            # Check for required scopes
            required_scopes = [
                'https://api.ebay.com/oauth/api_scope',
                'https://api.ebay.com/oauth/api_scope/commerce.taxonomy',
                'https://api.ebay.com/oauth/api_scope/sell.marketing'
            ]
            
            print("\n🔍 Checking required scopes:")
            for required in required_scopes:
                if required in scopes:
                    print(f"   ✅ {required}")
                else:
                    print(f"   ❌ {required} (MISSING)")
                    
        except jwt.DecodeError:
            print("⚠️ Could not decode token (might not be JWT)")
            
    except ImportError:
        print("⚠️ Install pyjwt to decode token: pip install pyjwt")
        print("   Alternatively, check scopes at: https://developer.ebay.com/my/keys")

def test_browse_api():
    """Test basic Browse API access"""
    print("\n🔍 Testing Browse API (basic)...")
    
    auth_token = os.getenv('EBAY_AUTH_TOKEN')
    if not auth_token:
        print("❌ No OAuth token available")
        return False
    
    try:
        headers = {
            'Authorization': f'Bearer {auth_token}',
            'Content-Type': 'application/json',
            'X-EBAY-C-MARKETPLACE-ID': 'EBAY_US'
        }
        
        # Simple search test
        url = "https://api.ebay.com/buy/browse/v1/item_summary/search"
        params = {
            'q': 'test',
            'limit': '1',
            'filter': 'soldItems:true'
        }
        
        response = requests.get(url, headers=headers, params=params, timeout=10)
        print(f"    Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            items = data.get('itemSummaries', [])
            print(f"    ✅ Browse API working! Found {len(items)} items")
            return True
        else:
            print(f"    ❌ Browse API failed: {response.text[:200]}")
            return False
            
    except Exception as e:
        print(f"❌ Browse API test error: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("eBay API Access Test")
    print("=" * 60)
    
    auth_token = os.getenv('EBAY_AUTH_TOKEN')
    app_id = os.getenv('EBAY_APP_ID')
    
    print(f"App ID: {app_id[:10]}..." if app_id else "❌ App ID: NOT SET")
    print(f"Auth Token: {auth_token[:10]}..." if auth_token else "❌ Auth Token: NOT SET")
    
    if not auth_token:
        print("\n⚠️ Please get an OAuth token first:")
        print("  1. Visit /ebay/oauth/start")
        print("  2. Authorize the app")
        print("  3. Run this test again")
        exit(1)
    
    # Test APIs
    browse_ok = test_browse_api()
    taxonomy_ok = test_taxonomy_api()
    marketing_ok = test_marketing_api()
    test_scopes()
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(f"  Browse API: {'✅ WORKING' if browse_ok else '❌ FAILED'}")
    print(f"  Taxonomy API: {'✅ WORKING' if taxonomy_ok else '❌ FAILED'}")
    print(f"  Marketing API: {'✅ WORKING' if marketing_ok else '❌ FAILED'}")
    
    if not taxonomy_ok:
        print("\n⚠️ Taxonomy API issues:")
        print("  - Check that token has scope: https://api.ebay.com/oauth/api_scope/commerce.taxonomy")
        print("  - Try updating OAuth with correct scopes")
    
    if not marketing_ok:
        print("\n⚠️ Marketing API issues:")
        print("  - May require setting up a campaign in Seller Hub first")
        print("  - Check token scope: https://api.ebay.com/oauth/api_scope/sell.marketing")
    
    if browse_ok and not (taxonomy_ok and marketing_ok):
        print("\n💡 SUGGESTION:")
        print("  The Browse API works but others don't. You might need to:")
        print("  1. Get a new OAuth token with all required scopes")
        print("  2. Update /ebay/oauth/start to request: commerce.taxonomy and sell.marketing scopes")
    
    print("=" * 60)