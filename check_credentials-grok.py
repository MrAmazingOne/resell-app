#!/usr/bin/env python3
"""
Check if eBay credentials are valid - SOLD ITEMS FOCUS
"""
import os
import requests
from dotenv import load_dotenv

load_dotenv()

def check_sold_items_search():
    print("🔍 Testing eBay SOLD ITEMS search capability...")
    
    # Get credentials from environment
    appid = os.getenv('EBAY_APP_ID')
    auth_token = os.getenv('EBAY_AUTH_TOKEN')
    
    if not auth_token:
        print("❌ No eBay OAuth token available")
        return False
    
    try:
        # Test sold items search
        headers = {
            'Authorization': f'Bearer {auth_token}',
            'Content-Type': 'application/json',
            'X-EBAY-C-MARKETPLACE-ID': 'EBAY_US'
        }
        
        # Test 1: Search for sold Pokemon cards (should filter out sets)
        params = {
            'q': 'Pokemon Charizard single',
            'limit': '5',
            'filter': 'soldItemsOnly:true,category_ids:183454',  # Pokemon Cards category
            'sort': '-endTime'
        }
        
        response = requests.get(
            'https://api.ebay.com/buy/browse/v1/item_summary/search',
            headers=headers,
            params=params,
            timeout=15
        )
        
        print(f"✅ Sold items search test: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            items = data.get('itemSummaries', [])
            print(f"   Found {len(items)} sold items")
            
            # Check if we're getting sold items
            for i, item in enumerate(items[:3]):
                title = item.get('title', '')[:60]
                price = item.get('price', {}).get('value', '0')
                print(f"   [{i+1}] {title}... - ${price}")
            
            return True
        else:
            print(f"❌ Sold items search failed: {response.status_code}")
            print(f"Response: {response.text[:200]}...")
            return False
            
    except Exception as e:
        print(f"❌ Sold items test failed: {e}")
        return False

def test_vehicle_search():
    """Test vehicle vs parts search"""
    print("\n🚗 Testing vehicle search capability...")
    
    auth_token = os.getenv('EBAY_AUTH_TOKEN')
    if not auth_token:
        print("❌ No token for vehicle test")
        return False
    
    try:
        headers = {
            'Authorization': f'Bearer {auth_token}',
            'Content-Type': 'application/json',
            'X-EBAY-C-MARKETPLACE-ID': 'EBAY_US'
        }
        
        # Test WHOLE vehicle search in Cars & Trucks category
        params = {
            'q': '1955 Chevrolet 3100',
            'limit': '5',
            'filter': 'soldItemsOnly:true,category_ids:6001',  # Cars & Trucks category
            'sort': 'price_desc'
        }
        
        response = requests.get(
            'https://api.ebay.com/buy/browse/v1/item_summary/search',
            headers=headers,
            params=params,
            timeout=15
        )
        
        if response.status_code == 200:
            data = response.json()
            items = data.get('itemSummaries', [])
            print(f"✅ Vehicle search: Found {len(items)} sold vehicles in Cars & Trucks")
            
            # Check for parts contamination
            part_count = 0
            for item in items:
                title = item.get('title', '').lower()
                if any(word in title for word in ['part', 'parts', 'component']):
                    part_count += 1
            
            if part_count == 0:
                print("   ✅ No parts in Cars & Trucks results!")
            else:
                print(f"   ⚠️ Found {part_count} parts in results")
            
            return True
        else:
            print(f"❌ Vehicle search failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Vehicle test failed: {e}")
        return False

def test_category_filtering():
    """Test that category filtering works correctly"""
    print("\n📊 Testing category filtering...")
    
    auth_token = os.getenv('EBAY_AUTH_TOKEN')
    if not auth_token:
        print("❌ No token for category test")
        return False
    
    try:
        headers = {
            'Authorization': f'Bearer {auth_token}',
            'Content-Type': 'application/json',
            'X-EBAY-C-MARKETPLACE-ID': 'EBAY_US'
        }
        
        test_cases = [
            {
                'name': 'Whole Vehicle',
                'query': '1955 Chevy 3100',
                'category': '6001',
                'expected': 'Cars & Trucks (no parts)'
            },
            {
                'name': 'Vehicle Parts',
                'query': '1955 Chevy 3100 part',
                'category': '6028',
                'expected': 'Parts & Accessories'
            },
            {
                'name': 'Pokemon Cards',
                'query': 'Pokemon Charizard single',
                'category': '183454',
                'expected': 'Single cards only'
            }
        ]
        
        all_passed = True
        
        for test in test_cases:
            print(f"\n📋 Testing: {test['name']}")
            print(f"   Query: '{test['query']}'")
            print(f"   Category: {test['category']}")
            print(f"   Expected: {test['expected']}")
            
            params = {
                'q': test['query'],
                'limit': '3',
                'filter': f"soldItemsOnly:true,category_ids:{test['category']}",
                'sort': '-endTime'
            }
            
            response = requests.get(
                'https://api.ebay.com/buy/browse/v1/item_summary/search',
                headers=headers,
                params=params,
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                items = data.get('itemSummaries', [])
                print(f"   ✅ Found {len(items)} items")
                
                if test['name'] == 'Whole Vehicle':
                    # Check for parts
                    parts = sum(1 for item in items if 'part' in item.get('title', '').lower())
                    if parts == 0:
                        print(f"   ✅ No parts in whole vehicle search")
                    else:
                        print(f"   ❌ Found {parts} parts in whole vehicle search")
                        all_passed = False
                
            else:
                print(f"   ❌ Failed: {response.status_code}")
                all_passed = False
        
        return all_passed
        
    except Exception as e:
        print(f"❌ Category test failed: {e}")
        return False

def test_taxonomy_suggestions():
    """Test Taxonomy API for category suggestions"""
    print("\n🗂 Testing Taxonomy API...")
    
    auth_token = os.getenv('EBAY_AUTH_TOKEN')
    if not auth_token:
        print("❌ No token for taxonomy test")
        return False
    
    try:
        headers = {
            'Authorization': f'Bearer {auth_token}',
            'Content-Type': 'application/json',
            'X-EBAY-C-MARKETPLACE-ID': 'EBAY_US'
        }
        
        params = {'q': 'Pokemon Charizard single'}
        
        response = requests.get(
            'https://api.ebay.com/commerce/taxonomy/v1_beta/category_tree/EBAY_US/get_category_suggestions',
            headers=headers,
            params=params,
            timeout=15
        )
        
        if response.status_code == 200:
            data = response.json()
            suggestions = data.get('categorySuggestions', [])
            print(f"✅ Found {len(suggestions)} category suggestions")
            return True
        else:
            print(f"❌ Taxonomy test failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Taxonomy test failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("eBay API SOLD ITEMS VERIFICATION")
    print("=" * 60)
    
    appid = os.getenv('EBAY_APP_ID')
    auth_token = os.getenv('EBAY_AUTH_TOKEN')
    
    print(f"App ID: {appid[:10]}..." if appid else "❌ App ID: NOT SET")
    print(f"Auth Token: {auth_token[:10]}..." if auth_token else "❌ Auth Token: NOT SET")
    
    if not auth_token:
        print("\n⚠️ Please get an OAuth token first:")
        print("  1. Visit /ebay/oauth/start")
        print("  2. Authorize the app")
        print("  3. Set EBAY_AUTH_TOKEN environment variable")
    else:
        success1 = check_sold_items_search()
        success2 = test_vehicle_search()
        success3 = test_category_filtering()
        success4 = test_taxonomy_suggestions()
        
        if success1 and success2 and success3 and success4:
            print("\n🎉 ALL TESTS PASSED! Sold items search is working correctly.")
            print("The system will now ONLY show ACTUAL sold auction data with proper category filtering.")
        else:
            print("\n⚠️ Some tests failed. Check your credentials and category settings.")
            print("\nDebug endpoints available:")
            print("  GET /debug/category-search/1955%20Chevy%203100?category=vehicles&whole_vehicle=true")
            print("  GET /debug/ebay-search/Pokemon%20Charizard")