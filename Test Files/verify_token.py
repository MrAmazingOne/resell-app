#!/usr/bin/env python3
"""
Verify eBay OAuth Token and Scopes
"""

import os
import requests
import base64
import json
from dotenv import load_dotenv

load_dotenv()

def decode_jwt_token(token):
    """Decode JWT token without verification to see scopes"""
    try:
        # Split JWT token
        parts = token.split('.')
        if len(parts) != 3:
            print("❌ Token is not a valid JWT")
            return None
        
        # Decode payload (middle part)
        payload = parts[1]
        # Add padding if needed
        padding = 4 - len(payload) % 4
        if padding != 4:
            payload += "=" * padding
        
        decoded = base64.urlsafe_b64decode(payload)
        return json.loads(decoded)
    except Exception as e:
        print(f"❌ Failed to decode token: {e}")
        return None

def test_token_scopes():
    """Test what scopes the token has"""
    print("\n🔍 Testing Token Scopes...")
    
    auth_token = os.getenv('EBAY_AUTH_TOKEN')
    if not auth_token:
        print("❌ No OAuth token in environment")
        return False
    
    # Decode token to see scopes
    decoded = decode_jwt_token(auth_token)
    if decoded:
        print("✅ Token successfully decoded")
        scopes = decoded.get('scope', '').split()
        exp_time = decoded.get('exp', 0)
        
        print(f"\n📋 Token Details:")
        print(f"   Expires at (UTC): {exp_time}")
        
        if scopes:
            print(f"\n🔑 Token Scopes ({len(scopes)}):")
            required_scopes = [
                'https://api.ebay.com/oauth/api_scope',
                'https://api.ebay.com/oauth/api_scope/commerce.taxonomy',
                'https://api.ebay.com/oauth/api_scope/sell.marketing.readonly',
                'https://api.ebay.com/oauth/api_scope/sell.marketing'
            ]
            
            for scope in scopes:
                check = "✅" if scope in required_scopes else "  "
                print(f"   {check} {scope}")
            
            print(f"\n📊 Scope Coverage:")
            for required in required_scopes:
                if required in scopes:
                    print(f"   ✅ {required.split('/')[-1]}")
                else:
                    print(f"   ❌ {required.split('/')[-1]} (MISSING)")
            
            return all(scope in scopes for scope in required_scopes)
        else:
            print("❌ No scopes found in token")
            return False
    
    return False

def test_browse_api_with_fresh_check():
    """Test Browse API with fresh token check"""
    print("\n🔍 Testing Browse API with fresh check...")
    
    auth_token = os.getenv('EBAY_AUTH_TOKEN')
    if not auth_token:
        print("❌ No OAuth token available")
        return False
    
    try:
        # First, verify the token is valid by calling a simple endpoint
        headers = {
            'Authorization': f'Bearer {auth_token}',
            'Content-Type': 'application/json',
            'X-EBAY-C-MARKETPLACE-ID': 'EBAY_US'
        }
        
        # Simple test - get item by known ID (e.g., a test item)
        test_item_id = "v1|382732719017|0"  # This is a sample format
        url = f"https://api.ebay.com/buy/browse/v1/item/{test_item_id}"
        
        print("   Testing token validity with simple GET...")
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            print("   ✅ Token is valid!")
            return True
        elif response.status_code == 401:
            print("   ❌ Token is INVALID (401 Unauthorized)")
            print(f"   Response: {response.text[:200]}")
            return False
        elif response.status_code == 403:
            print("   ⚠️ Token valid but missing scopes (403 Forbidden)")
            return False
        else:
            print(f"   ⚠️ Unexpected status: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False

def check_environment():
    """Check environment setup"""
    print("\n🔍 Checking Environment...")
    
    auth_token = os.getenv('EBAY_AUTH_TOKEN')
    app_id = os.getenv('EBAY_APP_ID')
    
    issues = []
    
    if not auth_token:
        issues.append("❌ EBAY_AUTH_TOKEN not set")
    else:
        print(f"✅ EBAY_AUTH_TOKEN: {auth_token[:20]}...")
    
    if not app_id:
        issues.append("❌ EBAY_APP_ID not set")
    else:
        print(f"✅ EBAY_APP_ID: {app_id}")
    
    # Check token format
    if auth_token:
        if auth_token.startswith('v^'):
            print("✅ Token format appears correct (eBay OAuth)")
        else:
            issues.append("⚠️ Token doesn't match eBay OAuth format")
    
    return len(issues) == 0

if __name__ == "__main__":
    print("=" * 60)
    print("eBay OAuth Token Verification")
    print("=" * 60)
    
    # Check environment
    env_ok = check_environment()
    
    if not env_ok:
        print("\n❌ Environment issues found. Fix before proceeding.")
        exit(1)
    
    # Test token scopes
    scopes_ok = test_token_scopes()
    
    # Test API access
    api_ok = test_browse_api_with_fresh_check()
    
    print("\n" + "=" * 60)
    print("DIAGNOSTIC RESULTS:")
    print("=" * 60)
    
    if not api_ok:
        print("\n🚨 CRITICAL ISSUE: Token is INVALID or EXPIRED")
        print("\n🔧 FIX STEPS:")
        print("1. Go to: https://resell-app-bi47.onrender.com/ebay/oauth/start")
        print("2. Click 'Authorize' on eBay")
        print("3. Make sure ALL scopes are checked")
        print("4. Get a NEW token")
        print("5. Update EBAY_AUTH_TOKEN in Render environment")
        print("\n💡 Or use direct URL:")
        print(f"https://auth.ebay.com/oauth2/authorize?client_id=JustinHa-ReReSell-PRD-b118c3532-f23588a1&response_type=code&redirect_uri=https%3A%2F%2Fresell-app-bi47.onrender.com%2Febay%2Foauth%2Fcallback&scope=https%3A%2F%2Fapi.ebay.com%2Foauth%2Fapi_scope%20https%3A%2F%2Fapi.ebay.com%2Foauth%2Fapi_scope%2Fcommerce.taxonomy%20https%3A%2F%2Fapi.ebay.com%2Foauth%2Fapi_scope%2Fsell.marketing.readonly&state=verify123&prompt=login&duration=permanent")
    
    elif not scopes_ok:
        print("\n⚠️ WARNING: Token missing required scopes")
        print("\n🔧 FIX: Get a new token with ALL scopes checked")
        
    else:
        print("\n✅ SUCCESS: Token is valid with all required scopes!")
        print("\nNext: Run the enhanced agent with full API access.")
    
    print("=" * 60)