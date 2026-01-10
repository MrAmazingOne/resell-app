import requests

# Test different scopes
test_scopes = [
    "https://api.ebay.com/oauth/api_scope",
    "https://api.ebay.com/oauth/api_scope/commerce.taxonomy",
    "https://api.ebay.com/oauth/api_scope/sell.marketing.readonly",
    "https://api.ebay.com/oauth/api_scope/sell.marketing",
    "https://api.ebay.com/oauth/api_scope/sell.inventory",
    "https://api.ebay.com/oauth/api_scope/sell.account"
]

print("🔍 Checking eBay OAuth scopes...")
print("=" * 60)

for scope in test_scopes:
    # Create a test auth URL
    auth_url = f"https://auth.ebay.com/oauth2/authorize?client_id=DUMMY&response_type=code&redirect_uri=https://test.com&scope={scope}"
    print(f"{scope}")
    print(f"   Last part: {scope.split('/')[-1]}")
print("=" * 60)