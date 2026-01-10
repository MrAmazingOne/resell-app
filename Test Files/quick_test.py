import os
import requests
from dotenv import load_dotenv

load_dotenv()

token = os.getenv('EBAY_AUTH_TOKEN')
print(f"Token: {token[:50]}...")

# Test Browse API
headers = {
    'Authorization': f'Bearer {token}',
    'Content-Type': 'application/json',
    'X-EBAY-C-MARKETPLACE-ID': 'EBAY_US'
}

print("\n🔍 Testing Browse API...")
response = requests.get(
    'https://api.ebay.com/buy/browse/v1/item_summary/search',
    headers=headers,
    params={'q': 'iphone', 'limit': '1', 'filter': 'soldItems:true'},
    timeout=10
)
print(f"Status: {response.status_code}")
print(f"Response: {response.text[:200]}")