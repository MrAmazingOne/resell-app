# test_env.py
import os
from dotenv import load_dotenv

load_dotenv()

print("🔍 Checking ALL environment variables:")
print(f"GOOGLE_API_KEY: {'✅ Set' if os.getenv('GOOGLE_API_KEY') else '❌ Missing'}")
print(f"EBAY_APP_ID: {'✅ Set' if os.getenv('EBAY_APP_ID') else '❌ Missing'}")
print(f"EBAY_CERT_ID: {'✅ Set' if os.getenv('EBAY_CERT_ID') else '❌ Missing'}")
print(f"EBAY_DEV_ID: {'✅ Set' if os.getenv('EBAY_DEV_ID') else '❌ Missing'}")
print(f"EBAY_RU_NAME: {'✅ Set' if os.getenv('EBAY_RU_NAME') else '❌ Missing'}")
print(f"EBAY_AUTH_TOKEN: {'✅ Set' if os.getenv('EBAY_AUTH_TOKEN') else '❌ Missing'}")

print("\n📋 Values (truncated for security):")
for var in ['GOOGLE_API_KEY', 'EBAY_APP_ID', 'EBAY_CERT_ID', 'EBAY_DEV_ID', 'EBAY_RU_NAME', 'EBAY_AUTH_TOKEN']:
    value = os.getenv(var)
    if value:
        print(f"{var}: {value[:10]}...{value[-10:] if len(value) > 20 else ''}")
    else:
        print(f"{var}: ❌ Not set")