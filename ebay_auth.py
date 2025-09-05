import os
import aiohttp
import base64
from urllib.parse import urlencode
from dotenv import load_dotenv

load_dotenv()

# eBay API configuration
EBAY_APP_ID = os.getenv("EBAY_APP_ID", "JustinHa-ReReSell-SBX-e11823bc1-b28e2f71")
EBAY_CERT_ID = os.getenv("EBAY_CERT_ID", "SBX-l1823bcla4a9-aa79-4e52-8a77-735e")
EBAY_DEV_ID = os.getenv("EBAY_DEV_ID", "a3376eb0-d27c-46a6-9a04-e6fdfed5e5fc")
EBAY_RU_NAME = os.getenv("EBAY_RU_NAME", "Justin_Harris-JustinHa-ReReSe-palkxra")

# Use your Render URL instead of localhost
REDIRECT_URI = "https://resell-app-bi47.onrender.com/auth/callback"

# Sandbox vs Production
EBAY_ENVIRONMENT = "sandbox"  # Change to "production" for live
EBAY_BASE_URL = f"https://api.{EBAY_ENVIRONMENT}.ebay.com"

def get_authorization_url():
    """Generate eBay OAuth authorization URL"""
    base_auth_url = f"https://auth.{EBAY_ENVIRONMENT}.ebay.com/oauth2/authorize"
    scopes = [
        "https://api.ebay.com/oauth/api_scope",
        "https://api.ebay.com/oauth/api_scope/sell.inventory",
        "https://api.ebay.com/oauth/api_scope/sell.account",
        "https://api.ebay.com/oauth/api_scope/sell.fulfillment"
    ]
    
    params = {
        "client_id": EBAY_APP_ID,
        "redirect_uri": REDIRECT_URI,
        "response_type": "code",
        "scope": " ".join(scopes),
        "state": "random_state_string"
    }
    
    return f"{base_auth_url}?{urlencode(params)}"

async def exchange_code_for_token(authorization_code):
    """Exchange authorization code for access token"""
    token_url = f"https://api.{EBAY_ENVIRONMENT}.ebay.com/identity/v1/oauth2/token"
    
    # Prepare the authorization header
    credentials = f"{EBAY_APP_ID}:{EBAY_CERT_ID}"
    encoded_credentials = base64.b64encode(credentials.encode()).decode()
    
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Authorization": f"Basic {encoded_credentials}"
    }
    
    data = {
        "grant_type": "authorization_code",
        "code": authorization_code,
        "redirect_uri": REDIRECT_URI
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(token_url, headers=headers, data=data) as response:
            if response.status == 200:
                token_data = await response.json()
                return token_data
            else:
                error_text = await response.text()
                raise Exception(f"Token exchange failed: {response.status} - {error_text}")

def get_ebay_oauth_token():
    """Get stored OAuth token from environment"""
    return os.environ.get('EBAY_AUTH_TOKEN')

if __name__ == "__main__":
    # Test the authorization URL generation
    auth_url = get_authorization_url()
    print("eBay Authorization URL:")
    print(auth_url)