import os
import webbrowser
from ebaysdk.exception import ConnectionError
from ebaysdk.trading import Connection as TradingConnection
from dotenv import load_dotenv

load_dotenv()

def get_authorization_url():
    """Generate eBay Legacy Auth URL (Auth'n'Auth)"""
    try:
        api = TradingConnection(
            config_file=None,  # ← CRITICAL: Added config_file=None
            appid=os.getenv('EBAY_APP_ID'),
            devid=os.getenv('EBAY_DEV_ID'),
            certid=os.getenv('EBAY_CERT_ID'),
            domain='api.sandbox.ebay.com',  # Sandbox environment
            warnings=True
        )
        
        # Get session ID for authentication
        response = api.execute('GetSessionID', {
            'RuName': os.getenv('EBAY_RU_NAME')
        })
        
        session_id = response.reply.SessionID
        auth_url = f"https://signin.sandbox.ebay.com/ws/eBayISAPI.dll?SignIn&runame={os.getenv('EBAY_RU_NAME')}&SessID={session_id}"
        
        return auth_url
        
    except ConnectionError as e:
        print(f"Authentication error: {e}")
        return None

async def exchange_session_for_token(session_id: str):
    """Exchange session ID for user token"""
    try:
        api = TradingConnection(
            config_file=None,  # ← CRITICAL: Added config_file=None
            appid=os.getenv('EBAY_APP_ID'),
            devid=os.getenv('EBAY_DEV_ID'),
            certid=os.getenv('EBAY_CERT_ID'),
            domain='api.sandbox.ebay.com',
            warnings=True
        )
        
        # Fetch the token using session ID
        response = api.execute('FetchToken', {
            'SessionID': session_id
        })
        
        ebay_token = response.reply.eBayAuthToken
        return ebay_token
        
    except ConnectionError as e:
        print(f"Token exchange error: {e}")
        return None

def get_ebay_oauth_token():
    """Get stored OAuth token from environment"""
    return os.environ.get('EBAY_AUTH_TOKEN')

if __name__ == "__main__":
    # Test the authorization URL generation
    auth_url = get_authorization_url()
    if auth_url:
        print("eBay Legacy Auth URL:")
        print(auth_url)
        print("\nOpen this URL in browser to authenticate")
    else:
        print("Failed to generate auth URL")