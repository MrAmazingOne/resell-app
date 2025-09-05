from ebaysdk.exception import ConnectionError
from ebaysdk.trading import Connection as TradingConnection
import webbrowser
import json
import os

def get_user_token():
    """Get eBay user token for making API calls"""
    try:
        # This will open a browser for user authentication
        api = TradingConnection(
            config_file=None,
            appid=os.getenv('EBAY_APP_ID', 'JustinHa-ReReSell-SBX-e11823bc1-b28e2f71'),
            devid=os.getenv('EBAY_DEV_ID', 'a3376eb0-d27c-46a6-9a04-e6fdfed5e5fc'),
            certid=os.getenv('EBAY_CERT_ID', 'SBX-l1823bcla4a9-aa79-4e52-8a77-735e'),
            warnings=True
        )
        
        # Get session ID for authentication
        response = api.execute('GetSessionID', {
            'RuName': 'Justin_Harris-JustinHa-ReReSe-palkxra'  # Use your actual RuName
        })
        
        session_id = response.reply.SessionID
        auth_url = f"https://signin.sandbox.ebay.com/ws/eBayISAPI.dll?SignIn&RuName=Justin_Harris-JustinHa-ReReSe-palkxra&SessID={session_id}"
        
        print(f"Please visit this URL to authenticate: {auth_url}")
        webbrowser.open(auth_url)
        
        input("After authenticating, press Enter to continue...")
        
        # Fetch the token
        response = api.execute('FetchToken', {
            'SessionID': session_id
        })
        
        ebay_token = response.reply.eBayAuthToken
        print(f"Your eBay token: {ebay_token}")
        
        # Save token to environment variable
        os.environ['EBAY_AUTH_TOKEN'] = ebay_token
        
        return ebay_token
        
    except ConnectionError as e:
        print(f"Authentication error: {e}")
        return None

if __name__ == "__main__":
    token = get_user_token()
    if token:
        print("Authentication successful! Token saved to environment.")