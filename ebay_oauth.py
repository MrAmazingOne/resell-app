[file name]: ebay_oauth.py
[file content begin]
"""
eBay OAuth Authentication for iOS App
"""
import os
import requests
import json
import base64
import logging
from datetime import datetime, timedelta
import uuid
from typing import Dict, Optional, Tuple
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

class eBayOAuth:
    def __init__(self):
        self.app_id = os.getenv('EBAY_APP_ID')
        self.cert_id = os.getenv('EBAY_CERT_ID')
        self.ru_name = os.getenv('EBAY_RU_NAME')
        
        # Production endpoints
        self.auth_url = "https://auth.ebay.com/oauth2/authorize"
        self.token_url = "https://api.ebay.com/identity/v1/oauth2/token"
        
        # For development/testing
        self.use_sandbox = os.getenv('EBAY_USE_SANDBOX', 'false').lower() == 'true'
        if self.use_sandbox:
            self.auth_url = "https://auth.sandbox.ebay.com/oauth2/authorize"
            self.token_url = "https://api.sandbox.ebay.com/identity/v1/oauth2/token"
            logger.info("Using eBay SANDBOX for OAuth")
        
        # In-memory token storage (in production, use database)
        self.tokens = {}
    
    def generate_auth_url(self, state: str = None) -> Tuple[str, str]:
        """
        Generate eBay OAuth authorization URL for iOS app
        Returns: (auth_url, state)
        """
        if not state:
            state = str(uuid.uuid4())
        
        # Scopes needed for our app
        scopes = [
            "https://api.ebay.com/oauth/api_scope",  # View public data
            "https://api.ebay.com/oauth/api_scope/sell.inventory",  # List items
            "https://api.ebay.com/oauth/api_scope/sell.account",  # View account
        ]
        
        # Use iOS app URL scheme for redirect
        params = {
            "client_id": self.app_id,
            "response_type": "code",
            "redirect_uri": "ai-resell-pro://ebay-auth",  # iOS app URL scheme
            "scope": " ".join(scopes),
            "state": state,
            "prompt": "login"
        }
        
        auth_url = f"{self.auth_url}?{requests.compat.urlencode(params)}"
        logger.info(f"Generated auth URL with state: {state}")
        
        return auth_url, state
    
    def exchange_code_for_token(self, authorization_code: str, state: str = None) -> Optional[Dict]:
        """
        Exchange authorization code for access token
        """
        try:
            headers = {
                "Content-Type": "application/x-www-form-urlencoded",
                "Authorization": f"Basic {base64.b64encode(f'{self.app_id}:{self.cert_id}'.encode()).decode()}"
            }
            
            # Use the SAME redirect_uri that was in the auth request
            data = {
                "grant_type": "authorization_code",
                "code": authorization_code,
                "redirect_uri": "ai-resell-pro://ebay-auth"
            }
            
            logger.info(f"Exchanging code for token: {authorization_code[:20]}...")
            
            response = requests.post(self.token_url, headers=headers, data=data, timeout=30)
            
            logger.info(f"Token exchange status: {response.status_code}")
            
            if response.status_code == 200:
                token_data = response.json()
                logger.info(f"✅ Token exchange successful. Expires in: {token_data.get('expires_in')}s")
                
                # Store token (in production, associate with user ID)
                user_token_id = str(uuid.uuid4())
                self.tokens[user_token_id] = {
                    **token_data,
                    "state": state,
                    "created_at": datetime.now().isoformat(),
                    "expires_at": (datetime.now() + timedelta(seconds=token_data.get('expires_in', 7200))).isoformat()
                }
                
                return {
                    "success": True,
                    "token_id": user_token_id,
                    "access_token": token_data["access_token"],
                    "refresh_token": token_data.get("refresh_token"),
                    "expires_in": token_data.get("expires_in"),
                    "token_type": token_data.get("token_type")
                }
            else:
                logger.error(f"Token exchange failed: {response.status_code} - {response.text}")
                return {
                    "success": False,
                    "error": f"Token exchange failed: {response.status_code}",
                    "details": response.text[:200]
                }
                
        except Exception as e:
            logger.error(f"Token exchange error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def refresh_token(self, refresh_token: str) -> Optional[Dict]:
        """
        Refresh an expired access token
        """
        try:
            headers = {
                "Content-Type": "application/x-www-form-urlencoded",
                "Authorization": f"Basic {base64.b64encode(f'{self.app_id}:{self.cert_id}'.encode()).decode()}"
            }
            
            data = {
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "scope": "https://api.ebay.com/oauth/api_scope https://api.ebay.com/oauth/api_scope/sell.inventory"
            }
            
            response = requests.post(self.token_url, headers=headers, data=data, timeout=30)
            
            if response.status_code == 200:
                token_data = response.json()
                logger.info("✅ Token refresh successful")
                
                return {
                    "success": True,
                    "access_token": token_data["access_token"],
                    "expires_in": token_data.get("expires_in"),
                    "refresh_token": token_data.get("refresh_token", refresh_token)  # Keep old if not provided
                }
            else:
                logger.error(f"Token refresh failed: {response.status_code}")
                return {
                    "success": False,
                    "error": f"Refresh failed: {response.status_code}"
                }
                
        except Exception as e:
            logger.error(f"Token refresh error: {e}")
            return None
    
    def get_user_token(self, token_id: str) -> Optional[Dict]:
        """
        Get stored token for a user
        """
        token_data = self.tokens.get(token_id)
        
        if not token_data:
            return None
        
        # Check if token is expired
        expires_at = datetime.fromisoformat(token_data["expires_at"])
        if datetime.now() > expires_at:
            logger.info(f"Token {token_id} expired, attempting refresh...")
            if "refresh_token" in token_data:
                refreshed = self.refresh_token(token_data["refresh_token"])
                if refreshed and refreshed["success"]:
                    # Update stored token
                    self.tokens[token_id].update({
                        "access_token": refreshed["access_token"],
                        "expires_at": (datetime.now() + timedelta(seconds=refreshed.get("expires_in", 7200))).isoformat(),
                        "refresh_token": refreshed.get("refresh_token", token_data["refresh_token"])
                    })
                    return self.tokens[token_id]
                else:
                    logger.error(f"Failed to refresh token {token_id}")
                    return None
        
        return token_data
    
    def revoke_token(self, token_id: str) -> bool:
        """
        Revoke/delete a token
        """
        if token_id in self.tokens:
            del self.tokens[token_id]
            logger.info(f"Token {token_id} revoked")
            return True
        return False

# Global instance
ebay_oauth = eBayOAuth()
[file content end]