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
        Generate eBay OAuth authorization URL
        ✅ ALWAYS USES WEB REDIRECT (eBay doesn't support custom URL schemes)
        """
        if not state:
            state = str(uuid.uuid4())
        
        # Scopes needed for our app - USING LONG-TERM SCOPE FOR 2-YEAR TOKENS
        scopes = [
            "https://api.ebay.com/oauth/api_scope",
            "https://api.ebay.com/oauth/api_scope/sell.inventory",
            "https://api.ebay.com/oauth/api_scope/sell.account",
            "https://api.ebay.com/oauth/api_scope/sell.marketing",
        ]
        
        # ✅ ALWAYS USE WEB REDIRECT (eBay requirement)
        redirect_uri = "https://resell-app-bi47.onrender.com/ebay/oauth/callback"
        
        params = {
            "client_id": self.app_id,
            "response_type": "code",
            "redirect_uri": redirect_uri,
            "scope": " ".join(scopes),
            "state": state,
            "prompt": "login",
            # ✅ CRITICAL: Request refresh token with maximum lifespan (2 years)
            "duration": "permanent"  # This is KEY for long-lived tokens
        }
        
        auth_url = f"{self.auth_url}?{requests.compat.urlencode(params)}"
        logger.info(f"Generated auth URL with scopes: {[s.split('/')[-1] for s in scopes]}")
        
        return auth_url, state
    
    def exchange_code_for_token(self, authorization_code: str, state: str = None) -> Optional[Dict]:
        """
        Exchange authorization code for access token
        ✅ USES WEB REDIRECT URI (matches auth request)
        ✅ REQUESTS LONG-LIVED REFRESH TOKEN (2 years)
        """
        try:
            headers = {
                "Content-Type": "application/x-www-form-urlencoded",
                "Authorization": f"Basic {base64.b64encode(f'{self.app_id}:{self.cert_id}'.encode()).decode()}"
            }
            
            # ✅ USE WEB REDIRECT URI (must match auth request)
            # ✅ CRITICAL: Include 'duration=permanent' for long-lived tokens
            data = {
                "grant_type": "authorization_code",
                "code": authorization_code,
                "redirect_uri": "https://resell-app-bi47.onrender.com/ebay/oauth/callback",
                "duration": "permanent"  # This gets 2-year refresh tokens
            }
            
            logger.info(f"Exchanging code for token: {authorization_code[:20]}... (duration: permanent)")
            
            response = requests.post(self.token_url, headers=headers, data=data, timeout=30)
            
            logger.info(f"Token exchange status: {response.status_code}")
            
            if response.status_code == 200:
                token_data = response.json()
                
                # ✅ eBay provides specific expiration times:
                # - access_token: 7200 seconds (2 hours)
                # - refresh_token: 63072000 seconds (2 years) when 'duration=permanent'
                access_token_expiry = token_data.get('expires_in', 7200)
                refresh_token_expiry = token_data.get('refresh_token_expires_in', 63072000)
                
                logger.info(f"✅ Token exchange successful. Access expires in: {access_token_expiry}s")
                logger.info(f"✅ Refresh token expires in: {refresh_token_expiry}s (~2 years)")
                
                # Store token (in production, associate with user ID)
                user_token_id = str(uuid.uuid4())
                self.tokens[user_token_id] = {
                    **token_data,
                    "state": state,
                    "created_at": datetime.now().isoformat(),
                    "access_expires_at": (datetime.now() + timedelta(seconds=access_token_expiry)).isoformat(),
                    "refresh_expires_at": (datetime.now() + timedelta(seconds=refresh_token_expiry)).isoformat(),
                    "is_permanent": True
                }
                
                return {
                    "success": True,
                    "token_id": user_token_id,
                    "access_token": token_data["access_token"],
                    "refresh_token": token_data.get("refresh_token"),
                    "expires_in": access_token_expiry,
                    "refresh_token_expires_in": refresh_token_expiry,
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
        ✅ PRESERVES LONG-LIVED REFRESH TOKEN
        """
        try:
            headers = {
                "Content-Type": "application/x-www-form-urlencoded",
                "Authorization": f"Basic {base64.b64encode(f'{self.app_id}:{self.cert_id}'.encode()).decode()}"
            }
            
            # ✅ IMPORTANT: Include 'duration=permanent' to maintain 2-year refresh token
            data = {
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "duration": "permanent",  # Keep the refresh token long-lived
                "scope": "https://api.ebay.com/oauth/api_scope https://api.ebay.com/oauth/api_scope/sell.inventory"
            }
            
            response = requests.post(self.token_url, headers=headers, data=data, timeout=30)
            
            if response.status_code == 200:
                token_data = response.json()
                logger.info("✅ Token refresh successful")
                
                # ✅ Check if we got a new refresh token (eBay may return same one)
                new_refresh_token = token_data.get("refresh_token", refresh_token)
                refresh_expires_in = token_data.get("refresh_token_expires_in", 63072000)
                
                return {
                    "success": True,
                    "access_token": token_data["access_token"],
                    "expires_in": token_data.get("expires_in", 7200),
                    "refresh_token": new_refresh_token,
                    "refresh_token_expires_in": refresh_expires_in
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
        
        # Check if access token is expired (NOT refresh token)
        expires_at_str = token_data.get("access_expires_at") or token_data.get("expires_at")
        if expires_at_str:
            try:
                expires_at = datetime.fromisoformat(expires_at_str)
                if datetime.now() > expires_at:
                    logger.info(f"Access token {token_id} expired, attempting refresh...")
                    if "refresh_token" in token_data:
                        refreshed = self.refresh_token(token_data["refresh_token"])
                        if refreshed and refreshed["success"]:
                            # Update stored token with new access token
                            # Preserve existing refresh token expiry if not updated
                            refresh_expires_at = datetime.fromisoformat(token_data["refresh_expires_at"]) if "refresh_expires_at" in token_data else datetime.now() + timedelta(seconds=63072000)
                            
                            self.tokens[token_id].update({
                                "access_token": refreshed["access_token"],
                                "access_expires_at": (datetime.now() + timedelta(seconds=refreshed.get("expires_in", 7200))).isoformat(),
                                "refresh_token": refreshed.get("refresh_token", token_data["refresh_token"]),
                                "refresh_expires_at": refresh_expires_at.isoformat()
                            })
                            return self.tokens[token_id]
                        else:
                            logger.error(f"Failed to refresh token {token_id}")
                            return None
            except Exception as e:
                logger.error(f"Error checking token expiry: {e}")
                # Continue with existing token
        
        return token_data
    
    def get_token_status(self, token_id: str) -> Dict:
        """
        Get detailed token status including expiry times
        """
        token_data = self.tokens.get(token_id)
        
        if not token_data:
            return {"valid": False, "error": "Token not found"}
        
        try:
            # Check access token expiry
            access_expires_str = token_data.get("access_expires_at") or token_data.get("expires_at")
            refresh_expires_str = token_data.get("refresh_expires_at")
            
            access_valid = True
            refresh_valid = True
            access_seconds = 0
            refresh_seconds = 0
            
            if access_expires_str:
                access_expires = datetime.fromisoformat(access_expires_str)
                access_seconds = (access_expires - datetime.now()).total_seconds()
                access_valid = access_seconds > 300  # 5 minutes buffer
            
            if refresh_expires_str:
                refresh_expires = datetime.fromisoformat(refresh_expires_str)
                refresh_seconds = (refresh_expires - datetime.now()).total_seconds()
                refresh_valid = refresh_seconds > 0
            
            return {
                "valid": access_valid and refresh_valid,
                "access_expires_at": access_expires_str,
                "refresh_expires_at": refresh_expires_str,
                "access_seconds_remaining": int(access_seconds),
                "refresh_seconds_remaining": int(refresh_seconds),
                "refreshable": "refresh_token" in token_data and refresh_valid,
                "message": "Token valid" if access_valid and refresh_valid else "Token expired",
                "is_permanent": token_data.get("is_permanent", False)
            }
            
        except Exception as e:
            logger.error(f"Error getting token status: {e}")
            return {"valid": False, "error": str(e)}
    
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