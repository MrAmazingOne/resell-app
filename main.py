# JOB QUEUE + POLLING SYSTEM - SOLD ITEMS ONLY
# Enhanced with eBay Taxonomy API for category suggestions and intelligent valuation

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from groq import Groq
from ebay_oauth import ebay_oauth
import uuid
import os
import json
from typing import Optional, Dict, Any
import logging
import base64
import requests
from datetime import datetime, timedelta, timezone
from ebay_integration import ebay_api
from dotenv import load_dotenv
import threading
import queue
import time
from concurrent.futures import ThreadPoolExecutor

# Image analysis imports (kept intact for Lift feature)
import cv2
import numpy as np
from PIL import Image
import io

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Resell Pro API - SOLD ITEMS ONLY",
    version="4.6.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Groq client setup
try:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set")
    groq_client = Groq(api_key=api_key)
    groq_model = "meta-llama/llama-4-scout-17b-16e-instruct"
    logger.info(f"Groq configured with model: {groq_model}")
except Exception as e:
    logger.error(f"Failed to configure Groq: {e}")
    groq_client = None
    groq_model = None

# Job queue & executor
job_queue = queue.Queue()
job_storage = {}
job_lock = threading.Lock()
job_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="JobWorker")

# Activity tracking
last_activity = time.time()
activity_lock = threading.Lock()

def update_activity():
    global last_activity
    with activity_lock:
        last_activity = time.time()

# eBay Token Storage
EBAY_AUTH_TOKEN = None
EBAY_TOKEN_LOCK = threading.Lock()

def get_ebay_token() -> Optional[str]:
    """Get eBay OAuth token from storage"""
    global EBAY_AUTH_TOKEN
    
    with EBAY_TOKEN_LOCK:
        if EBAY_AUTH_TOKEN:
            logger.debug(f"🔑 Using stored eBay token: {EBAY_AUTH_TOKEN[:20]}...")
            return EBAY_AUTH_TOKEN
        
        token = os.getenv('EBAY_AUTH_TOKEN')
        if token:
            logger.info(f"🔑 Loaded eBay token from env: {token[:20]}...")
            EBAY_AUTH_TOKEN = token
            return token
        
        logger.warning("⚠️ No eBay OAuth token available")
        return None

def store_ebay_token(token: str):
    """Store eBay OAuth token"""
    global EBAY_AUTH_TOKEN
    
    with EBAY_TOKEN_LOCK:
        EBAY_AUTH_TOKEN = token
        logger.info(f"🔑 Stored new eBay token: {token[:20]}...")

# ============= EBAY OAUTH ENDPOINTS =============

@app.get("/ebay/oauth/start")
async def ebay_oauth_start():
    """Generate eBay authorization URL for user consent"""
    update_activity()
    
    try:
        app_id = os.getenv('EBAY_APP_ID')
        
        if not app_id:
            raise HTTPException(status_code=500, detail="eBay credentials not configured")
        
        auth_url, state = ebay_oauth.generate_auth_url()
        
        logger.info(f"📗 Generated auth URL for long-term token (2 years)")
        logger.info(f"Generated auth URL: {auth_url}")
        
        return {
            "success": True,
            "auth_url": auth_url,
            "state": state,
            "redirect_uri": "https://resell-app-bi47.onrender.com/ebay/oauth/callback",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "token_duration": "permanent (2-year refresh token)"
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to generate auth URL: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ebay/oauth/callback")
async def ebay_oauth_callback(
    code: Optional[str] = None, 
    error: Optional[str] = None, 
    state: Optional[str] = None
):
    """Handle eBay OAuth callback"""
    update_activity()
    
    logger.info(f"📨 OAuth callback: code={code is not None}, error={error}")
    
    if error:
        redirect_url = f"ai-resell-pro://ebay-oauth-callback?error={error}"
        return HTMLResponse(content=f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Authorization Failed</title>
                <meta http-equiv="refresh" content="0; url={redirect_url}">
            </head>
            <body>
                <script>window.location.href = "{redirect_url}";</script>
                <h1>❌ Authorization Failed</h1>
                <p>Error: {error}</p>
            </body>
            </html>
        """)
    
    if not code:
        redirect_url = "ai-resell-pro://ebay-oauth-callback?error=no_code"
        return HTMLResponse(content=f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Authorization Failed</title>
                <meta http-equiv="refresh" content="0; url={redirect_url}">
            </head>
            <body>
                <script>window.location.href = "{redirect_url}";</script>
                <h1>❌ No Code</h1>
            </body>
            </html>
        """)
    
    try:
        logger.info(f"✅ Received code: {code[:20]}...")
        
        token_response = ebay_oauth.exchange_code_for_token(code, state=state)
        
        if token_response and token_response.get("success"):
            logger.info("✅ Token obtained with 2-year refresh token")
            
            token_id = token_response["token_id"]
            
            token_data = ebay_oauth.get_user_token(token_id)
            if token_data and "access_token" in token_data:
                access_token = token_data["access_token"]
                store_ebay_token(access_token)
                logger.info(f"✅ Token stored: {access_token[:20]}...")
            
            # Get token status to include expiry info
            token_status = ebay_oauth.get_token_status(token_id)
            
            redirect_url = f"ai-resell-pro://ebay-oauth-callback?success=true&token_id={token_id}&state={state}&permanent={token_status.get('is_permanent', False)}"
            
            return HTMLResponse(content=f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Success</title>
                    <meta http-equiv="refresh" content="0; url={redirect_url}">
                    <style>
                        body {{
                            font-family: -apple-system, sans-serif;
                            display: flex;
                            justify-content: center;
                            align-items: center;
                            min-height: 100vh;
                            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                            color: white;
                            text-align: center;
                        }}
                        .container {{
                            background: rgba(255,255,255,0.1);
                            padding: 40px;
                            border-radius: 20px;
                        }}
                    </style>
                </head>
                <body>
                    <div class="container">
                        <h1>✅ Connected!</h1>
                        <p>Long-term access granted (2 years)</p>
                        <p>Redirecting...</p>
                    </div>
                    <script>window.location.href = "{redirect_url}";</script>
                </body>
                </html>
            """)
        else:
            error_msg = token_response.get("error", "unknown") if token_response else "no_response"
            redirect_url = f"ai-resell-pro://ebay-oauth-callback?error=token_failed&details={error_msg}"
            
            return HTMLResponse(content=f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Failed</title>
                    <meta http-equiv="refresh" content="0; url={redirect_url}">
                </head>
                <body>
                    <script>window.location.href = "{redirect_url}";</script>
                    <h1>❌ Token Failed</h1>
                </body>
                </html>
            """)
        
    except Exception as e:
        logger.error(f"❌ Callback error: {e}")
        error_msg = str(e)[:100]
        redirect_url = f"ai-resell-pro://ebay-oauth-callback?error=callback_error&details={error_msg}"
        
        return HTMLResponse(content=f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Error</title>
                <meta http-equiv="refresh" content="0; url={redirect_url}">
            </head>
            <body>
                <script>window.location.href = "{redirect_url}";</script>
                <h1>❌ Error</h1>
            </body>
            </html>
        """)

@app.get("/ebay/oauth/token/{token_id}")
async def get_ebay_token_endpoint(token_id: str):
    """Get eBay access token"""
    update_activity()
    
    logger.info(f"🔑 Token request for: {token_id}")
    
    try:
        token_data = ebay_oauth.get_user_token(token_id)
        
        if not token_data:
            logger.error(f"❌ Token not found: {token_id}")
            logger.info(f"   Available tokens: {list(ebay_oauth.tokens.keys())}")
            raise HTTPException(status_code=404, detail="Token not found")
        
        logger.info(f"✅ Token found and valid")
        
        # Get token status for expiry info
        token_status = ebay_oauth.get_token_status(token_id)
        
        return {
            "success": True,
            "access_token": token_data["access_token"],
            "expires_at": token_data.get("expires_at", ""),
            "refresh_expires_at": token_data.get("refresh_expires_at", ""),
            "token_type": token_data.get("token_type", "Bearer"),
            "is_permanent": token_data.get("is_permanent", False),
            "token_status": token_status
        }
        
    except Exception as e:
        logger.error(f"❌ Get token error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ebay/oauth/status/{token_id}")
async def get_token_status(token_id: str):
    """Get status of a specific token including expiry times"""
    update_activity()
    
    logger.info(f"📋 Token status check for: {token_id}")
    
    try:
        token_status = ebay_oauth.get_token_status(token_id)
        
        if not token_status or token_status.get("valid") is None:
            logger.warning(f"⚠️ Token {token_id} not found in storage")
            # Check if we have any tokens at all
            available_tokens = list(ebay_oauth.tokens.keys())
            logger.info(f"   Available tokens: {available_tokens}")
            
            # Try to get from environment as fallback
            env_token = os.getenv('EBAY_AUTH_TOKEN')
            if env_token:
                logger.info("   Found token in environment")
                return {
                    "valid": True,
                    "expires_at": "Unknown (from env)",
                    "refresh_expires_at": "Unknown",
                    "seconds_remaining": 3600,
                    "refresh_seconds_remaining": 63072000,
                    "refreshable": False,
                    "message": "Token from environment",
                    "source": "environment",
                    "is_permanent": False
                }
            
            return {
                "valid": False,
                "error": f"Token {token_id} not found",
                "available_tokens": available_tokens,
                "is_permanent": False
            }
        
        return token_status
        
    except Exception as e:
        logger.error(f"❌ Token status error: {e}")
        return {
            "valid": False,
            "error": str(e),
            "message": "Error checking token status",
            "is_permanent": False
        }

@app.delete("/ebay/oauth/token/{token_id}")
async def revoke_ebay_token(token_id: str):
    """Revoke eBay token"""
    update_activity()
    
    try:
        success = ebay_oauth.revoke_token(token_id)
        
        if success:
            store_ebay_token(None)
            return {"success": True, "message": "Token revoked"}
        else:
            raise HTTPException(status_code=404, detail="Token not found")
        
    except Exception as e:
        logger.error(f"Revoke error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============= MAIN ENDPOINTS =============

@app.get("/health")
@app.get("/health/")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": app.version,
        "service": "resell-pro-api",
        "ebay_auth": "connected" if get_ebay_token() else "required"
    }

@app.get("/ping")
async def ping():
    update_activity()
    
    taxonomy_status = "Not tested"
    try:
        cat_id = ebay_api.get_category_suggestions("test item")
        taxonomy_status = "Ready" if cat_id else "Limited"
    except:
        taxonomy_status = "Failed"
    
    return {
        "status": "PONG",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ebay_ready": bool(get_ebay_token()),
        "taxonomy_api": taxonomy_status,
        "search_method": "soldItemsOnly + intelligent relaxation",
        "version": "4.6.0"
    }

@app.get("/")
async def root():
    update_activity()
    ebay_token = get_ebay_token()
    
    return {
        "message": "AI Resell Pro API - v4.6 (Optimized Valuation)",
        "status": "OPERATIONAL" if groq_client and ebay_token else "PARTIAL",
        "ebay_authentication": "✅ Connected" if ebay_token else "❌ Required",
        "features": [
            "Sold auctions only",
            "Intelligent filter relaxation",
            "20–50+ comp target with confidence tiers",
            "Core aspects only (Year/Make/Model)",
            "Dynamic Taxonomy suggestions",
            "Long-term OAuth tokens (2 years)"
        ],
        "valuation_endpoint": "/analyze_value/",
        "auth_endpoint": "/ebay/oauth/start",
        "docs": "/docs"
    }

@app.post("/analyze_value/")
async def analyze_item_value(item_data: Dict):
    """
    Main valuation endpoint - Optimized for reliable comps
    """
    update_activity()
    
    # Check authentication
    if not get_ebay_token():
        raise HTTPException(
            status_code=401,
            detail="eBay authentication required. Please connect your eBay account at /ebay/oauth/start"
        )
    
    keywords = item_data.get("keywords") or item_data.get("title", "")
    if not keywords:
        raise HTTPException(400, "keywords or title required")

    category_id = item_data.get("category_id") or ebay_api.get_category_suggestions(keywords)

    analysis = ebay_api.analyze_market_trends(
        keywords=keywords,
        category_id=category_id,
        item_data=item_data
    )

    message = (
        f"Based on {analysis.get('sample_size', 0)} comparable sold items. "
        f"Confidence: {analysis.get('confidence', 'unknown').upper()}. "
    )
    
    if analysis.get('sample_size', 0) <= 10:
        message += "Thin market – small sample expected for higher-value/rare items."
    elif analysis.get('sample_size', 0) >= 50:
        message += "Strong data – very reliable median price."

    return {
        "status": "success",
        "item": keywords,
        "valuation": analysis,
        "message": message,
        "recommended_strategy": (
            "Buy below recommended_buy_below if possible. "
            "List at/near median_price for fastest sale."
        )
    }

@app.get("/debug/valuation/{keywords:path}")
async def debug_valuation(keywords: str):
    """Debug endpoint to test valuation"""
    update_activity()
    
    # Check authentication
    if not get_ebay_token():
        return {
            "error": "eBay authentication required",
            "auth_url": "/ebay/oauth/start"
        }
    
    category = ebay_api.get_category_suggestions(keywords)
    return await analyze_item_value({
        "keywords": keywords,
        "category_id": category,
        "year": "1955",
        "make": "Chevrolet",
        "model": "3100"
    })

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    logger.info(f"Starting server on port {port}")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        workers=1,
        timeout_keep_alive=30,
        log_level="info"
    )