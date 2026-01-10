# JOB QUEUE + POLLING SYSTEM - SOLD ITEMS ONLY
# Enhanced with eBay Taxonomy API for category suggestions and intelligent valuation

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
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

# Keep existing OAuth and category mapping (as fallback only)
EBAY_AUTH_TOKEN = None
EBAY_TOKEN_LOCK = threading.Lock()

EBAY_CATEGORY_MAPPING = {
    'vehicles': {'id': '6000', 'subcategories': {'cars_trucks': '6001'}},
    'collectibles': {'id': '1', 'subcategories': {'pokemon': '183454'}},
}

@app.get("/health")
@app.get("/health/")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": app.version,
        "service": "resell-pro-api"
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
        "ebay_ready": True,  # Simplified for now
        "taxonomy_api": taxonomy_status,
        "search_method": "soldItemsOnly + intelligent relaxation",
        "version": "4.6.0"
    }

@app.get("/")
async def root():
    update_activity()
    return {
        "message": "AI Resell Pro API - v4.6 (Optimized Valuation)",
        "status": "OPERATIONAL" if groq_client else "PARTIAL",
        "features": [
            "Sold auctions only",
            "Intelligent filter relaxation",
            "20–50+ comp target with confidence tiers",
            "Core aspects only (Year/Make/Model)",
            "Dynamic Taxonomy suggestions",
            "Full eBay OAuth flow"
        ],
        "valuation_endpoint": "/analyze_value/",
        "docs": "/docs"
    }

# === eBay OAuth Routes (fully restored for iOS app) ===

@app.get("/ebay/oauth/start")
async def ebay_oauth_start():
    """Generate eBay authorization URL for user consent"""
    try:
        auth_url, state = ebay_oauth.generate_auth_url()
        logger.info(f"Generated auth URL: {auth_url}")
        return {
            "auth_url": auth_url,
            "state": state,
            "success": True
        }
    except Exception as e:
        logger.error(f"Error generating auth URL: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate eBay auth URL")

@app.get("/ebay/oauth/callback")
async def ebay_oauth_callback(code: str, state: str = None, request: Request = None):
    """Handle eBay OAuth callback after user consent"""
    logger.info(f"Callback hit - code: {code[:10]}..., state: {state}, query params: {dict(request.query_params)}")
    try:
        token_data = ebay_oauth.exchange_code_for_token(authorization_code=code, state=state)
        if not token_data or not token_data.get("success"):
            logger.warning("Token exchange failed")
            raise HTTPException(status_code=400, detail="Token exchange failed")

        # Generate a persistent token ID for the iOS app
        token_id = str(uuid.uuid4())
        
        # Store in ebay_oauth.tokens (adjust if your class stores differently)
        ebay_oauth.tokens[token_id] = token_data
        
        logger.info(f"Token exchange success - new token_id: {token_id}")
        
        # Return JSON the iOS app can parse
        return {
            "success": True,
            "message": "eBay account connected successfully",
            "token_id": token_id,
            "token_data": {  # Minimal safe subset
                "expires_in": token_data.get("expires_in"),
                "refresh_token_expires_in": token_data.get("refresh_token_expires_in"),
                "is_permanent": token_data.get("is_permanent", False)
            }
        }
    except Exception as e:
        logger.error(f"Callback error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Callback processing failed: {str(e)}")

@app.get("/ebay/oauth/status/{token_id}")
async def ebay_oauth_status(token_id: str):
    """Check status of stored OAuth token (used by iOS app)"""
    try:
        status = ebay_oauth.get_token_status(token_id)
        logger.info(f"Status check for token {token_id}: {status}")
        return status
    except Exception as e:
        logger.error(f"Status check error for {token_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to check token status")

@app.get("/ebay/oauth/token/{token_id}")
async def ebay_get_token(token_id: str):
    """Endpoint for iOS to fetch fresh token data (called by getAccessToken)"""
    try:
        # Retrieve from storage
        token_data = ebay_oauth.tokens.get(token_id)
        if not token_data:
            raise HTTPException(status_code=404, detail="Token not found")
        
        # Return only what's needed for iOS
        return {
            "success": True,
            "access_token": token_data.get("access_token"),
            "expires_at": token_data.get("access_expires_at"),
            "refresh_token": token_data.get("refresh_token"),
            "refresh_token_expires_at": token_data.get("refresh_expires_at"),
            "is_permanent": token_data.get("is_permanent", False)
        }
    except Exception as e:
        logger.error(f"Get token error for {token_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve token")

@app.delete("/ebay/oauth/token/{token_id}")
async def ebay_revoke_token(token_id: str):
    """Revoke and delete stored token"""
    try:
        if ebay_oauth.revoke_token(token_id):
            logger.info(f"Token {token_id} revoked successfully")
            return {"success": True, "message": "Token revoked"}
        else:
            raise HTTPException(status_code=404, detail="Token not found")
    except Exception as e:
        logger.error(f"Revoke error for {token_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to revoke token")

# === eBay Market Analysis Routes (matching iOS eBayAPIClient.swift) ===

@app.post("/api/ebay/analyze")
async def ebay_analyze_market(body: Dict):
    """Analyze market value (called from iOS eBayAPIClient)"""
    query = body.get("query")
    if not query:
        raise HTTPException(status_code=400, detail="Query required")
    
    analysis = ebay_api.analyze_market_trends(keywords=query)
    return analysis

@app.post("/api/ebay/search/sold")
async def ebay_search_sold(body: Dict):
    """Search sold items (called from iOS eBayAPIClient)"""
    query = body.get("query")
    limit = body.get("limit", 10)
    if not query:
        raise HTTPException(status_code=400, detail="Query required")
    
    items, meta = ebay_api.search_sold_items(keywords=query, max_items=limit)
    return {"items": items, "meta": meta}

# === Core Valuation Endpoint (keep for web/debug) ===
@app.post("/analyze_value/")
async def analyze_item_value(item_data: Dict):
    keywords = item_data.get("keywords") or item_data.get("title", "")
    if not keywords:
        raise HTTPException(400, "keywords or title required")

    category_id = item_data.get("category_id") or ebay_api.get_category_suggestions(keywords)

    core_aspects = {}
    if "year" in item_data:
        core_aspects["Year"] = str(item_data["year"])
    if "make" in item_data:
        core_aspects["Make"] = item_data["make"]
    if "model" in item_data:
        core_aspects["Model"] = item_data["model"]

    if item_data.get("strict_match", False):
        if "condition" in item_data:
            core_aspects["Condition"] = item_data["condition"]

    analysis = ebay_api.analyze_market_trends(
        keywords=keywords,
        category_id=category_id,
        core_aspects=core_aspects
    )

    message = (
        f"Based on {analysis.get('sample_size', 0)} comparable sold items. "
        f"Confidence: {analysis.get('confidence', 'unknown').upper()}. "
    )
    
    if analysis.get('sample_size', 0) <= 10:
        message += "Thin market — small sample expected for higher-value/rare items."
    elif analysis.get('sample_size', 0) >= 50:
        message += "Strong data — very reliable median price."

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