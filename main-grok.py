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
        "ebay_ready": bool(ebay_oauth.get_user_token("some_token_id")),  # adjust as needed
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
            "Dynamic Taxonomy suggestions"
        ],
        "valuation_endpoint": "/analyze_value/",
        "docs": "/docs"
    }

# eBay OAuth Routes
@app.get("/ebay/oauth/start")
async def ebay_oauth_start():
    """Generate eBay authorization URL for user consent"""
    try:
        auth_url, state = ebay_oauth.generate_auth_url()
        logger.info(f"Generated auth URL: {auth_url}")
        return {
            "auth_url": auth_url,
            "state": state
        }
    except Exception as e:
        logger.error(f"Error generating auth URL: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate eBay auth URL")

@app.get("/ebay/oauth/callback")
async def ebay_oauth_callback(code: str, state: str = None, request: Request = None):
    """Handle eBay OAuth callback after user consent"""
    logger.info(f"Callback received - code: {code[:10]}..., state: {state}, full query: {request.query_params}")
    try:
        token_data = ebay_oauth.exchange_code_for_token(authorization_code=code, state=state)
        if token_data and token_data.get("success"):
            token_id = str(uuid.uuid4())
            # Store token (your ebay_oauth.tokens should handle this)
            # ebay_oauth.tokens[token_id] = token_data  # Uncomment/adjust as per your class
            logger.info(f"Token exchange success - token_id: {token_id}")
            # Return success - iOS app can poll status with this token_id
            return {
                "success": True,
                "message": "eBay account connected",
                "token_id": token_id
            }
        else:
            logger.warning("Token exchange failed")
            raise HTTPException(status_code=400, detail="Token exchange failed")
    except Exception as e:
        logger.error(f"Callback error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/ebay/oauth/status/{token_id}")
async def ebay_oauth_status(token_id: str):
    """Check status of stored OAuth token"""
    try:
        status = ebay_oauth.get_token_status(token_id)
        logger.info(f"Token status for {token_id}: {status}")
        return status
    except Exception as e:
        logger.error(f"Status check error for {token_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to check token status")

@app.post("/analyze_value/")
async def analyze_item_value(item_data: Dict):
    """
    Main valuation endpoint - Optimized for reliable comps
    """
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