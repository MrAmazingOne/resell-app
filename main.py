# JOB QUEUE + POLLING SYSTEM - SOLD ITEMS ONLY
# Enhanced with eBay Taxonomy API for category suggestions and intelligent valuation

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from groq import Groq
from ebay_oauth import ebay_oauth
import uuid
import os
import json
from typing import Optional, Dict, Any
import logging
import base64
import requests
from datetime import datetime, timedelta
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

@app.get("/health")
@app.get("/health/")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": app.version,
        "service": "resell-pro-api"
    }

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
    # ... (keep your full mapping if desired, but dynamic preferred)
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
        "timestamp": datetime.now().isoformat(),
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

@app.post("/analyze_value/")
async def analyze_item_value(item_data: Dict):
    """
    Main valuation endpoint - Optimized for reliable comps
    """
    keywords = item_data.get("keywords") or item_data.get("title", "")
    if not keywords:
        raise HTTPException(400, "keywords or title required")

    # Try to get category dynamically first
    category_id = item_data.get("category_id") or ebay_api.get_category_suggestions(keywords)

    # Core structural aspects only (default for valuation)
    core_aspects = {}
    if "year" in item_data:
        core_aspects["Year"] = str(item_data["year"])
    if "make" in item_data:
        core_aspects["Make"] = item_data["make"]
    if "model" in item_data:
        core_aspects["Model"] = item_data["model"]

    # Allow optional override for stricter matching (rarely recommended for valuation)
    if item_data.get("strict_match", False):
        if "condition" in item_data:
            core_aspects["Condition"] = item_data["condition"]

    # Run the optimized analysis
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

# Debug endpoint to test valuation behavior
@app.get("/debug/valuation/{keywords:path}")
async def debug_valuation(keywords: str):
    category = ebay_api.get_category_suggestions(keywords)
    return await analyze_item_value({
        "keywords": keywords,
        "category_id": category,
        "year": "1955",           # example override
        "make": "Chevrolet",
        "model": "3100"
    })

# Keep your existing image upload/processing endpoints here
# (assuming you have one - the CV2/PIL imports are still available)

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