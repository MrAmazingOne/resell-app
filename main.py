# JOB QUEUE + POLLING SYSTEM
# This API is designed for Render's 30s timeout limit.
# /upload_item/ queues jobs and returns a job_id immediately.
# Clients must poll /job/{job_id}/status for results.
# The background worker processes jobs asynchronously and updates job status/results.
# Thread safety is ensured with job_lock. See endpoint docstrings for details.

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
import google.generativeai as genai
import os
import json
from typing import Optional, List, Dict, Any
import tempfile
from PIL import Image
import logging
import base64
import requests
import re
from datetime import datetime, timedelta
import asyncio
import aiohttp
from enum import Enum
from ebay_integration import ebay_api
from ebay_auth import get_authorization_url, exchange_session_for_token
from dotenv import load_dotenv
import uuid
import threading
import queue
import time
from concurrent.futures import ThreadPoolExecutor

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Resell Pro API", version="3.1.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Google Generative AI
try:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set")
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
    logger.info("Google Generative AI configured successfully")
except Exception as e:
    logger.error(f"Failed to configure Google Generative AI: {e}")
    model = None

# Category Enum for specialized analysis
class ItemCategory(Enum):
    ELECTRONICS = "electronics"
    CLOTHING = "clothing"
    FURNITURE = "furniture"
    COLLECTIBLES = "collectibles"
    BOOKS = "books"
    TOYS = "toys"
    UNKNOWN = "unknown"

# In-memory job queue (persists for the lifetime of the process)
job_queue = queue.Queue()
job_storage = {}  # Stores job status and results
job_lock = threading.Lock()

# Thread pool for background processing
executor = ThreadPoolExecutor(max_workers=1)  # Single worker thread

def background_worker():
    """Background worker that processes jobs from the queue"""
    while True:
        try:
            job_id = job_queue.get(timeout=30)  # Wait for jobs with timeout
            process_job(job_id)
            job_queue.task_done()
        except queue.Empty:
            # No jobs, continue waiting
            continue
        except Exception as e:
            print(f"Background worker error: {e}")
            time.sleep(5)  # Wait before trying again

def process_job(job_id):
    """Process a single job"""
    try:
        with job_lock:
            job_data = job_storage.get(job_id)
            if not job_data:
                return
            
            # Update status to processing
            job_data['status'] = 'processing'
            job_data['started_at'] = datetime.now().isoformat()
            job_storage[job_id] = job_data
        
        # Your existing processing logic
        result = process_image_standard(job_data)
        
        with job_lock:
            if result['status'] == 'completed':
                job_data['status'] = 'completed'
                job_data['completed_at'] = datetime.now().isoformat()
                job_data['result'] = result['result']
            else:
                job_data['status'] = 'failed'
                job_data['error'] = result.get('error', 'Unknown error')
                job_data['completed_at'] = datetime.now().isoformat()
            
            job_storage[job_id] = job_data
            
    except Exception as e:
        with job_lock:
            job_data = job_storage.get(job_id, {})
            job_data['status'] = 'failed'
            job_data['error'] = str(e)
            job_data['completed_at'] = datetime.now().isoformat()
            job_storage[job_id] = job_data
        print(f"Job {job_id} failed: {e}")

# Start background worker when app starts
@app.on_event("startup")
async def startup_event():
    # Start background worker in a separate thread
    threading.Thread(target=background_worker, daemon=True).start()
    print("Background worker started")

# ENHANCED MARKET ANALYSIS PROMPT
market_analysis_prompt = """
EXPERT RESELL ANALYST - MAXIMUM ACCURACY ANALYSIS:

You are analyzing items for resale profitability. Focus on:

🔍 **IDENTIFICATION PHASE:**
- Extract EVERY visible text, number, logo, brand mark, model number
- Identify materials, construction quality, age indicators
- Note condition issues, wear patterns, damage
- Capture size, dimensions, serial numbers

📊 **MARKET ANALYSIS PHASE:**
- Estimate current market value range based on condition
- Consider brand popularity, rarity, demand trends
- Factor in condition deductions and market saturation
- Account for seasonal pricing variations

💰 **PROFITABILITY ANALYSIS:**
- Calculate realistic resale price range
- Suggest maximum purchase price for profit
- Estimate profit margins after fees (eBay: 13%, shipping: $8-15)
- Rate resellability 1-10 based on demand/competition

Return analysis in JSON format:

{
  "title": "eBay-optimized title with brand, model, key features",
  "description": "Detailed description with ALL visible details and condition notes",
  "price_range": "Current market range: $X - $Y",
  "resellability_rating": 8,
  "suggested_cost": "Maximum to pay: $X (for profitable resale)",
  "market_insights": "Market demand, competition level, selling tips",
  "authenticity_checks": "Red flags to verify before purchase",
  "profit_potential": "Expected profit: $X-Y after fees",
  "category": "Primary eBay category",
  "ebay_specific_tips": ["Photography tips", "Listing optimization", "Timing advice"],
  
  // Extended details
  "brand": "Exact brand if visible",
  "model": "Model number/name if visible", 
  "year": "Production year if determinable",
  "condition": "Detailed condition assessment",
  "confidence": 0.85,
  "analysis_depth": "comprehensive",
  "key_features": ["Notable features that add value"],
  "comparable_items": "Similar items selling for $X-Y"
}

CRITICAL: Base pricing on actual market conditions, not retail prices.
Account for condition issues that reduce value.
"""

def detect_category(title: str, description: str) -> ItemCategory:
    """Detect the most likely category for an item"""
    title_lower = title.lower()
    description_lower = description.lower()
    
    category_keywords = {
        ItemCategory.ELECTRONICS: ["electronic", "computer", "phone", "tablet", "camera", "laptop", "charger", "battery", "screen"],
        ItemCategory.CLOTHING: ["shirt", "pants", "dress", "jacket", "shoe", "sweater", "fabric", "cotton", "wool", "brand"],
        ItemCategory.FURNITURE: ["chair", "table", "desk", "cabinet", "sofa", "couch", "wood", "furniture", "drawer"],
        ItemCategory.COLLECTIBLES: ["collectible", "rare", "vintage", "antique", "edition", "limited", "signed", "autograph"],
        ItemCategory.BOOKS: ["book", "novel", "author", "page", "edition", "publish", "hardcover", "paperback"],
        ItemCategory.TOYS: ["toy", "game", "play", "action figure", "doll", "puzzle", "lego", "model kit"]
    }
    
    scores = {category: 0 for category in ItemCategory}
    scores[ItemCategory.UNKNOWN] = 0
    
    for category, keywords in category_keywords.items():
        for keyword in keywords:
            if keyword in title_lower or keyword in description_lower:
                scores[category] += 1
    
    return max(scores.items(), key=lambda x: x[1])[0]

def enhance_with_ebay_data(item_data: Dict) -> Dict:
    """Enhance AI analysis with real eBay market data"""
    try:
        # Create search keywords from item data
        keywords = []
        if item_data.get('brand'):
            keywords.append(item_data['brand'])
        if item_data.get('model'):
            keywords.append(item_data['model'])
        
        # Fallback to title keywords if no brand/model
        if not keywords:
            title_words = item_data.get('title', '').split()[:3]  # First 3 words
            keywords = [word for word in title_words if len(word) > 2]
        
        search_query = ' '.join(keywords)
        
        if search_query.strip():
            logger.info(f"Searching eBay for: {search_query}")
            market_analysis = ebay_api.analyze_market_trends(search_query)
            
            # Update item data with real eBay market data
            if market_analysis['confidence'] in ['high', 'medium']:
                item_data['market_insights'] = f"eBay Analysis: {market_analysis['market_notes']}. " + item_data.get('market_insights', '')
                item_data['price_range'] = market_analysis['price_range']
                item_data['suggested_cost'] = f"${market_analysis['recommended_price']:.2f}"
                
                # Calculate profit potential
                avg_price = market_analysis['average_price']
                recommended_cost = market_analysis['recommended_price']
                ebay_fees = avg_price * 0.13  # 13% eBay fees
                shipping_cost = 10.00  # Average shipping
                profit = avg_price - recommended_cost - ebay_fees - shipping_cost
                
                item_data['profit_potential'] = f"${profit:.2f} profit (after fees)" if profit > 0 else "Low profit margin"
                
            # Add eBay-specific tips
            item_data['ebay_specific_tips'] = [
                "Use all 12 photo slots with detailed shots",
                "Include measurements and condition details",
                "List during peak hours (Sun-Wed evenings)",
                "Consider auction format for rare/uncertain value items",
                f"Sell-through rate: {market_analysis['sell_through_rate']}%"
            ]
            
            logger.info(f"Enhanced with eBay data: confidence={market_analysis['confidence']}")
        else:
            logger.warning("No suitable keywords found for eBay search")
            
        return item_data
        
    except Exception as e:
        logger.error(f"eBay data enhancement failed: {e}")
        return item_data

# ENHANCED DATA MODELS
class EnhancedAppItem:
    def __init__(self, data: Dict[str, Any]):
        self.title = data.get("title", "Unknown Item")
        self.description = data.get("description", "No description available")
        self.price_range = data.get("price_range", "$0-0")
        self.resellability_rating = min(10, max(1, data.get("resellability_rating", 5)))
        self.suggested_cost = data.get("suggested_cost", "$0")
        self.market_insights = data.get("market_insights", "No market data available")
        self.authenticity_checks = data.get("authenticity_checks", "Check item condition")
        self.profit_potential = data.get("profit_potential", "Unknown")
        self.category = data.get("category", "unknown")
        self.ebay_specific_tips = data.get("ebay_specific_tips", [])
        
        # Extended properties
        self.brand = data.get("brand", "")
        self.model = data.get("model", "")
        self.year = data.get("year", "")
        self.condition = data.get("condition", "")
        self.confidence = data.get("confidence", 0.5)
        self.key_features = data.get("key_features", [])
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "description": self.description,
            "price_range": self.price_range,
            "resellability_rating": self.resellability_rating,
            "suggested_cost": self.suggested_cost,
            "market_insights": self.market_insights,
            "authenticity_checks": self.authenticity_checks,
            "profit_potential": self.profit_potential,
            "category": self.category,
            "ebay_specific_tips": self.ebay_specific_tips,
            "brand": self.brand,
            "model": self.model,
            "year": self.year,
            "condition": self.condition,
            "confidence": self.confidence,
            "key_features": self.key_features
        }

def parse_json_response(response_text: str) -> List[Dict]:
    """Extract JSON from AI response with robust error handling"""
    try:
        json_text = response_text.strip()
        
        # Clean up JSON response
        if "```json" in json_text:
            json_start = json_text.find("```json") + 7
            json_end = json_text.find("```", json_start)
            json_text = json_text[json_start:json_end].strip()
        elif "```" in json_text:
            json_start = json_text.find("```") + 3
            json_end = json_text.rfind("```")
            json_text = json_text[json_start:json_end].strip()
        
        # Remove any non-JSON content
        json_match = re.search(r'\[.*\]|\{.*\}', json_text, re.DOTALL)
        if json_match:
            json_text = json_match.group(0)
        
        parsed_data = json.loads(json_text)
        
        # Ensure we always return a list of dictionaries
        if isinstance(parsed_data, dict):
            return [parsed_data]
        elif isinstance(parsed_data, list):
            return parsed_data
        else:
            logger.warning(f"Unexpected JSON format: {type(parsed_data)}")
            return []
            
    except Exception as e:
        logger.warning(f"JSON parsing failed: {e}")
        logger.warning(f"Response text that failed: {response_text[:200]}...")
        return []

def process_image_standard(job_data: Dict) -> Dict:
    """Enhanced single-stage processing with eBay market integration"""
    try:
        image_base64 = job_data['image_base64']
        mime_type = job_data['mime_type']
        
        image_part = {
            "mime_type": mime_type,
            "data": image_base64
        }
        
        prompt = market_analysis_prompt
        if job_data.get('title'):
            prompt += f"\nUser-provided title: {job_data['title']}"
        if job_data.get('description'):
            prompt += f"\nUser-provided description: {job_data['description']}"
        
        logger.info("Starting enhanced market analysis...")
        response = model.generate_content([prompt, image_part])
        logger.info("AI analysis completed, parsing response...")
        
        ai_response = parse_json_response(response.text)
        logger.info(f"Parsed {len(ai_response)} items from response")
        
        enhanced_items = []
        for item_data in ai_response:
            if isinstance(item_data, dict):
                # Enhance with real eBay market data
                item_data = enhance_with_ebay_data(item_data)
                
                detected_category = detect_category(item_data.get("title", ""), item_data.get("description", ""))
                item_data["category"] = detected_category.value
                
                enhanced_items.append(EnhancedAppItem(item_data).to_dict())
            else:
                logger.warning(f"Skipping non-dictionary item: {item_data}")
        
        return {
            "status": "completed",
            "result": {
                "message": "Enhanced analysis with eBay market data completed",
                "items": enhanced_items,
                "processing_time": "25s",
                "analysis_stages": 1,
                "confidence_level": "enhanced_with_market_data",
                "analysis_timestamp": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Enhanced processing failed: {e}")
        return {"status": "failed", "error": str(e)}

# MAIN UPLOAD ENDPOINT
@app.post("/upload_item/")
async def create_upload_file(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    accuracy_mode: str = Form("maximum")
):
    try:
        # Read the image
        image_bytes = await file.read()
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        
        # Create job ID
        job_id = str(uuid.uuid4())
        
        # Store job data
        job_data = {
            'image_base64': image_base64,
            'mime_type': file.content_type,
            'title': title,
            'description': description,
            'accuracy_mode': accuracy_mode,
            'status': 'queued',
            'created_at': datetime.now().isoformat()
        }
        
        with job_lock:
            job_storage[job_id] = job_data
        
        # Add to queue for processing
        job_queue.put(job_id)
        
        logger.info(f"Job {job_id} queued for processing")
        
        return {
            "message": "Analysis queued",
            "job_id": job_id,
            "status": "queued",
            "estimated_time": "60-75 seconds" if accuracy_mode == "maximum" else "20-30 seconds",
            "check_status_url": f"/job/{job_id}/status"
        }
            
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

# Add job status endpoint
@app.get("/job/{job_id}/status")
async def get_job_status(job_id: str):
    """Check status of a job"""
    with job_lock:
        job_data = job_storage.get(job_id)
    
    if not job_data:
        raise HTTPException(status_code=404, detail="Job not found")
    
    response = {
        "job_id": job_id,
        "status": job_data.get('status', 'unknown'),
        "created_at": job_data.get('created_at')
    }
    
    if job_data.get('status') == 'completed':
        response["result"] = job_data.get('result')
        response["completed_at"] = job_data.get('completed_at')
    elif job_data.get('status') == 'failed':
        response["error"] = job_data.get('error', 'Unknown error')
        response["completed_at"] = job_data.get('completed_at')
    elif job_data.get('status') == 'processing':
        response["started_at"] = job_data.get('started_at')
    
    return response

# Add cleanup endpoint (optional)
@app.delete("/job/{job_id}")
async def delete_job(job_id: str):
    """Remove a job from storage"""
    with job_lock:
        if job_id in job_storage:
            del job_storage[job_id]
            return {"message": "Job deleted"}
        else:
            raise HTTPException(status_code=404, detail="Job not found")

# eBay API test endpoint
@app.get("/test_ebay/{keywords}")
async def test_ebay_search(keywords: str):
    try:
        completed_items = ebay_api.search_completed_items(keywords, max_results=10)
        market_analysis = ebay_api.analyze_market_trends(keywords)
        
        return {
            "search_query": keywords,
            "completed_items_found": len(completed_items),
            "sample_items": completed_items[:3],
            "market_analysis": market_analysis
        }
    except Exception as e:
        logger.error(f"eBay test failed: {e}")
        return {"error": str(e), "status": "failed"}

# Health check with eBay status
@app.get("/health")
async def health_check():
    ebay_status = "configured"
    try:
        # Test eBay API with a simple search
        test_results = ebay_api.search_completed_items("iphone", max_results=1)
        ebay_status = "working" if test_results else "no_results"
    except Exception as e:
        ebay_status = f"error: {str(e)}"
    
    return {
        "status": "healthy",
        "ai_configured": bool(model),
        "ebay_status": ebay_status,
        "model": "gemini-2.5-flash-preview-05-20",
        "version": "3.1.0",
        "processing_mode": "enhanced_with_market_data",
        "features": [
            "Real eBay market data integration",
            "Completed listings analysis",
            "Profit calculation with fees",
            "Market trend analysis"
        ]
    }

@app.get("/")
async def root():
    return {
        "message": "AI Resell Pro API v3.1 - Now with Real eBay Market Data! 🚀",
        "status": "healthy",
        "features": [
            "Real eBay completed listings analysis",
            "Market trend analysis with confidence scoring",
            "Profit calculations after eBay fees",
            "Enhanced accuracy mode"
        ],
        "endpoints": {
            "upload": "/upload_item",
            "health": "/health",
            "test_ebay": "/test_ebay/{keywords}"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
