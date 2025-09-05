from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, BackgroundTasks
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
from redis import Redis
from rq import Queue
from rq.job import Job
import threading

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Resell Pro API", version="3.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Redis for job queue
redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
redis_conn = Redis.from_url(redis_url)
job_queue = Queue(connection=redis_conn, default_timeout=600)  # 10 minute timeout

# Job storage
jobs = {}

# Configure Google Generative AI
try:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set")
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash-preview-05-20')
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

# CATEGORY-SPECIFIC PROMPTS
category_prompts = {
    ItemCategory.ELECTRONICS: """
    FOCUS ON: Model numbers, serial numbers, technical specifications, manufacturer logos, condition indicators.
    IMPORTANT: Check for common failure points, battery health indicators, screen condition, and accessory compatibility.
    PRICING: Consider depreciation curves for electronics and compatibility with modern systems.
    """,
    
    ItemCategory.CLOTHING: """
    FOCUS ON: Brand tags, size labels, material composition, care instructions, signs of wear.
    IMPORTANT: Look for authenticity markers, vintage indicators, designer signatures, and fabric quality.
    PRICING: Consider seasonal trends, brand popularity, and condition factors like stains or fading.
    """,
    
    ItemCategory.FURNITURE: """
    FOCUS ON: Construction quality, wood type, joinery methods, manufacturer marks, upholstery condition.
    IMPORTANT: Check for structural integrity, repairs, woodworm, and original vs replacement parts.
    PRICING: Consider mid-century modern premium, designer pieces, and regional style preferences.
    """,
    
    ItemCategory.COLLECTIBLES: """
    FOCUS ON: Edition numbers, signatures, certificates of authenticity, rarity indicators.
    IMPORTANT: Look for reproduction markers, condition issues that affect collectibility, and complete sets.
    PRICING: Consider auction records, collector demand spikes, and cultural relevance.
    """,
    
    ItemCategory.BOOKS: """
    FOCUS ON: Edition information, dust jackets, inscriptions, publisher marks, condition of binding.
    IMPORTANT: Check for first edition indicators, author signatures, and completeness of multi-volume sets.
    PRICING: Consider rarity, author popularity, and cultural significance.
    """,
    
    ItemCategory.TOYS: """
    FOCUS ON: Manufacturer stamps, copyright dates, completeness, battery compartments.
    IMPORTANT: Look for vintage vs reproduction, working condition, and original packaging.
    PRICING: Consider nostalgia factor, complete sets, and working condition.
    """
}

# ENHANCED MARKET ANALYSIS PROMPT (25-second "maximum" quality)
market_analysis_prompt = """
ULTRA-ACCURATE 25-SECOND ANALYSIS - MAXIMUM DETAIL IN MINIMUM TIME:

You are an expert forensic resell analyst. Achieve maximum accuracy in a single pass by:

🔍 **CRITICAL FOCUS AREAS:**
- ZOOM IN on EVERY character, number, logo, symbol, serial number
- Identify EXACT model numbers, years, editions with 100% certainty
- Capture ALL visible text no matter how small
- Analyze materials, construction quality, condition markers

📊 **COMPREHENSIVE OUTPUT REQUIREMENTS:**
For EACH valuable item, provide ALL these details in JSON:

{
  "title": "eBay-optimized title with EXACT model numbers and key search terms",
  "description": "Detailed description with ALL visible identifying features, condition notes, and specifications",
  "price_range": "Current eBay market range based on recent sold listings for IDENTICAL items",
  "resellability_rating": 8,
  "suggested_cost": "Maximum to pay for profitable resale",
  "market_insights": "eBay-specific demand analysis and sell-through rates",
  "authenticity_checks": "Specific authenticity markers to verify",
  "profit_potential": "Estimated profit after eBay fees and shipping",
  "category": "Most relevant eBay category",
  "ebay_specific_tips": ["Tip 1", "Tip 2", "Tip 3"],
  
  // EXTENDED DETAILS (capture everything visible):
  "brand": "EXACT brand from logos/text",
  "model": "EXACT model number from visible text", 
  "year": "Production year if identifiable",
  "condition": "Detailed condition assessment",
  "confidence": 0.95,
  "analysis_depth": "25-second comprehensive",
  "processing_time_seconds": 25
}

🎯 **SUCCESS CRITERIA:**
- NO GUESSING: Only report what you can see with certainty
- MAXIMUM PRECISION: Every character matters - zoom in completely
- EBAY-FOCUSED: Base everything on actual eBay market data
- TIME-AWARE: Complete analysis in under 25 seconds

IMPORTANT: This is a SINGLE-PASS analysis but must achieve NEAR-MAXIMUM accuracy.
Focus intensely on visible details and provide complete eBay market context.
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
        keywords = f"{item_data.get('brand', '')} {item_data.get('title', '')}".strip()
        completed_items = ebay_api.search_completed_items(keywords, max_results=10)
        
        if completed_items:
            prices = [item['price'] for item in completed_items]
            avg_price = sum(prices) / len(prices)
            
            if 'price_range' in item_data:
                item_data['market_insights'] += f" Based on {len(completed_items)} recent eBay sales, similar items sold for ${min(prices):.2f}-${max(prices):.2f}."
                item_data['price_range'] = f"${min(prices):.2f}-${max(prices):.2f}"
            
            item_data['ebay_specific_tips'] = [
                "Use high-quality photos from multiple angles",
                "Include measurements and detailed condition description",
                "Offer combined shipping for multiple items",
                "Consider auction format for rare items with uncertain value"
            ]
            
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
            "ebay_specific_tips": self.ebay_specific_tips
        }

# JOB PROCESSING FUNCTIONS
def parse_json_response(response_text: str) -> List[Dict]:
    """Extract JSON from AI response with robust error handling"""
    try:
        json_text = response_text.strip()
        
        if "```json" in json_text:
            json_start = json_text.find("```json") + 7
            json_end = json_text.find("```", json_start)
            json_text = json_text[json_start:json_end].strip()
        elif "```" in json_text:
            json_start = json_text.find("```") + 3
            json_end = json_text.rfind("```")
            json_text = json_text[json_start:json_end].strip()
        
        json_text = re.sub(r'^[^{]*', '', json_text)
        json_text = re.sub(r'[^}]*$', '', json_text)
        
        return json.loads(json_text)
        
    except Exception as e:
        logger.warning(f"JSON parsing failed: {e}")
        return []

def combine_stage_results(stage1: List, stage2: List, stage3: List) -> List[Dict]:
    """Combine results from all 3 stages into ultimate accuracy output"""
    combined = []
    
    for i, (s1, s2, s3) in enumerate(zip(stage1, stage2, stage3)):
        combined_item = {
            "item_id": i + 1,
            "title": s2.get("title", s1.get("title", "Unknown Item")),
            "brand": s2.get("brand", s1.get("brand", "")),
            "model": s2.get("model", s1.get("model", "")),
            "year": s2.get("year", s1.get("year", "")),
            "category": s2.get("category", s1.get("category", "")),
            "description": s2.get("description", s1.get("description", "")),
            "condition": s2.get("condition", s1.get("condition", "")),
            "authenticity_checks": s2.get("authenticity_checks", []),
            "price_range": s3.get("price_range", "$0-0"),
            "suggested_cost": s3.get("suggested_cost", "$0"),
            "profit_potential": s3.get("profit_potential", "$0"),
            "market_insights": s3.get("market_insights", ""),
            "resellability_rating": s3.get("resellability_rating", 5),
            "ebay_specific_tips": s3.get("ebay_specific_tips", []),
            "confidence": s2.get("confidence", 0.95),
            "analysis_depth": "3-stage maximum accuracy",
            "processing_time_seconds": 75
        }
        combined.append(combined_item)
    
    return combined

def process_image_max_accuracy(job_data: Dict) -> Dict:
    """3-STAGE ULTIMATE ACCURACY PROCESSING - 75 seconds total"""
    try:
        image_base64 = job_data['image_base64']
        mime_type = job_data['mime_type']
        
        image_part = {
            "mime_type": mime_type,
            "data": image_base64
        }
        
        # STAGE 1: BROAD IDENTIFICATION (25s)
        stage1_prompt = """
        ULTIMATE ACCURACY STAGE 1/3: BROAD OBJECT IDENTIFICATION
        Identify EVERY potential resellable item in this image with 90%+ confidence.
        Focus on electronics, designer fashion, collectibles, quality furniture.
        Output: JSON array with initial identifications and confidence scores.
        """
        
        logger.info("STAGE 1: Starting broad identification (25s)")
        stage1_response = model.generate_content([stage1_prompt, image_part])
        stage1_results = parse_json_response(stage1_response.text)
        logger.info(f"STAGE 1: Found {len(stage1_results)} potential items")
        
        # STAGE 2: PRECISE IDENTIFICATION (25s)
        stage2_prompt = """
        ULTIMATE ACCURACY STAGE 2/3: PRECISE ITEM IDENTIFICATION
        For EACH item from Stage 1, achieve 100% certainty.
        Zoom in on EVERY visible character, number, logo, and symbol.
        Identify EXACT model numbers, years, editions, variants.
        Output: JSON array with definitive identifications.
        """
        
        logger.info("STAGE 2: Starting precise identification (25s)")
        stage2_response = model.generate_content([stage2_prompt, image_part])
        stage2_results = parse_json_response(stage2_response.text)
        logger.info(f"STAGE 2: Precisely identified {len(stage2_results)} items")
        
        # STAGE 3: MARKET ANALYSIS (25s)
        stage3_prompt = """
        ULTIMATE ACCURACY STAGE 3/3: REAL-TIME MARKET ANALYSIS
        Determine EXACT current market value and profit potential for each identified item.
        Analyze RECENT eBay sold listings for identical items.
        Calculate REAL profit margins after fees.
        Output: JSON array with complete market analysis.
        """
        
        logger.info("STAGE 3: Starting market analysis (25s)")
        stage3_response = model.generate_content([stage3_prompt, image_part])
        stage3_results = parse_json_response(stage3_response.text)
        logger.info(f"STAGE 3: Completed market analysis for {len(stage3_results)} items")
        
        final_results = combine_stage_results(stage1_results, stage2_results, stage3_results)
        
        return {
            "status": "completed",
            "result": {
                "message": "3-STAGE ULTIMATE ACCURACY ANALYSIS COMPLETE",
                "items": final_results,
                "processing_time": "75s",
                "analysis_stages": 3,
                "confidence_level": "maximum",
                "analysis_timestamp": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"3-stage processing failed: {e}")
        return {"status": "failed", "error": str(e)}

def process_image_standard(job_data: Dict) -> Dict:
    """Enhanced single-stage processing (25s) with maximum-like quality"""
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
        
        response = model.generate_content([prompt, image_part])
        ai_response = parse_json_response(response.text)
        
        enhanced_items = []
        for item_data in ai_response:
            detected_category = detect_category(item_data.get("title", ""), item_data.get("description", ""))
            item_data["category"] = detected_category.value
            item_data = enhance_with_ebay_data(item_data)
            
            if detected_category in category_prompts:
                item_data["market_insights"] += f" {category_prompts[detected_category]}"
            
            enhanced_items.append(EnhancedAppItem(item_data).to_dict())
        
        return {
            "status": "completed",
            "result": {
                "message": "Enhanced 25-second analysis completed",
                "items": enhanced_items,
                "processing_time": "25s",
                "analysis_stages": 1,
                "confidence_level": "enhanced_standard",
                "analysis_timestamp": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Enhanced processing failed: {e}")
        return {"status": "failed", "error": str(e)}

# EBAY AUTH ENDPOINTS
@app.get("/auth/ebay")
async def auth_ebay():
    auth_url = get_authorization_url()
    if auth_url:
        return RedirectResponse(url=auth_url)
    else:
        raise HTTPException(status_code=500, detail="Failed to generate auth URL")

@app.get("/auth/callback")
async def auth_callback(SessID: str, runame: str):
    try:
        token = await exchange_session_for_token(SessID)
        if token:
            os.environ['EBAY_AUTH_TOKEN'] = token
            return RedirectResponse(url="/auth/success")
        else:
            return RedirectResponse(url="/auth/failed")
    except Exception as e:
        logger.error(f"Auth callback error: {e}")
        return RedirectResponse(url="/auth/failed")

@app.get("/auth/success")
async def auth_success():
    return HTMLResponse("""
    <html><body style="font-family: Arial, sans-serif; text-align: center; padding: 50px;">
        <h1 style="color: green;">✅ Authentication Successful!</h1>
        <p>You have successfully authenticated with eBay.</p>
        <p><a href="/" style="color: #007bff; text-decoration: none;">Return to ReReSell App</a></p>
    </body></html>
    """)

@app.get("/auth/failed")
async def auth_failed():
    return HTMLResponse("""
    <html><body style="font-family: Arial, sans-serif; text-align: center; padding: 50px;">
        <h1 style="color: red;">❌ Authentication Failed</h1>
        <p>There was an issue authenticating with eBay.</p>
        <p><a href="/auth/ebay" style="color: #007bff; text-decoration: none;">Try again</a></p>
    </body></html>
    """)

@app.get("/privacy")
async def privacy_policy():
    return HTMLResponse("""
    <html><body style="font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px;">
        <h1>Privacy Policy</h1>
        <p><strong>ReReSell AI eBay Assistant</strong> respects your privacy.</p>
    </body></html>
    """)

# MAIN UPLOAD ENDPOINT WITH FREE PLAN OPTIMIZATION
@app.post("/upload_item/")
async def create_upload_file(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    accuracy_mode: str = Form("maximum")
):
    try:
        image_bytes = await file.read()
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        
        # FREE PLAN SMART ADJUSTMENT: Use enhanced standard mode for all requests
        # This prevents 30-second timeout on Render's free plan while maintaining quality
        effective_mode = "enhanced_standard"
        if accuracy_mode == "maximum":
            logger.info("Using enhanced standard mode (25s) instead of maximum (75s) to avoid timeout")
        
        # Process with enhanced standard mode (25-second maximum-like quality)
        result = process_image_standard({
            'image_base64': image_base64,
            'mime_type': file.content_type,
            'title': title,
            'description': description
        })
        
        if result['status'] == 'completed':
            return result['result']
        else:
            raise HTTPException(status_code=500, detail=result['error'])
            
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

# JOB STATUS ENDPOINTS (kept for compatibility)
@app.get("/job/{job_id}/status")
async def get_job_status(job_id: str):
    return {
        "status": "completed",
        "message": "Direct processing mode - jobs complete immediately"
    }

@app.get("/job/{job_id}/result")
async def get_job_result(job_id: str):
    raise HTTPException(status_code=400, detail="Direct processing mode - use /upload_item directly")

# DEBUG ENDPOINTS
@app.get("/debug/endpoints")
async def debug_endpoints():
    import re
    endpoints = []
    with open(__file__, 'r') as f:
        content = f.read()
        routes = re.findall(r'@app\.(get|post|put|delete)\(["\']([^"\']+)["\']\)', content)
        for method, path in routes:
            endpoints.append(f"{method.upper()} {path}")
    
    return {
        "endpoints": endpoints,
        "auth_configured": "auth/ebay" in [e for e in endpoints if "auth" in e],
        "total_endpoints": len(endpoints)
    }

@app.get("/debug/routes")
async def debug_routes():
    routes = []
    for route in app.routes:
        routes.append({
            "path": route.path,
            "name": route.name,
            "methods": getattr(route, "methods", [])
        })
    return {"routes": routes}

@app.get("/debug/jobs")
async def debug_jobs():
    return {
        "total_jobs": len(jobs),
        "message": "Direct processing mode - no job queue active"
    }

# BATCH PROCESSING ENDPOINT
@app.post("/batch_analyze/")
async def batch_analyze_items(files: List[UploadFile] = File(...)):
    try:
        results = []
        for file in files:
            image_bytes = await file.read()
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            
            result = process_image_standard({
                'image_base64': image_base64,
                'mime_type': file.content_type
            })
            
            if result['status'] == 'completed':
                results.extend(result['result']['items'])
        
        return {
            "message": f"Processed {len(files)} images",
            "total_items": len(results),
            "items": results,
            "processing_mode": "enhanced_standard",
            "batch_timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Batch analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")

@app.post("/list_to_ebay/")
async def list_to_ebay(item_data: Dict):
    try:
        result = ebay_api.list_item(item_data)
        return result
    except Exception as e:
        logger.error(f"eBay listing failed: {e}")
        raise HTTPException(status_code=500, detail=f"eBay listing failed: {str(e)}")

@app.get("/ebay_search/{keywords}")
async def ebay_search(keywords: str, category: Optional[str] = None):
    try:
        results = ebay_api.search_completed_items(keywords, category)
        return {"results": results}
    except Exception as e:
        logger.error(f"eBay search failed: {e}")
        raise HTTPException(status_code=500, detail=f"eBay search failed: {str(e)}")

@app.get("/")
async def root():
    return {
        "message": "AI Resell Pro API v3.0 - Enhanced 25s Analysis! 🚀",
        "status": "healthy",
        "features": [
            "Enhanced 25-second analysis",
            "Maximum-like accuracy in minimum time",
            "eBay market integration",
            "Free plan optimized (no timeout)"
        ],
        "endpoints": {
            "upload": "/upload_item",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "ai_configured": bool(model),
        "ebay_configured": bool(ebay_api),
        "redis_connected": redis_conn.ping() if redis_conn else False,
        "model": "gemini-2.5-flash-preview-05-20",
        "version": "3.0.0",
        "processing_mode": "enhanced_standard",
        "timeout_optimized": True
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)