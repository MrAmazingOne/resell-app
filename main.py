# AI RESELL PRO API - COMPLETE UPDATED VERSION
# Enhanced with OAuth 2.0, Taxonomy API, Marketing API, and Vision Analysis

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from groq import Groq
from ebay_oauth import ebay_oauth
import uuid
import os
import json
from typing import Optional, List, Dict, Any
import logging
import base64
import requests
import re
from datetime import datetime, timedelta
from dotenv import load_dotenv
import threading
import queue
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

# Image analysis imports (essential for Lift)
import cv2
import numpy as np
from PIL import Image
import io

# Custom modules
from ebay_integration import ebay_api

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Resell Pro API - Enhanced Edition",
    version="5.0.0",
    description="Complete resell analysis system with eBay OAuth 2.0, Taxonomy API, Marketing API, and Vision Analysis",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============= CONFIGURATION =============

# Configure Groq client
try:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable not set")
    
    groq_client = Groq(api_key=api_key)
    groq_model = "meta-llama/llama-4-scout-17b-16e-instruct"
    logger.info(f"✅ Groq configured successfully with model: {groq_model}")
except Exception as e:
    logger.error(f"❌ Failed to configure Groq: {e}")
    groq_client = None
    groq_model = None

# Initialize eBay API (will verify OAuth token)
try:
    ebay_api_instance = ebay_api
    logger.info("✅ eBay API initialized")
except Exception as e:
    logger.error(f"❌ Failed to initialize eBay API: {e}")
    ebay_api_instance = None

# ============= JOB QUEUE SYSTEM =============

# In-memory job queue for async processing
job_queue = queue.Queue()
job_storage = {}
job_lock = threading.Lock()
job_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="JobWorker")

# Activity tracking for keep-alive
last_activity = time.time()
activity_lock = threading.Lock()

# eBay Token Storage
EBAY_AUTH_TOKEN = None
EBAY_TOKEN_LOCK = threading.Lock()

# ============= HELPER FUNCTIONS =============

def update_activity():
    """Update last activity timestamp"""
    with activity_lock:
        global last_activity
        last_activity = time.time()

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

def calculate_resellability(analysis: Dict) -> int:
    """Calculate resellability rating 1-10 based on market analysis"""
    rating = 5  # Base rating
    
    # Adjust based on sample size
    sample_size = analysis.get('sample_size', 0)
    if sample_size >= 50:
        rating += 3
    elif sample_size >= 20:
        rating += 2
    elif sample_size >= 10:
        rating += 1
    elif sample_size < 5:
        rating -= 2
    
    # Adjust based on confidence
    confidence = analysis.get('confidence', 'low')
    if confidence == 'high':
        rating += 2
    elif confidence == 'good':
        rating += 1
    elif confidence == 'very low':
        rating -= 2
    
    # Adjust based on price stability
    if analysis.get('price_stability') == 'high':
        rating += 1
    
    # Ensure rating stays within 1-10
    return max(1, min(10, rating))

def generate_ebay_tips(analysis: Dict) -> List[str]:
    """Generate eBay-specific tips based on analysis"""
    tips = []
    
    sample_size = analysis.get('sample_size', 0)
    confidence = analysis.get('confidence', 'low')
    
    if sample_size >= 20:
        tips.append("Strong market data - list with confidence")
    elif sample_size >= 10:
        tips.append("Moderate data - consider checking similar items")
    else:
        tips.append("Limited data - research similar items before listing")
    
    if confidence in ['high', 'good']:
        tips.append("Price competitively based on recent sold data")
    
    if analysis.get('days_since_last_sale', 999) <= 7:
        tips.append("Very active market - list now for best results")
    
    tips.append("Use clear photos and detailed description")
    tips.append("Consider free shipping to increase visibility")
    
    return tips

def extract_vision_keywords(vision_analysis: Dict) -> List[str]:
    """Extract keywords specifically from vision analysis"""
    keywords = []
    
    if not vision_analysis:
        return keywords
    
    # From detected objects
    detected_objects = vision_analysis.get('detected_objects', [])
    for obj in detected_objects:
        # Split compound objects
        obj_words = re.findall(r'\b\w+\b', obj.lower())
        keywords.extend([w for w in obj_words if len(w) > 2])
    
    # From color analysis
    if vision_analysis.get('dominant_color'):
        color = vision_analysis['dominant_color'].lower()
        keywords.append(color)
        keywords.append(f"{color} color")
    
    # From vehicle detection
    if vision_analysis.get('likely_vehicle', False):
        keywords.append('vehicle')
        if vision_analysis.get('likely_full_vehicle', False):
            keywords.append('complete vehicle')
            keywords.append('entire vehicle')
        else:
            keywords.append('vehicle part')
            keywords.append('component')
    
    # From size/scale indicators
    if vision_analysis.get('size_category'):
        size = vision_analysis['size_category'].lower()
        keywords.append(size)
        keywords.append(f"{size} size")
    
    return list(set(keywords))

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
        
        return {
            "success": True,
            "auth_url": auth_url,
            "state": state,
            "redirect_uri": "https://resell-app-bi47.onrender.com/ebay/oauth/callback",
            "timestamp": datetime.now().isoformat(),
            "token_duration": "permanent (2-year refresh token)",
            "required_scopes": [
                "https://api.ebay.com/oauth/api_scope",
                "https://api.ebay.com/oauth/api_scope/commerce.taxonomy",
                "https://api.ebay.com/oauth/api_scope/sell.marketing.readonly"
            ]
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
                        <p>All API access enabled</p>
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
            available_tokens = list(ebay_oauth.tokens.keys())
            logger.info(f"   Available tokens: {available_tokens}")
            
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

# ============= VISION ANALYSIS =============

def analyze_image_with_vision(image_data: bytes) -> Dict:
    """Analyze image to extract visual features and keywords"""
    try:
        # Convert to PIL Image
        image = Image.open(io.BytesIO(image_data))
        
        # Basic analysis
        analysis = {
            "width": image.width,
            "height": image.height,
            "format": image.format,
            "detected_objects": [],
            "suggested_keywords": [],
            "dominant_color": None,
            "likely_vehicle": False,
            "likely_full_vehicle": False,
            "size_category": "medium"
        }
        
        # Try to detect if this is a vehicle image
        try:
            # Convert to OpenCV format
            nparr = np.frombuffer(image_data, np.uint8)
            img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img_cv is not None:
                # Simple color analysis
                avg_color = np.mean(img_cv, axis=(0, 1))
                colors = ['Blue', 'Green', 'Red']
                dominant_color_idx = np.argmax(avg_color)
                analysis["dominant_color"] = colors[dominant_color_idx]
                
                # Check if image shows full vehicle vs part
                height, width = img_cv.shape[:2]
                aspect_ratio = width / height
                
                # Detect if it looks like a vehicle
                if 1.2 <= aspect_ratio <= 2.5:  # Typical vehicle aspect ratios
                    analysis["likely_vehicle"] = True
                    analysis["likely_full_vehicle"] = True if aspect_ratio > 1.5 else False
                    
                # Size estimation based on aspect ratio
                if aspect_ratio > 2.0:
                    analysis["size_category"] = "large"
                elif aspect_ratio < 0.8:
                    analysis["size_category"] = "small"
                
                # Simple object detection based on color distribution
                color_std = np.std(img_cv)
                if color_std < 30:
                    analysis["detected_objects"].append("solid color object")
                elif color_std > 60:
                    analysis["detected_objects"].append("multicolor object")
                    
        except Exception as e:
            logger.debug(f"CV2 analysis skipped: {e}")
        
        # Generate suggested keywords based on analysis
        if analysis["likely_vehicle"]:
            analysis["suggested_keywords"].extend(["vehicle", "auto", "transport"])
            if analysis["likely_full_vehicle"]:
                analysis["suggested_keywords"].extend(["complete", "whole", "entire"])
            else:
                analysis["suggested_keywords"].extend(["part", "component", "piece"])
        
        if analysis["dominant_color"]:
            analysis["suggested_keywords"].append(analysis["dominant_color"].lower())
        
        logger.info(f"✅ Vision analysis complete: {len(analysis['suggested_keywords'])} keywords generated")
        return analysis
    except Exception as e:
        logger.error(f"❌ Vision analysis error: {e}")
        return {"error": str(e)}

# ============= IMAGE UPLOAD ENDPOINT =============

@app.post("/upload_item/")
async def create_upload_file(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    description: Optional[str] = Form(None)
):
    """Upload image for analysis with eBay APIs"""
    update_activity()
    
    try:
        # Check eBay authentication
        ebay_token = get_ebay_token()
        if not ebay_token:
            raise HTTPException(
                status_code=400, 
                detail="eBay authentication required. Please connect via /ebay/oauth/start"
            )
        
        # Check if eBay API is initialized
        if not ebay_api_instance:
            raise HTTPException(
                status_code=500,
                detail="eBay API not configured. Check server logs."
            )
        
        # Read and validate image
        image_bytes = await file.read()
        if len(image_bytes) > 8 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Image too large (max 8MB)")
        
        # Create job ID
        job_id = str(uuid.uuid4())
        
        # Vision analysis for keyword extraction
        vision_analysis = analyze_image_with_vision(image_bytes)
        
        # Store job data
        with job_lock:
            # Clean up old jobs (older than 2 hours)
            current_time = datetime.now()
            for old_id, old_job in list(job_storage.items()):
                try:
                    created = datetime.fromisoformat(old_job.get('created_at', ''))
                    if (current_time - created).seconds > 7200:
                        del job_storage[old_id]
                except:
                    pass
            
            job_storage[job_id] = {
                'image_bytes': image_bytes,
                'mime_type': file.content_type,
                'title': title.strip() if title else "",
                'description': description.strip() if description else "",
                'vision_analysis': vision_analysis,
                'status': 'queued',
                'created_at': datetime.now().isoformat(),
                'requires_ebay_auth': not bool(ebay_token)
            }
        
        # Queue job for processing
        job_queue.put(job_id)
        logger.info(f"📤 Job {job_id} queued for processing")
        
        return {
            "message": "Analysis queued with enhanced eBay API integration",
            "job_id": job_id,
            "status": "queued",
            "estimated_time": "25-30 seconds",
            "check_status_url": f"/job/{job_id}/status",
            "ebay_auth_status": "connected" if ebay_token else "required",
            "features_enabled": [
                "eBay Taxonomy API",
                "eBay Marketing API", 
                "Vision Analysis",
                "Market Trend Analysis"
            ]
        }
            
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"❌ Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)[:200])

@app.get("/job/{job_id}/status")
async def get_job_status(job_id: str):
    """Check status of a processing job"""
    update_activity()
    
    with job_lock:
        job_data = job_storage.get(job_id)
    
    if not job_data:
        raise HTTPException(status_code=404, detail="Job not found")
    
    response = {
        "job_id": job_id,
        "status": job_data.get('status', 'unknown'),
        "created_at": job_data.get('created_at'),
        "requires_ebay_auth": job_data.get('requires_ebay_auth', False)
    }
    
    if job_data.get('status') == 'completed':
        result = job_data.get('result', {})
        response["result"] = result
        response["completed_at"] = job_data.get('completed_at')
        
        # Ensure items key exists for iOS compatibility
        if "items" not in result:
            response["result"]["items"] = []
            
    elif job_data.get('status') == 'failed':
        response["error"] = job_data.get('error', 'Unknown error')
        response["completed_at"] = job_data.get('completed_at')
    elif job_data.get('status') == 'processing':
        response["started_at"] = job_data.get('started_at')
    
    return response

# ============= JOB PROCESSING =============

def process_image_job(job_data: Dict) -> Dict:
    """Process image with enhanced eBay API integration"""
    try:
        # Verify eBay API is available
        if not ebay_api_instance:
            return {"status": "failed", "error": "eBay API not configured"}
        
        if not groq_client:
            return {"status": "failed", "error": "Groq client not configured"}
        
        # Extract data
        title = job_data.get('title', '').strip()
        description = job_data.get('description', '').strip()
        vision_analysis = job_data.get('vision_analysis', {})
        
        # STEP 1: Extract keywords from all sources
        base_keywords = f"{title} {description}".strip()
        vision_keywords = extract_vision_keywords(vision_analysis)
        
        # Combine keywords
        if base_keywords:
            all_keywords = base_keywords
            if vision_keywords:
                all_keywords = f"{base_keywords} {' '.join(vision_keywords)}"
        elif vision_keywords:
            all_keywords = ' '.join(vision_keywords)
        else:
            all_keywords = "item"
        
        logger.info(f"📋 Combined keywords: {all_keywords[:100]}...")
        
        # STEP 2: Get category using eBay Taxonomy API
        try:
            category_suggestions = ebay_api_instance.get_category_suggestions(all_keywords)
            if not category_suggestions:
                return {"status": "failed", "error": "eBay Taxonomy API failed to return category suggestions"}
            
            # Extract best category
            suggestions = category_suggestions.get('categorySuggestions', [])
            if not suggestions:
                return {"status": "failed", "error": "No category suggestions found"}
            
            best_category = suggestions[0]
            category_id = best_category['category']['categoryId']
            category_name = best_category['category']['categoryName']
            
            logger.info(f"✅ Taxonomy API category: {category_name} (ID: {category_id})")
            
        except Exception as e:
            logger.error(f"❌ Taxonomy API error: {e}")
            return {"status": "failed", "error": f"Category identification failed: {str(e)}"}
        
        # STEP 3: Get keyword suggestions using eBay Marketing API
        try:
            keyword_suggestions = ebay_api_instance.get_keyword_suggestions(all_keywords, category_id)
            
            optimized_keywords = []
            if keyword_suggestions and 'suggestedKeywords' in keyword_suggestions:
                suggested = keyword_suggestions['suggestedKeywords']
                optimized_keywords = [kw.get('keywordText', '') for kw in suggested[:10] if kw.get('keywordText')]
                logger.info(f"✅ Marketing API provided {len(optimized_keywords)} keyword suggestions")
            else:
                # Fallback to our own keywords if Marketing API fails
                optimized_keywords = [all_keywords]
                logger.warning("⚠️ Marketing API returned no suggestions, using base keywords")
            
        except Exception as e:
            logger.error(f"❌ Marketing API error: {e}")
            optimized_keywords = [all_keywords]
        
        # STEP 4: Analyze market trends using optimized keywords
        best_keyword = optimized_keywords[0] if optimized_keywords else all_keywords
        
        # Prepare item data for aspect extraction
        item_data = {
            "title": title,
            "description": description,
            "vision_keywords": vision_keywords
        }
        
        # Analyze market with eBay API
        try:
            market_analysis = ebay_api_instance.analyze_market_trends(
                keywords=best_keyword,
                category_id=category_id,
                item_data=item_data
            )
            
            if market_analysis.get('sample_size', 0) == 0:
                return {"status": "failed", "error": "No sold items found for analysis"}
            
        except Exception as e:
            logger.error(f"❌ Market analysis error: {e}")
            return {"status": "failed", "error": f"Market analysis failed: {str(e)}"}
        
        # STEP 5: Calculate profit potential
        profit_potential = 0.0
        if market_analysis['median_price'] > 0 and market_analysis['recommended_buy_below'] > 0:
            profit_potential = market_analysis['median_price'] - market_analysis['recommended_buy_below']
        
        # STEP 6: Generate AI insights with Groq
        ai_insights = {
            "listing_tips": [],
            "pricing_strategy": [],
            "market_timing": []
        }
        
        try:
            if groq_client:
                context = f"""
                Item Analysis:
                Title: {title or 'Not provided'}
                Description: {description or 'Not provided'}
                Category: {category_name}
                Market Analysis: {json.dumps({k: v for k, v in market_analysis.items() if k != 'sold_items_sample'})}
                Optimized Keywords: {', '.join(optimized_keywords[:5])}
                
                Provide 3-5 concise tips for:
                1. eBay listing optimization
                2. Pricing strategy
                3. Best timing to list
                """
                
                response = groq_client.chat.completions.create(
                    model=groq_model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert eBay reselling advisor. Provide concise, actionable tips."
                        },
                        {
                            "role": "user",
                            "content": context
                        }
                    ],
                    temperature=0.7,
                    max_tokens=300
                )
                
                ai_response = response.choices[0].message.content
                
                # Parse AI response
                lines = ai_response.split('\n')
                current_section = None
                
                for line in lines:
                    line = line.strip()
                    if 'listing' in line.lower() or 'optimization' in line.lower():
                        current_section = "listing_tips"
                    elif 'pricing' in line.lower() or 'strategy' in line.lower():
                        current_section = "pricing_strategy"
                    elif 'timing' in line.lower() or 'when' in line.lower():
                        current_section = "market_timing"
                    elif line and current_section and line.startswith(('-', '•', '*')):
                        tip = line.lstrip('-•* ').strip()
                        if tip and current_section in ai_insights:
                            ai_insights[current_section].append(tip)
                
                logger.info(f"🤖 Generated AI insights")
                
        except Exception as e:
            logger.warning(f"⚠️ AI insights generation failed: {e}")
        
        # STEP 7: Create comprehensive result
        result = {
            "message": f"Analysis complete - {market_analysis['confidence'].upper()} confidence",
            "items": [{
                "title": title or "Item from image",
                "description": description or "Uploaded image analysis",
                "price_range": market_analysis['price_range'],
                "lowest_price": market_analysis['lowest_price'],
                "highest_price": market_analysis['highest_price'],
                "average_price": market_analysis['average_price'],
                "median_price": market_analysis['median_price'],
                "resellability_rating": calculate_resellability(market_analysis),
                "suggested_cost": f"${market_analysis['recommended_buy_below']:.2f}",
                "market_insights": market_analysis['market_notes'],
                "profit_potential": f"${profit_potential:.2f}",
                "category_id": category_id,
                "category_name": category_name,
                "sample_size": market_analysis['sample_size'],
                "confidence": market_analysis['confidence'],
                "confidence_reason": market_analysis['confidence_reason'],
                "days_since_last_sale": market_analysis['days_since_last_sale'],
                "ebay_specific_tips": generate_ebay_tips(market_analysis),
                "ai_insights": ai_insights,
                "optimized_keywords": optimized_keywords[:10],
                "data_source": "eBay APIs + Vision Analysis",
                "apis_used": ["Taxonomy API", "Marketing API", "Browse API"]
            }],
            "processing_time": "25s",
            "apis_working": True,
            "market_analysis_summary": {
                "confidence": market_analysis['confidence'],
                "sample_size": market_analysis['sample_size'],
                "price_stability": market_analysis.get('price_stability', 'unknown')
            }
        }
        
        return {
            "status": "completed",
            "result": result
        }
        
    except Exception as e:
        logger.error(f"❌ Job processing error: {e}")
        return {"status": "failed", "error": str(e)}

def background_worker():
    """Background worker for job processing"""
    logger.info("🎯 Background worker started")
    
    while True:
        try:
            job_id = job_queue.get(timeout=30)
            update_activity()
            
            with job_lock:
                job_data = job_storage.get(job_id)
                if not job_data:
                    continue
                
                job_data['status'] = 'processing'
                job_data['started_at'] = datetime.now().isoformat()
                job_storage[job_id] = job_data
            
            logger.info(f"📄 Processing job {job_id}")
            
            future = job_executor.submit(process_image_job, job_data)
            try:
                result = future.result(timeout=25)
                
                with job_lock:
                    if result.get('status') == 'completed':
                        job_data['status'] = 'completed'
                        job_data['result'] = result['result']
                        logger.info(f"✅ Job {job_id} completed successfully")
                    else:
                        job_data['status'] = 'failed'
                        job_data['error'] = result.get('error', 'Unknown error')
                        logger.error(f"❌ Job {job_id} failed: {result.get('error')}")
                    
                    job_data['completed_at'] = datetime.now().isoformat()
                    job_storage[job_id] = job_data
                    
            except FutureTimeoutError:
                with job_lock:
                    job_data['status'] = 'failed'
                    job_data['error'] = 'Processing timeout (25s)'
                    job_data['completed_at'] = datetime.now().isoformat()
                    job_storage[job_id] = job_data
                logger.warning(f"⏱️ Job {job_id} timed out")
                
            job_queue.task_done()
            
        except queue.Empty:
            update_activity()
            continue
        except Exception as e:
            logger.error(f"Worker error: {e}")
            update_activity()
            time.sleep(5)

# ============= HEALTH & STATUS ENDPOINTS =============

@app.get("/health")
@app.get("/health/")
async def health_check():
    """Health check endpoint"""
    update_activity()
    
    # Check eBay API status
    ebay_status = "unknown"
    if ebay_api_instance:
        try:
            # Quick test of eBay API
            test_result = ebay_api_instance.verify_token()
            ebay_status = "connected" if test_result else "failed"
        except:
            ebay_status = "error"
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": app.version,
        "service": "resell-pro-api",
        "ebay_api": ebay_status,
        "groq_status": "ready" if groq_client else "not_configured",
        "job_queue_size": job_queue.qsize(),
        "active_jobs": len([j for j in job_storage.values() if j.get('status') == 'processing'])
    }

@app.get("/ping")
async def ping():
    """Simple ping endpoint"""
    update_activity()
    return {
        "status": "PONG",
        "timestamp": datetime.now().isoformat(),
        "ebay_ready": bool(get_ebay_token()),
        "version": "5.0.0",
        "apis_enabled": ["Taxonomy", "Marketing", "Browse"],
        "requires_auth": True
    }

@app.get("/")
async def root():
    """Root endpoint with system info"""
    update_activity()
    
    ebay_token = get_ebay_token()
    has_groq = groq_client is not None
    has_ebay_api = ebay_api_instance is not None
    
    status = "OPERATIONAL" if has_groq and has_ebay_api and ebay_token else "PARTIAL"
    
    return {
        "message": "AI Resell Pro API - Enhanced Edition",
        "status": status,
        "version": "5.0.0",
        "ebay_authentication": "✅ Connected" if ebay_token else "❌ Required",
        "groq_ai": "✅ Configured" if has_groq else "❌ Not Configured",
        "ebay_api": "✅ Initialized" if has_ebay_api else "❌ Failed",
        "features": [
            "✅ Image upload with vision analysis",
            "✅ eBay OAuth 2.0 authentication",
            "✅ eBay Taxonomy API (category suggestions)",
            "✅ eBay Marketing API (keyword optimization)",
            "✅ Market trend analysis (20-50+ comps target)",
            "✅ AI-powered insights with Groq",
            "✅ Job queue system (timeout prevention)",
            "✅ iOS Lift integration compatible"
        ],
        "endpoints": {
            "upload": "/upload_item/",
            "auth": "/ebay/oauth/start",
            "health": "/health",
            "docs": "/docs",
            "debug": "/debug/valuation/{keywords}"
        },
        "requirements": {
            "ebay_token": "Required (2-year OAuth)",
            "groq_key": "Required for AI insights",
            "image_size": "Max 8MB",
            "processing_time": "25-30 seconds"
        }
    }

# ============= DEBUG & TESTING ENDPOINTS =============

@app.get("/debug/ebay-search/{keywords}")
async def debug_ebay_search(keywords: str, category: Optional[str] = None):
    """Debug endpoint to test eBay search directly"""
    update_activity()
    
    try:
        if not ebay_api_instance:
            raise HTTPException(status_code=500, detail="eBay API not initialized")
        
        category_id = None
        if category == 'vehicles':
            category_id = '6001'
        elif category == 'pokemon':
            category_id = '183454'
        elif category == 'electronics':
            category_id = '58058'
        
        # Test search
        items = ebay_api_instance.search_sold_items(keywords, category_id=category_id, limit=10)
        
        return {
            "keywords": keywords,
            "category_id": category_id,
            "items_found": len(items),
            "items": items[:5],
            "sample_prices": [item['price'] for item in items[:10]]
        }
        
    except Exception as e:
        logger.error(f"Debug search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/debug/valuation/{keywords}")
async def debug_valuation(keywords: str, category: Optional[str] = None):
    """Debug endpoint to test valuation directly"""
    update_activity()
    
    try:
        if not ebay_api_instance:
            raise HTTPException(status_code=500, detail="eBay API not initialized")
        
        category_id = None
        if category == 'vehicles':
            category_id = '6001'
        elif category == 'pokemon':
            category_id = '183454'
        
        # Test market analysis
        analysis = ebay_api_instance.analyze_market_trends(keywords, category_id=category_id)
        
        return {
            "keywords": keywords,
            "category_id": category_id,
            "analysis": analysis,
            "data_quality": "high" if analysis['sample_size'] >= 20 else "medium" if analysis['sample_size'] >= 10 else "low"
        }
        
    except Exception as e:
        logger.error(f"Debug valuation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/debug/api-test")
async def debug_api_test():
    """Test all eBay APIs"""
    update_activity()
    
    try:
        if not ebay_api_instance:
            return {"status": "error", "message": "eBay API not initialized"}
        
        results = {}
        
        # Test 1: Token verification
        token_ok = ebay_api_instance.verify_token()
        results["token_verification"] = "✅ Valid" if token_ok else "❌ Invalid"
        
        # Test 2: Taxonomy API
        try:
            category_test = ebay_api_instance.get_category_suggestions("iphone 14")
            results["taxonomy_api"] = "✅ Working" if category_test else "❌ Failed"
        except:
            results["taxonomy_api"] = "❌ Error"
        
        # Test 3: Marketing API
        try:
            keyword_test = ebay_api_instance.get_keyword_suggestions("iphone", "9355")
            results["marketing_api"] = "✅ Working" if keyword_test else "⚠️ Limited"
        except:
            results["marketing_api"] = "❌ Error"
        
        # Test 4: Browse API
        try:
            search_test = ebay_api_instance.search_sold_items("test", limit=1)
            results["browse_api"] = f"✅ Working ({len(search_test)} items)" if search_test else "❌ Failed"
        except:
            results["browse_api"] = "❌ Error"
        
        return {
            "status": "complete",
            "timestamp": datetime.now().isoformat(),
            "results": results,
            "recommendation": "All APIs should show ✅ for optimal performance"
        }
        
    except Exception as e:
        logger.error(f"API test error: {e}")
        return {"status": "error", "message": str(e)}

# ============= SERVER LIFECYCLE =============

@app.on_event("startup")
async def startup_event():
    """Startup event - initialize background worker"""
    # Start background job processor
    threading.Thread(target=background_worker, daemon=True, name="JobWorker").start()
    logger.info("🚀 Server started with enhanced eBay API integration")
    logger.info("📊 Features enabled: Taxonomy API, Marketing API, Vision Analysis")
    logger.info("⏱️ Job queue system active")

@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event - clean up resources"""
    job_executor.shutdown(wait=False)
    logger.info("🛑 Server shutdown - resources cleaned up")

# ============= MAIN ENTRY POINT =============

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