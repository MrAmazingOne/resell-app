# JOB QUEUE + POLLING SYSTEM - SOLD ITEMS ONLY
# Enhanced with eBay Taxonomy API + Optimized Search Accuracy

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
from ebay_integration import ebay_api
from dotenv import load_dotenv
import threading
import queue
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

# Image analysis imports (KEPT - essential for Lift)
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
    version="4.8.0",
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

# In-memory job queue
job_queue = queue.Queue()
job_storage = {}
job_lock = threading.Lock()
job_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="JobWorker")

# Activity tracking
last_activity = time.time()
activity_lock = threading.Lock()

# eBay Token Storage
EBAY_AUTH_TOKEN = None
EBAY_TOKEN_LOCK = threading.Lock()

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
            "timestamp": datetime.now().isoformat(),
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
    """Analyze image to extract keywords for search"""
    try:
        # Convert to PIL Image
        image = Image.open(io.BytesIO(image_data))
        
        # Basic analysis
        analysis = {
            "width": image.width,
            "height": image.height,
            "format": image.format,
            "detected_objects": [],
            "suggested_keywords": []
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
                
                if 1.2 <= aspect_ratio <= 2.5:  # Typical vehicle aspect ratios
                    analysis["likely_vehicle"] = True
                    analysis["likely_full_vehicle"] = True if aspect_ratio > 1.5 else False
                    
        except Exception as e:
            logger.debug(f"CV2 analysis skipped: {e}")
        
        return analysis
    except Exception as e:
        logger.error(f"Vision analysis error: {e}")
        return {"error": str(e)}

# ============= IMAGE UPLOAD ENDPOINT =============

@app.post("/upload_item/")
async def create_upload_file(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    description: Optional[str] = Form(None)
):
    update_activity()
    
    try:
        ebay_token = get_ebay_token()
        if not ebay_token:
            raise HTTPException(
                status_code=400, 
                detail="eBay authentication required"
            )
        
        image_bytes = await file.read()
        if len(image_bytes) > 8 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Image too large (max 8MB)")
        
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        job_id = str(uuid.uuid4())
        
        # Vision analysis for keyword extraction
        vision_analysis = analyze_image_with_vision(image_bytes)
        
        with job_lock:
            current_time = datetime.now()
            for old_id, old_job in list(job_storage.items()):
                try:
                    created = datetime.fromisoformat(old_job.get('created_at', ''))
                    if (current_time - created).seconds > 7200:
                        del job_storage[old_id]
                except:
                    pass
            
            job_storage[job_id] = {
                'image_base64': image_base64,
                'image_bytes': image_bytes,  # Keep bytes for processing
                'mime_type': file.content_type,
                'title': title,
                'description': description,
                'vision_analysis': vision_analysis,
                'status': 'queued',
                'created_at': datetime.now().isoformat(),
                'requires_ebay_auth': not bool(ebay_token)
            }
        
        job_queue.put(job_id)
        logger.info(f"📤 Job {job_id} queued")
        
        return {
            "message": "Analysis queued with optimized search",
            "job_id": job_id,
            "status": "queued",
            "estimated_time": "25-30 seconds",
            "check_status_url": f"/job/{job_id}/status",
            "ebay_auth_status": "connected" if ebay_token else "required"
        }
            
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"❌ Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)[:200])

@app.get("/job/{job_id}/status")
async def get_job_status(job_id: str):
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
    """Process image with Groq AI + eBay market analysis"""
    try:
        if not groq_client:
            return {"status": "failed", "error": "Groq client not configured"}
        
        # Extract keywords from user input + vision
        title = job_data.get('title', '')
        description = job_data.get('description', '')
        vision_analysis = job_data.get('vision_analysis', {})
        
        # Combine all sources for keywords
        combined_keywords = f"{title} {description}".strip()
        if combined_keywords == "":
            # Use vision analysis to suggest keywords
            if vision_analysis.get('likely_vehicle'):
                combined_keywords = "vehicle car truck"
        
        # Get category suggestions
        category_id = ebay_api.get_category_suggestions(combined_keywords)
        
        # Prepare item data for aspect extraction
        item_data = {
            "title": title,
            "description": description
        }
        
        # Analyze market trends with optimized search
        analysis = ebay_api.analyze_market_trends(
            keywords=combined_keywords,
            category_id=category_id,
            item_data=item_data
        )
        
        # Calculate profit potential
        profit_potential = 0.0
        if analysis['median_price'] > 0 and analysis['recommended_buy_below'] > 0:
            profit_potential = analysis['median_price'] - analysis['recommended_buy_below']
        
        # Create result with iOS-compatible format
        result = {
            "message": f"Analysis complete - {analysis['confidence'].upper()} confidence",
            "items": [{
                "title": title or "Identified Item",
                "description": description or "No description provided",
                "price_range": analysis['price_range'],
                "lowest_price": analysis['lowest_price'],
                "highest_price": analysis['highest_price'],
                "average_price": analysis['average_price'],
                "median_price": analysis['median_price'],
                "resellability_rating": self._calculate_resellability(analysis),
                "suggested_cost": f"${analysis['recommended_buy_below']:.2f}",
                "market_insights": analysis['market_notes'],
                "profit_potential": f"${profit_potential:.2f}",
                "category": category_id,
                "sample_size": analysis['sample_size'],
                "confidence": analysis['confidence'],
                "confidence_reason": analysis['confidence_reason'],
                "days_since_last_sale": analysis['days_since_last_sale'],
                "ebay_specific_tips": self._generate_ebay_tips(analysis),
                "data_source": "eBay Sold Listings",
                "aspects_used": analysis.get('aspects_used', [])
            }],
            "processing_time": "25s",
            "ebay_data_used": True,
            "market_analysis": analysis
        }
        
        return {
            "status": "completed",
            "result": result
        }
        
    except Exception as e:
        logger.error(f"Job processing error: {e}")
        return {"status": "failed", "error": str(e)}

def _calculate_resellability(self, analysis: Dict) -> int:
    """Calculate resellability rating 1-10"""
    rating = 5  # Base
    
    # Adjust based on sample size
    sample_size = analysis['sample_size']
    if sample_size >= 50:
        rating += 3
    elif sample_size >= 20:
        rating += 2
    elif sample_size >= 10:
        rating += 1
    elif sample_size < 5:
        rating -= 2
    
    # Adjust based on confidence
    confidence = analysis['confidence']
    if confidence == 'high':
        rating += 2
    elif confidence == 'good':
        rating += 1
    elif confidence == 'very low':
        rating -= 2
    
    # Adjust based on price stability
    if analysis['price_stability'] == 'high':
        rating += 1
    
    # Ensure rating stays within 1-10
    return max(1, min(10, rating))

def _generate_ebay_tips(self, analysis: Dict) -> List[str]:
    """Generate eBay-specific tips based on analysis"""
    tips = []
    
    sample_size = analysis['sample_size']
    confidence = analysis['confidence']
    
    if sample_size >= 20:
        tips.append("Strong market data - list with confidence")
    elif sample_size >= 10:
        tips.append("Moderate data - consider checking similar items")
    else:
        tips.append("Limited data - research similar items before listing")
    
    if confidence in ['high', 'good']:
        tips.append("Price competitively based on recent sold data")
    
    if analysis['days_since_last_sale'] <= 7:
        tips.append("Very active market - list now for best results")
    
    tips.append("Use clear photos and detailed description")
    tips.append("Consider free shipping to increase visibility")
    
    return tips

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
                        logger.info(f"✅ Job {job_id} completed")
                    else:
                        job_data['status'] = 'failed'
                        job_data['error'] = result.get('error', 'Unknown error')
                        logger.error(f"❌ Job {job_id} failed")
                    
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

@app.on_event("startup")
async def startup_event():
    threading.Thread(target=background_worker, daemon=True, name="JobWorker").start()
    logger.info("🚀 Server started with job queue system")

@app.on_event("shutdown")
async def shutdown_event():
    job_executor.shutdown(wait=False)
    logger.info("🛑 Server shutdown")

# ============= HEALTH & STATUS ENDPOINTS =============

@app.get("/health")
@app.get("/health/")
async def health_check():
    update_activity()
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": app.version,
        "service": "resell-pro-api",
        "ebay_auth": "connected" if get_ebay_token() else "required",
        "groq_status": "ready" if groq_client else "not_configured"
    }

@app.get("/ping")
async def ping():
    update_activity()
    return {
        "status": "PONG",
        "timestamp": datetime.now().isoformat(),
        "ebay_ready": bool(get_ebay_token()),
        "version": "4.8.0"
    }

@app.get("/")
async def root():
    update_activity()
    ebay_token = get_ebay_token()
    
    return {
        "message": "AI Resell Pro API - v4.8 (Full System + Optimized Search)",
        "status": "OPERATIONAL" if groq_client and ebay_token else "PARTIAL",
        "ebay_authentication": "✅ Connected" if ebay_token else "❌ Required",
        "features": [
            "✅ Image upload with vision analysis",
            "✅ Job queue system (timeout prevention)",
            "✅ Multi-item processing",
            "✅ Optimized eBay search (20-50+ comps target)",
            "✅ Core aspect filtering (Year/Make/Model)",
            "✅ Intelligent sample size handling",
            "✅ Long-term OAuth (2 years)"
        ],
        "upload_endpoint": "/upload_item/",
        "auth_endpoint": "/ebay/oauth/start",
        "docs": "/docs"
    }

# ============= DEBUG ENDPOINTS =============

@app.get("/debug/ebay-search/{keywords}")
async def debug_ebay_search(keywords: str, category: Optional[str] = None):
    """Debug endpoint to test eBay search directly"""
    update_activity()
    
    try:
        category_id = None
        if category == 'vehicles':
            category_id = '6001'
        elif category == 'pokemon':
            category_id = '183454'
        
        items = ebay_api.search_sold_items(keywords, category_id=category_id, limit=10)
        
        return {
            "keywords": keywords,
            "category_id": category_id,
            "items_found": len(items),
            "items": items[:5],  # Return first 5 for debugging
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
        category_id = None
        if category == 'vehicles':
            category_id = '6001'
        elif category == 'pokemon':
            category_id = '183454'
        
        analysis = ebay_api.analyze_market_trends(keywords, category_id=category_id)
        
        return {
            "keywords": keywords,
            "category_id": category_id,
            "analysis": analysis
        }
        
    except Exception as e:
        logger.error(f"Debug valuation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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