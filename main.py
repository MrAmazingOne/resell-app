# JOB QUEUE + POLLING SYSTEM - FULL ACCURACY VERSION
# Maintains full analysis accuracy while working within Render's 30s timeout

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from groq import Groq
import os
import json
from typing import Optional, List, Dict, Any
import logging
import base64
import requests
import re
from datetime import datetime
from ebay_integration import ebay_api
from dotenv import load_dotenv
import uuid
import threading
import queue
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import asyncio

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Resell Pro API - Full Accuracy", 
    version="3.2.0",
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

def update_activity():
    """Update last activity timestamp"""
    with activity_lock:
        global last_activity
        last_activity = time.time()

# FULL DETAILED MARKET ANALYSIS PROMPT (MAINTAINING ACCURACY)
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

IMPORTANT: Return ONLY valid JSON, no additional text or explanations.
"""

def detect_category(title: str, description: str) -> str:
    """Accurate category detection"""
    title_lower = title.lower()
    description_lower = description.lower()
    
    category_keywords = {
        "electronics": ["electronic", "computer", "phone", "tablet", "camera", "laptop", "charger", "battery", "screen", "iphone", "samsung", "android"],
        "clothing": ["shirt", "pants", "dress", "jacket", "shoe", "sweater", "fabric", "cotton", "wool", "brand", "nike", "adidas", "levi"],
        "furniture": ["chair", "table", "desk", "cabinet", "sofa", "couch", "wood", "furniture", "drawer", "shelf", "bed"],
        "collectibles": ["collectible", "rare", "vintage", "antique", "edition", "limited", "signed", "autograph", "memorabilia"],
        "books": ["book", "novel", "author", "page", "edition", "publish", "hardcover", "paperback", "literature"],
        "toys": ["toy", "game", "play", "action figure", "doll", "puzzle", "lego", "model kit", "collectible figure"]
    }
    
    scores = {category: 0 for category in category_keywords}
    scores["unknown"] = 0
    
    for category, keywords in category_keywords.items():
        for keyword in keywords:
            if keyword in title_lower or keyword in description_lower:
                scores[category] += 1
    
    return max(scores.items(), key=lambda x: x[1])[0]

def enhance_with_ebay_data(item_data: Dict) -> Dict:
    """Full eBay market data enhancement with proper error handling"""
    try:
        # Create comprehensive search keywords
        keywords = []
        
        # Prioritize brand and model
        if item_data.get('brand'):
            keywords.append(item_data['brand'])
        if item_data.get('model'):
            keywords.append(item_data['model'])
        
        # Add key features
        if item_data.get('key_features'):
            keywords.extend(item_data['key_features'][:2])
        
        # Fallback to title analysis
        if not keywords:
            title_words = item_data.get('title', '').split()
            # Extract likely product words (not generic)
            product_words = [word for word in title_words[:4] if len(word) > 3 and word.lower() not in ['the', 'and', 'with', 'for']]
            keywords.extend(product_words)
        
        search_query = ' '.join(keywords[:4])  # Use up to 4 keywords
        
        if search_query.strip():
            logger.info(f"🔍 Searching eBay for: {search_query}")
            
            # Get comprehensive market analysis
            completed_items = ebay_api.search_completed_items(search_query, max_results=20)
            current_items = ebay_api.get_current_listings(search_query, max_results=10)
            
            if completed_items:
                # Calculate detailed statistics
                sold_prices = [item['price'] for item in completed_items if item['price'] > 0]
                
                if sold_prices:
                    avg_price = sum(sold_prices) / len(sold_prices)
                    min_price = min(sold_prices)
                    max_price = max(sold_prices)
                    
                    # Calculate price quartiles for better insights
                    sorted_prices = sorted(sold_prices)
                    median_price = sorted_prices[len(sorted_prices) // 2]
                    
                    # Market saturation analysis
                    price_std = (sum((p - avg_price) ** 2 for p in sold_prices) / len(sold_prices)) ** 0.5
                    price_volatility = "high" if price_std > avg_price * 0.3 else "medium" if price_std > avg_price * 0.15 else "low"
                    
                    # Update item data with comprehensive market analysis
                    item_data['price_range'] = f"${min_price:.2f} - ${max_price:.2f}"
                    item_data['suggested_cost'] = f"${median_price * 0.85:.2f}"
                    
                    # Enhanced profit calculation
                    ebay_fees = median_price * 0.13  # 13% eBay fees
                    shipping_cost = 12.00  # Average shipping with packaging
                    estimated_net = median_price - ebay_fees - shipping_cost
                    suggested_purchase = median_price * 0.85
                    profit = estimated_net - suggested_purchase
                    
                    item_data['profit_potential'] = f"${profit:.2f} profit (after all fees)" if profit > 0 else f"${profit:.2f} loss (not recommended)"
                    
                    # Enhanced market insights
                    sell_through_rate = (len(completed_items) / (len(completed_items) + len(current_items))) * 100 if (len(completed_items) + len(current_items)) > 0 else 50
                    
                    insights = []
                    insights.append(f"Based on {len(completed_items)} recent sold listings")
                    insights.append(f"Median sold price: ${median_price:.2f}")
                    insights.append(f"Price volatility: {price_volatility}")
                    insights.append(f"Estimated sell-through: {sell_through_rate:.1f}%")
                    
                    if len(current_items) > 0:
                        current_avg = sum(item['price'] for item in current_items) / len(current_items)
                        insights.append(f"Current listings average: ${current_avg:.2f}")
                    
                    item_data['market_insights'] = ". ".join(insights) + ". " + item_data.get('market_insights', '')
                    
                    # Enhanced eBay tips
                    item_data['ebay_specific_tips'] = [
                        "Use all 12 photo slots with multiple angles and close-ups",
                        "Include measurements, weight, and detailed condition report",
                        f"Best listing time: Sunday-Wednesday evenings (peak traffic)",
                        "Consider 'Buy It Now' with Best Offer for price flexibility",
                        f"Competition level: {'High' if len(current_items) > 15 else 'Medium' if len(current_items) > 5 else 'Low'}",
                        f"Market liquidity: {'Good' if sell_through_rate > 60 else 'Fair' if sell_through_rate > 40 else 'Slow'}"
                    ]
                    
                    logger.info(f"✅ eBay enhancement successful: {len(completed_items)} sold items analyzed")
                else:
                    logger.warning(f"⚠️ eBay search found items but no price data")
            else:
                logger.warning(f"⚠️ No eBay results for: {search_query}")
        else:
            logger.warning("⚠️ No searchable keywords extracted")
            
        return item_data
        
    except Exception as e:
        logger.error(f"❌ eBay data enhancement failed: {e}")
        # Keep original data if eBay fails
        return item_data

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
        self.brand = data.get("brand", "")
        self.model = data.get("model", "")
        self.year = data.get("year", "")
        self.condition = data.get("condition", "")
        self.confidence = data.get("confidence", 0.5)
        self.analysis_depth = data.get("analysis_depth", "standard")
        self.key_features = data.get("key_features", [])
        self.comparable_items = data.get("comparable_items", "")
        
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
            "analysis_depth": self.analysis_depth,
            "key_features": self.key_features,
            "comparable_items": self.comparable_items
        }

def parse_json_response(response_text: str) -> List[Dict]:
    """Robust JSON parsing maintaining all details"""
    try:
        json_text = response_text.strip()
        
        # Clean up JSON response while preserving all content
        if "```json" in json_text:
            json_start = json_text.find("```json") + 7
            json_end = json_text.find("```", json_start)
            json_text = json_text[json_start:json_end].strip()
        elif "```" in json_text:
            json_start = json_text.find("```") + 3
            json_end = json_text.rfind("```")
            json_text = json_text[json_start:json_end].strip()
        
        # Extract JSON object
        json_match = re.search(r'\{.*\}', json_text, re.DOTALL)
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

def call_groq_api(prompt: str, image_base64: str = None, mime_type: str = None) -> str:
    """Full-featured Groq API call with timeout protection"""
    if not groq_client:
        raise Exception("Groq client not configured")
    
    messages = []
    
    if image_base64 and mime_type:
        # Clean base64 string
        if image_base64.startswith('data:'):
            parts = image_base64.split(',', 1)
            if len(parts) > 1:
                image_base64 = parts[1]
        
        image_content = {
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime_type};base64,{image_base64}"
            }
        }
        messages.append({
            "role": "user",
            "content": [image_content, {"type": "text", "text": prompt}]
        })
    else:
        messages.append({
            "role": "user",
            "content": prompt
        })
    
    try:
        logger.info(f"📤 Calling Groq API with {len(prompt)} chars prompt")
        
        response = groq_client.chat.completions.create(
            model=groq_model,
            messages=messages,
            temperature=0.1,  # Low temperature for consistent results
            max_tokens=4000,  # Full token allowance for detailed analysis
            top_p=0.95,
            stream=False,
            timeout=15.0  # Reasonable timeout
        )
        
        if response.choices and len(response.choices) > 0:
            content = response.choices[0].message.content
            logger.info(f"📥 Groq API response received ({len(content)} chars)")
            return content
        else:
            logger.error("Empty response from Groq API")
            raise Exception("Empty response from Groq API")
        
    except Exception as e:
        logger.error(f"Groq API call failed: {e}")
        raise Exception(f"Groq API error: {str(e)[:100]}")

def process_image_full_accuracy(job_data: Dict) -> Dict:
    """Full accuracy processing with all features"""
    try:
        if not groq_client:
            return {"status": "failed", "error": "Groq client not configured"}
        
        image_base64 = job_data['image_base64']
        mime_type = job_data['mime_type']
        
        # Build comprehensive prompt
        prompt = market_analysis_prompt
        if job_data.get('title'):
            prompt += f"\nUser-provided title: {job_data['title']}"
        if job_data.get('description'):
            prompt += f"\nUser-provided description: {job_data['description']}"
        
        accuracy_mode = job_data.get('accuracy_mode', 'maximum')
        if accuracy_mode == 'maximum':
            prompt += "\n\nANALYSIS MODE: MAXIMUM ACCURACY - Provide the most detailed analysis possible."
        
        logger.info(f"🔬 Starting {accuracy_mode} accuracy analysis...")
        
        # Call Groq API for detailed analysis
        response_text = call_groq_api(prompt, image_base64, mime_type)
        logger.info("✅ AI analysis completed, parsing response...")
        
        ai_response = parse_json_response(response_text)
        logger.info(f"📊 Parsed {len(ai_response)} items from response")
        
        enhanced_items = []
        for item_data in ai_response:
            if isinstance(item_data, dict):
                # Enhance with comprehensive eBay market data
                item_data = enhance_with_ebay_data(item_data)
                
                # Detect category
                detected_category = detect_category(item_data.get("title", ""), item_data.get("description", ""))
                item_data["category"] = detected_category
                
                enhanced_items.append(EnhancedAppItem(item_data).to_dict())
            else:
                logger.warning(f"Skipping non-dictionary item: {item_data}")
        
        if not enhanced_items:
            # Create comprehensive fallback response
            enhanced_items.append(EnhancedAppItem({
                "title": "Item Analysis - Need Clearer Image",
                "description": "Unable to extract detailed information. Please ensure:\n1. Good lighting on the item\n2. Clear focus on text/serial numbers\n3. Multiple angles if possible\n4. Avoid glare and shadows",
                "price_range": "$0-0",
                "resellability_rating": 5,
                "suggested_cost": "$0",
                "market_insights": "Image quality insufficient for accurate analysis. Try retaking with better lighting and focus.",
                "authenticity_checks": "Cannot verify authenticity without clear details. Look for serial numbers, brand markings, and quality indicators.",
                "profit_potential": "Unknown - requires clearer identification",
                "category": "unknown",
                "ebay_specific_tips": [
                    "Retake photo with natural daylight",
                    "Include all sides and any labels",
                    "Add a size reference (coin, ruler)",
                    "Clean the item before photographing"
                ],
                "brand": "",
                "model": "",
                "year": "",
                "condition": "",
                "confidence": 0.2,
                "analysis_depth": "limited",
                "key_features": [],
                "comparable_items": ""
            }).to_dict())
        
        logger.info(f"✅ Processing complete: {len(enhanced_items)} items with full analysis")
        
        return {
            "status": "completed",
            "result": {
                "message": f"Comprehensive analysis completed with {len(enhanced_items)} items",
                "items": enhanced_items,
                "processing_time": "20-30s",
                "analysis_stages": 2,
                "confidence_level": "full_accuracy",
                "analysis_timestamp": datetime.now().isoformat(),
                "model_used": groq_model,
                "accuracy_mode": accuracy_mode
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Full accuracy processing failed: {e}")
        return {"status": "failed", "error": str(e)[:200]}

def background_worker():
    """Background worker with full accuracy processing"""
    logger.info("🎯 Background worker started - FULL ACCURACY MODE")
    
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
            
            logger.info(f"🔄 Processing job {job_id} with full accuracy...")
            
            # Process with timeout (25 seconds for Render's 30s limit)
            future = job_executor.submit(process_image_full_accuracy, job_data)
            try:
                result = future.result(timeout=25)  # 25 second timeout
                
                with job_lock:
                    if result.get('status') == 'completed':
                        job_data['status'] = 'completed'
                        job_data['result'] = result['result']
                        logger.info(f"✅ Job {job_id} completed successfully")
                    else:
                        job_data['status'] = 'failed'
                        job_data['error'] = result.get('error', 'Unknown error')
                        logger.error(f"❌ Job {job_id} failed: {job_data['error']}")
                    
                    job_data['completed_at'] = datetime.now().isoformat()
                    job_storage[job_id] = job_data
                    
            except FutureTimeoutError:
                with job_lock:
                    job_data['status'] = 'failed'
                    job_data['error'] = 'Processing timeout (25s) - Try with standard accuracy mode'
                    job_data['completed_at'] = datetime.now().isoformat()
                    job_storage[job_id] = job_data
                logger.warning(f"⏱️ Job {job_id} timed out after 25 seconds")
                
            job_queue.task_done()
            
        except queue.Empty:
            update_activity()
            continue
        except Exception as e:
            logger.error(f"🎯 Background worker error: {e}")
            update_activity()
            time.sleep(5)

# Start background worker on startup
@app.on_event("startup")
async def startup_event():
    # Start background worker
    threading.Thread(target=background_worker, daemon=True, name="JobWorker-FullAccuracy").start()
    
    # Start keep-alive thread
    def keep_alive_loop():
        while True:
            time.sleep(25)  # Ping every 25 seconds
            try:
                update_activity()
                # Self-ping to keep alive
                requests.get(f"http://localhost:{os.getenv('PORT', 8000)}/ping", timeout=5)
            except:
                pass
    
    threading.Thread(target=keep_alive_loop, daemon=True, name="KeepAlive").start()
    
    logger.info("🚀 Server started with FULL ACCURACY processing and keep-alive")

@app.on_event("shutdown")
async def shutdown_event():
    job_executor.shutdown(wait=False)
    logger.info("🛑 Server shutting down")

# MAIN ENDPOINTS
@app.post("/upload_item/")
async def create_upload_file(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    accuracy_mode: str = Form("maximum")
):
    update_activity()
    
    try:
        # Read image with reasonable limit
        image_bytes = await file.read()
        if len(image_bytes) > 8 * 1024 * 1024:  # 8MB limit
            raise HTTPException(status_code=400, detail="Image too large (max 8MB)")
        
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        job_id = str(uuid.uuid4())
        
        with job_lock:
            # Clean old jobs (older than 2 hours)
            current_time = datetime.now()
            for old_id, old_job in list(job_storage.items()):
                try:
                    created = datetime.fromisoformat(old_job.get('created_at', ''))
                    if (current_time - created).seconds > 7200:  # 2 hours
                        del job_storage[old_id]
                except:
                    pass
            
            job_storage[job_id] = {
                'image_base64': image_base64,
                'mime_type': file.content_type,
                'title': title,
                'description': description,
                'accuracy_mode': accuracy_mode,
                'status': 'queued',
                'created_at': datetime.now().isoformat()
            }
        
        job_queue.put(job_id)
        logger.info(f"📤 Job {job_id} queued (accuracy: {accuracy_mode})")
        
        return {
            "message": "Analysis queued with FULL ACCURACY mode",
            "job_id": job_id,
            "status": "queued",
            "accuracy_mode": accuracy_mode,
            "estimated_time": "25-30 seconds for maximum accuracy",
            "check_status_url": f"/job/{job_id}/status",
            "note": "Processing with comprehensive eBay market data integration"
        }
            
    except Exception as e:
        logger.error(f"❌ Upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)[:200]}")

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
        "accuracy_mode": job_data.get('accuracy_mode', 'maximum')
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

@app.get("/health")
async def health_check():
    update_activity()
    
    with activity_lock:
        time_since = time.time() - last_activity
    
    groq_status = "✅ Ready" if groq_client else "❌ Not configured"
    ebay_status = "✅ Configured" if os.getenv('EBAY_APP_ID') else "⚠️ Not configured"
    
    return {
        "status": "✅ HEALTHY",
        "timestamp": datetime.now().isoformat(),
        "time_since_last_activity": f"{int(time_since)}s",
        "jobs_queued": job_queue.qsize(),
        "jobs_stored": len(job_storage),
        "groq_status": groq_status,
        "ebay_status": ebay_status,
        "processing_mode": "FULL_ACCURACY",
        "features": [
            "Comprehensive Groq AI analysis",
            "Detailed eBay market integration",
            "Job queue for Render timeout protection",
            "Keep-alive system for instance stability"
        ]
    }

@app.get("/ping")
async def ping():
    update_activity()
    return {
        "status": "✅ PONG",
        "timestamp": datetime.now().isoformat(),
        "message": "Server is awake and processing with full accuracy",
        "keep_alive": "active"
    }

@app.get("/")
async def root():
    update_activity()
    return {
        "message": "🎯 AI Resell Pro API - FULL ACCURACY EDITION",
        "status": "🚀 OPERATIONAL",
        "version": "3.2.0",
        "processing_capabilities": [
            "Maximum accuracy item identification",
            "Comprehensive eBay market analysis",
            "Detailed profit calculations",
            "Condition and authenticity assessment"
        ],
        "endpoints": {
            "upload": "POST /upload_item (with accuracy_mode: maximum/standard)",
            "job_status": "GET /job/{job_id}/status",
            "health": "GET /health",
            "ping": "GET /ping (keep-alive)"
        },
        "notes": [
            "Uses job queue system to avoid Render's 30s timeout",
            "Maintains full analysis accuracy while staying reliable",
            "Includes keep-alive to prevent instance spin-down"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8000))
    logger.info(f"🚀 Starting FULL ACCURACY server on port {port}")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        workers=1,  # Single worker for job queue consistency
        timeout_keep_alive=30,
        log_level="info"
    )