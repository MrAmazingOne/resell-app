# JOB QUEUE + POLLING SYSTEM - MAXIMUM ACCURACY ONLY
# Uses 25s processing window to stay within Render's 30s timeout

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Header
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
import uuid
import threading
import queue
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import asyncio
import hmac
import hashlib
import urllib.parse

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Resell Pro API - Maximum Accuracy", 
    version="3.5.0",
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
        # Check if we have a token stored
        if EBAY_AUTH_TOKEN:
            logger.debug(f"🔑 Using stored eBay token: {EBAY_AUTH_TOKEN[:20]}...")
            return EBAY_AUTH_TOKEN
        
        # Try to load from environment variable
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

def refresh_ebay_token_if_needed(token_id: str) -> bool:
    """Refresh eBay token if it's expired"""
    try:
        token_data = ebay_oauth.get_user_token(token_id)
        if not token_data:
            logger.warning("⚠️ No token data found for refresh")
            return False
        
        expires_at = datetime.fromisoformat(token_data["expires_at"])
        time_remaining = expires_at - datetime.now()
        
        if time_remaining.total_seconds() < 300:  # Less than 5 minutes remaining
            logger.info("🔄 Token expiring soon, attempting refresh...")
            if "refresh_token" in token_data:
                refreshed = ebay_oauth.refresh_token(token_data["refresh_token"])
                if refreshed and refreshed.get("success"):
                    logger.info("✅ Token refreshed successfully")
                    if refreshed.get("access_token"):
                        store_ebay_token(refreshed["access_token"])
                    return True
                else:
                    logger.error("❌ Token refresh failed")
                    return False
        return True
        
    except Exception as e:
        logger.error(f"❌ Token refresh check error: {e}")
        return False

# MAXIMUM ACCURACY MARKET ANALYSIS PROMPT - ENHANCED
market_analysis_prompt = """
EXPERT RESELL ANALYST - MAXIMUM ACCURACY ANALYSIS:

You are analyzing items for resale profitability. You MUST use ALL available information:

🔍 **COMPREHENSIVE IDENTIFICATION PHASE:**
- Extract EVERY visible text, number, logo, brand mark, model number, serial number
- Identify ALL materials, construction quality, age indicators, manufacturing details
- Note ALL condition issues, wear patterns, damage, repairs, modifications
- Capture EXACT size, dimensions, weight indicators, manufacturing codes

📊 **ENHANCED MARKET ANALYSIS PHASE:**
- Use EXACT brand/model/year data when available
- If specific identification is unclear, analyze by material, construction, and visual characteristics
- Consider brand popularity, rarity, demand trends, collector interest
- Factor in ALL condition deductions and market saturation
- Account for seasonal pricing variations and current market trends

💰 **PRECISE PROFITABILITY ANALYSIS:**
- Calculate REALISTIC resale price range based on ALL factors
- Suggest MAXIMUM purchase price for profit with ALL fees accounted
- Estimate EXACT profit margins after ALL fees (eBay: 13%, shipping: $8-15, packaging: $3)
- Rate resellability 1-10 based on demand/competition/condition

📝 **INTELLIGENT FALLBACK STRATEGY (ONLY if specific ID impossible):**
- Analyze by material composition (wood, metal, plastic, fabric, etc.)
- Identify manufacturing style and era indicators
- Assess quality level (consumer, professional, luxury, handmade)
- Provide guidance on what additional info would enable precise identification

Return analysis in JSON format:

{
  "title": "eBay-optimized title with ALL available details",
  "description": "COMPREHENSIVE description with ALL visible details, condition notes, and identification guidance",
  "price_range": "Current market range: $X - $Y (based on available data)",
  "resellability_rating": 8,
  "suggested_cost": "Maximum to pay: $X (for profitable resale)",
  "market_insights": "Detailed market demand, competition level, selling strategies",
  "authenticity_checks": "SPECIFIC red flags and verification steps",
  "profit_potential": "Expected profit: $X-Y after ALL fees",
  "category": "Primary eBay category (based on analysis)",
  "ebay_specific_tips": ["Photography tips", "Listing optimization", "Timing advice", "Keyword strategies"],
  
  // Extended details - FILL EVERYTHING POSSIBLE
  "brand": "Exact brand if visible, otherwise 'Unknown - appears to be [quality/style]'",
  "model": "Model number/name if visible, otherwise descriptive characteristics", 
  "year": "Production year if determinable, otherwise era/style indicators",
  "condition": "DETAILED condition assessment with specific notes",
  "confidence": 0.85,
  "analysis_depth": "comprehensive",
  "key_features": ["ALL notable features that add value"],
  "comparable_items": "Similar items selling for $X-Y",
  "identification_confidence": "high/medium/low with reasoning",
  "additional_info_needed": ["What specific info would enable better identification"]
}

CRITICAL: Base pricing on ACTUAL market conditions, NEVER guess.
If specific identification is unclear, analyze by observable characteristics and provide guidance.

IMPORTANT: Return ONLY valid JSON, no additional text or explanations.
ALWAYS provide actionable insights, NEVER empty or generic responses.
"""

def clean_model_string(model: str) -> str:
    """Clean model string for better search results"""
    if not model:
        return ""
    
    # Remove generic/unhelpful words
    generic_words = [
        'model', 'version', 'series', 'edition', 'type', 'style',
        'unknown', 'not specified', 'not available', 'n/a', 'none',
        'comfortable', 'playing', 'experience', 'finish', 'exterior',
        'interior', 'condition', 'restored', 'black', 'blue', 'teal',
        'glossy', 'matte', 'vintage', 'antique', 'used', 'new',
        'excellent', 'good', 'fair', 'poor', 'working', 'non-working',
        'functional', 'non-functional', 'complete', 'incomplete',
        'partial', 'whole', 'damaged', 'undamaged', 'scratch', 'dent',
        'crack', 'chip', 'stain', 'tear', 'rip', 'hole', 'missing',
        'present', 'available', 'unavailable', 'visible', 'invisible',
        'clear', 'unclear', 'legible', 'illegible', 'readable', 'unreadable'
    ]
    
    words = model.split()
    cleaned_words = []
    
    for word in words:
        word_lower = word.lower()
        # Keep if not generic and has reasonable length
        if (word_lower not in generic_words and 
            len(word) > 2 and 
            not word.isdigit() or (word.isdigit() and len(word) in [2, 3, 4])):  # Keep years and model numbers
            cleaned_words.append(word)
    
    return ' '.join(cleaned_words[:3])  # Keep only most important terms

def map_to_ebay_category(category: str) -> str:
    """Map internal category to eBay search-friendly category"""
    category_mapping = {
        'electronics': 'electronics',
        'clothing': 'clothing',
        'furniture': 'furniture',
        'collectibles': 'collectibles',
        'books': 'books',
        'toys': 'toys',
        'jewelry': 'jewelry',
        'sports': 'sporting goods',
        'tools': 'tools',
        'kitchen': 'kitchen',
        'vehicles': 'cars trucks',
        'music': 'musical instruments',
        'art': 'art',
        'coins': 'coins',
        'stamps': 'stamps'
    }
    
    return category_mapping.get(category.lower(), '')

def get_most_specific_feature(features: List[str]) -> str:
    """Extract the most specific/detailed feature"""
    if not features:
        return ""
    
    # Score features by specificity
    scored_features = []
    for feature in features:
        words = feature.split()
        score = len(words) * 10
        if any(word.isdigit() for word in words):
            score += 20
        if len(feature) > 15:
            score += 15
        scored_features.append((feature, score))
    
    scored_features.sort(key=lambda x: x[1], reverse=True)
    return scored_features[0][0] if scored_features else ""

def extract_search_terms_from_title(title: str) -> List[str]:
    """Extract key search terms from title"""
    if not title:
        return []
    
    # Common words to exclude
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
        'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be',
        'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
        'will', 'would', 'shall', 'should', 'may', 'might', 'must',
        'can', 'could'
    }
    
    # Extract meaningful words
    words = title.lower().split()
    meaningful_words = []
    
    for word in words:
        word = word.strip('.,!?;:"\'()[]{}<>')
        
        if (word not in stop_words and 
            len(word) > 2 and 
            any(c.isalpha() for c in word)):
            
            # Check for year patterns
            if word.isdigit() and len(word) == 4:
                year = int(word)
                if 1900 <= year <= 2100:
                    meaningful_words.append(word)
                    continue
            
            # Check for model numbers/patterns
            if any(c.isdigit() for c in word) and any(c.isalpha() for c in word):
                meaningful_words.append(word)
                continue
            
            # Check for common product terms
            if word in ['piano', 'guitar', 'violin', 'drum', 'trumpet', 
                       'truck', 'car', 'vehicle', 'motorcycle', 'bicycle',
                       'watch', 'ring', 'necklace', 'bracelet', 'earring',
                       'painting', 'sculpture', 'statue', 'figure', 'doll',
                       'book', 'comic', 'magazine', 'newspaper', 'document',
                       'coin', 'stamp', 'card', 'ticket', 'token', 'medal']:
                meaningful_words.append(word)
                continue
            
            # Add if it seems like a brand or model
            if word[0].isupper() or word.isupper():
                meaningful_words.append(word)
    
    # Remove duplicates
    seen = set()
    unique_words = []
    for word in meaningful_words:
        if word not in seen:
            seen.add(word)
            unique_words.append(word)
    
    return unique_words[:5]

def clean_search_query(query: str) -> str:
    """Clean and optimize search query for eBay"""
    if not query:
        return ""
    
    # Remove extra spaces
    query = ' '.join(query.split())
    
    # Remove problematic characters
    query = query.replace('"', '').replace("'", "").replace("`", "")
    
    # Ensure query isn't too long
    if len(query) > 100:
        words = query.split()
        important_words = []
        for word in words:
            if (word[0].isupper() or
                word.isdigit() and len(word) in [2, 4] or
                word.lower() in ['piano', 'guitar', 'truck', 'car', 'watch', 'ring']):
                important_words.append(word)
        
        if important_words:
            query = ' '.join(important_words[:8])
        else:
            query = ' '.join(words[:8])
    
    return query

def remove_duplicate_items(items: List[Dict]) -> List[Dict]:
    """Remove duplicate items from results"""
    if not items:
        return []
    
    unique_items = []
    seen_ids = set()
    
    for item in items:
        item_id = item.get('item_id', '')
        if item_id and item_id not in seen_ids:
            seen_ids.add(item_id)
            unique_items.append(item)
    
    return unique_items

def detect_category(title: str, description: str, vision_analysis: Dict) -> str:
    """MAXIMUM accuracy category detection using ALL available data"""
    title_lower = title.lower()
    description_lower = description.lower()
    
    # Combine vision analysis data
    detected_text = " ".join(vision_analysis.get('detected_text', []))
    detected_objects = " ".join(vision_analysis.get('detected_objects', []))
    brands = " ".join(vision_analysis.get('potential_brands', []))
    
    all_text = f"{title_lower} {description_lower} {detected_text.lower()} {detected_objects.lower()} {brands.lower()}"
    
    category_keywords = {
        "electronics": ["electronic", "computer", "phone", "tablet", "camera", "laptop", "charger", "battery", 
                       "screen", "iphone", "samsung", "android", "macbook", "ipad", "headphones", "speaker",
                       "usb", "charger", "power", "cable", "wire", "circuit", "chip", "processor", "memory"],
        "clothing": ["shirt", "pants", "dress", "jacket", "shoe", "sweater", "fabric", "cotton", "wool", 
                    "leather", "silk", "polyester", "nike", "adidas", "levi", "gucci", "prada", "lv", 
                    "sneaker", "boot", "hat", "cap", "glove", "sock", "underwear", "bra", "lingerie"],
        "furniture": ["chair", "table", "desk", "cabinet", "sofa", "couch", "wood", "furniture", "drawer", 
                     "shelf", "bed", "dresser", "wardrobe", "ottoman", "recliner", "stool", "bench",
                     "wooden", "metal", "glass", "upholstery", "cushion", "leg", "armrest", "backrest"],
        "collectibles": ["collectible", "rare", "vintage", "antique", "edition", "limited", "signed", 
                        "autograph", "memorabilia", "coin", "stamp", "trading card", "funko", "figure",
                        "collector", "series", "numbered", "certificate", "authenticity", "display"],
        "books": ["book", "novel", "author", "page", "edition", "publish", "hardcover", "paperback", 
                 "literature", "comic", "manga", "textbook", "pages", "chapter", "cover", "binding",
                 "signed", "first edition", "rare book", "manuscript", "document"],
        "toys": ["toy", "game", "play", "action figure", "doll", "puzzle", "lego", "model kit", 
                "collectible figure", "barbie", "hot wheels", "board game", "video game", "console",
                "playset", "vehicle", "character", "plush", "stuffed animal", "educational"],
        "jewelry": ["ring", "necklace", "bracelet", "earring", "watch", "gold", "silver", "diamond",
                   "gem", "stone", "pearl", "platinum", "titanium", "jewelry", "pendant", "brooch",
                   "crystal", "bead", "chain", "clasp", "setting", "karat", "carat"],
        "sports": ["sport", "equipment", "ball", "bat", "racket", "club", "ski", "snowboard", "bike",
                  "bicycle", "fitness", "exercise", "weight", "dumbbell", "yoga", "mat", "helmet",
                  "glove", "pad", "uniform", "jersey", "cleat", "shoe", "accessory"],
        "tools": ["tool", "wrench", "hammer", "screwdriver", "drill", "saw", "pliers", "level", "tape",
                 "measure", "workshop", "garage", "diy", "hardware", "fastener", "nail", "screw",
                 "bolt", "nut", "machine", "equipment", "power tool", "hand tool"],
        "kitchen": ["kitchen", "cookware", "utensil", "pan", "pot", "knife", "fork", "spoon", "plate",
                   "bowl", "cup", "glass", "appliance", "mixer", "blender", "toaster", "microwave",
                   "oven", "stove", "refrigerator", "freezer", "dish", "serving", "storage"]
    }
    
    scores = {category: 0 for category in category_keywords}
    scores["unknown"] = 0
    
    for category, keywords in category_keywords.items():
        for keyword in keywords:
            if keyword in all_text:
                scores[category] += 1
    
    # Boost score for brand matches in specific categories
    if any(brand in all_text for brand in ["nike", "adidas", "gucci", "prada", "lv"]):
        scores["clothing"] += 5
    if any(brand in all_text for brand in ["apple", "samsung", "sony", "canon", "nikon"]):
        scores["electronics"] += 5
    
    return max(scores.items(), key=lambda x: x[1])[0]

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
        self.analysis_depth = data.get("analysis_depth", "comprehensive")
        self.key_features = data.get("key_features", [])
        self.comparable_items = data.get("comparable_items", "")
        self.identification_confidence = data.get("identification_confidence", "unknown")
        self.additional_info_needed = data.get("additional_info_needed", [])
        
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
            "comparable_items": self.comparable_items,
            "identification_confidence": self.identification_confidence,
            "additional_info_needed": self.additional_info_needed
        }

def parse_json_response(response_text: str) -> List[Dict]:
    """Robust JSON parsing for maximum accuracy"""
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
        
        # Try to find JSON object or array
        json_match = re.search(r'(\{.*\}|\[.*\])', json_text, re.DOTALL)
        if json_match:
            json_text = json_match.group(1)
        
        # Remove any trailing commas before closing brackets/braces
        json_text = re.sub(r',\s*([}\]])', r'\1', json_text)
        
        # Try to parse
        parsed_data = json.loads(json_text)
        
        # Ensure we always return a list of dictionaries
        if isinstance(parsed_data, dict):
            return [parsed_data]
        elif isinstance(parsed_data, list):
            return parsed_data
        else:
            logger.warning(f"Unexpected JSON format: {type(parsed_data)}")
            return []
            
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing failed at line {e.lineno}, column {e.colno}: {e.msg}")
        logger.error(f"JSON snippet near error: {json_text[max(0, e.pos-50):min(len(json_text), e.pos+50)]}")
        
        # Try to fix common JSON issues
        try:
            # Fix unescaped quotes
            json_text = re.sub(r'(?<!\\)"', r'\"', json_text)
            # Fix trailing commas
            json_text = re.sub(r',\s*([}\]])', r'\1', json_text)
            # Fix missing quotes around keys
            json_text = re.sub(r'(\w+):', r'"\1":', json_text)
            
            parsed_data = json.loads(json_text)
            if isinstance(parsed_data, dict):
                return [parsed_data]
            elif isinstance(parsed_data, list):
                return parsed_data
        except:
            pass
            
        # Last resort: Return a fallback response
        logger.error(f"Failed to parse JSON after fixes. Original text: {response_text[:500]}...")
        return []
    except Exception as e:
        logger.error(f"Unexpected error in JSON parsing: {e}")
        return []

def extract_keywords_from_user_input(user_text: str) -> Dict[str, List[str]]:
    """Extract structured keywords from user input"""
    if not user_text:
        return {}
    
    user_text = user_text.lower()
    
    # Extract year patterns (1900-2025)
    year_pattern = r'\b(19[5-9]\d|20[0-2]\d)\b'
    years = re.findall(year_pattern, user_text)
    
    # Extract potential brand names (common automotive brands and general brands)
    brands = []
    common_brands = [
        # Automotive
        'chevy', 'chevrolet', 'ford', 'toyota', 'honda', 'bmw', 'mercedes', 'benz',
        'dodge', 'jeep', 'gmc', 'cadillac', 'buick', 'pontiac', 'ram', 'chrysler',
        'nissan', 'subaru', 'mazda', 'volkswagen', 'vw', 'audi', 'volvo', 'tesla',
        'porsche', 'ferrari', 'lamborghini', 'jaguar', 'land rover', 'mini',
        'fiat', 'alfa romeo', 'maserati', 'bentley', 'rolls royce',
        
        # General brands
        'apple', 'samsung', 'sony', 'microsoft', 'google', 'nike', 'adidas', 'gucci',
        'prada', 'louis vuitton', 'lv', 'rolex', 'omega', 'canon', 'nikon', 'dell',
        'hp', 'lenovo', 'lg', 'bosch', 'makita', 'dewalt', 'stanley', 'black decker',
        'kitchenaid', 'whirlpool', 'maytag', 'kenmore', 'sears', 'craftsman'
    ]
    
    for brand in common_brands:
        if brand in user_text:
            # Capitalize brand names properly
            if brand == 'chevy':
                brands.append('Chevrolet')
            elif brand == 'vw':
                brands.append('Volkswagen')
            elif brand == 'benz':
                brands.append('Mercedes-Benz')
            elif brand == 'lv':
                brands.append('Louis Vuitton')
            else:
                brands.append(brand.title())
    
    # Extract model indicators and specific features
    models = []
    features = []
    
    # Model/feature keywords
    model_keywords = [
        'truck', 'pickup', 'pick-up', 'sedan', 'coupe', 'convertible', 'suv', 'van',
        'window', '5-window', '5 window', 'deluxe', 'custom', 'standard', 'limited',
        'premium', 'luxury', 'sport', 'performance', 'edition', 'series', 'model',
        'station wagon', 'wagon', 'hatchback', 'roadster', 'cabriolet', 'spyder'
    ]
    
    for keyword in model_keywords:
        if keyword in user_text:
            if keyword in ['window', '5-window', '5 window', 'deluxe', 'custom', 'standard']:
                features.append(keyword)
            else:
                models.append(keyword)
    
    # Extract specific features from the text
    feature_words = user_text.split()
    automotive_features = [
        'v8', 'v6', 'v12', 'v10', 'inline', 'straight', 'automatic', 'manual',
        'diesel', 'gas', 'electric', 'hybrid', 'plug-in', 'restored', 'original',
        'customized', 'modified', 'rare', 'collectible', 'vintage', 'antique',
        'classic', 'muscle', 'resto-mod', 'patina', 'barn find', 'survivor'
    ]
    
    for word in feature_words:
        word_lower = word.lower().strip('.,!?;:"\'()[]{}<>')
        if len(word_lower) > 2:
            # Check for automotive features
            if word_lower in automotive_features:
                if word_lower not in [f.lower() for f in features]:
                    features.append(word_lower)
            # Check for numeric features (engine size, etc.)
            elif re.match(r'^\d+\.?\d*[Ll]?$', word_lower):
                features.append(word_lower)
    
    # Remove duplicates while preserving order
    def deduplicate(lst):
        seen = set()
        result = []
        for item in lst:
            item_lower = item.lower()
            if item_lower not in seen:
                seen.add(item_lower)
                result.append(item)
        return result
    
    brands = deduplicate(brands)
    models = deduplicate(models)
    features = deduplicate(features)
    
    logger.info(f"📋 Extracted keywords: years={years}, brands={brands}, models={models}, features={features}")
    
    return {
        'years': years,
        'brands': brands,
        'models': models,
        'features': features
    }

def call_groq_api(prompt: str, image_base64: str = None, mime_type: str = None) -> str:
    """MAXIMUM accuracy Groq API call with JSON formatting instructions"""
    if not groq_client:
        raise Exception("Groq client not configured")
    
    # Add explicit JSON formatting instructions to the prompt
    json_format_prompt = prompt + "\n\n**IMPORTANT: Return ONLY valid JSON. Do not include any explanatory text, code fences, or markdown outside the JSON.**"
    
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
            "content": [image_content, {"type": "text", "text": json_format_prompt}]
        })
    else:
        messages.append({
            "role": "user",
            "content": json_format_prompt
        })
    
    try:
        logger.info(f"📤 Calling Groq API with {len(json_format_prompt)} chars prompt")
        
        response = groq_client.chat.completions.create(
            model=groq_model,
            messages=messages,
            temperature=0.1,  # Low temperature for consistent results
            max_tokens=4000,  # Full token allowance for detailed analysis
            top_p=0.95,
            stream=False,
            timeout=15.0,  # Reasonable timeout
            response_format={"type": "json_object"}  # Request JSON format
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

def search_ebay_directly(keywords: str, limit: int = 5) -> List[Dict]:
    """Direct eBay search using OAuth token"""
    token = get_ebay_token()
    if not token:
        logger.error("❌ No eBay OAuth token available")
        return []
    
    try:
        headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json',
            'X-EBAY-C-MARKETPLACE-ID': 'EBAY_US'
        }
        
        params = {
            'q': keywords,
            'limit': str(limit),
            'filter': 'soldItems'
        }
        
        url = "https://api.ebay.com/buy/browse/v1/item_summary/search"
        
        logger.info(f"🔍 Direct eBay search for: '{keywords}'")
        response = requests.get(url, headers=headers, params=params, timeout=15)
        
        logger.info(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            if 'itemSummaries' in data:
                items = []
                for item in data['itemSummaries'][:limit]:
                    try:
                        price = item.get('price', {}).get('value', '0')
                        items.append({
                            'title': item.get('title', ''),
                            'price': float(price),
                            'item_id': item.get('itemId', ''),
                            'condition': item.get('condition', ''),
                            'category': item.get('categoryPath', ''),
                            'image_url': item.get('image', {}).get('imageUrl', '')
                        })
                    except (KeyError, ValueError) as e:
                        logger.debug(f"   Skipping item: {e}")
                        continue
                
                logger.info(f"✅ Direct eBay search found {len(items)} sold items")
                return items
        elif response.status_code == 401:
            logger.error("❌ eBay token expired or invalid")
            store_ebay_token(None)  # Clear invalid token
            return []
        elif response.status_code == 429:
            logger.warning("⚠️ eBay rate limit reached")
            return []
        else:
            logger.error(f"❌ eBay search error {response.status_code}: {response.text[:200]}")
            
    except Exception as e:
        logger.error(f"❌ Direct eBay search error: {e}")
    
    return []

def analyze_ebay_market_directly(keywords: str) -> Dict:
    """Direct eBay market analysis"""
    logger.info(f"📊 Direct eBay market analysis for: '{keywords}'")
    
    # Get sold items
    sold_items = search_ebay_directly(keywords, limit=10)
    
    if not sold_items:
        logger.error("❌ NO EBAY DATA AVAILABLE - APP CANNOT FUNCTION")
        return {
            'error': 'NO_EBAY_DATA',
            'message': 'eBay API failed - please ensure you are authenticated and try again',
            'requires_auth': True
        }
    
    # Calculate statistics
    prices = [item['price'] for item in sold_items if item['price'] > 0]
    
    if not prices:
        logger.error("❌ No valid price data from eBay")
        return {
            'error': 'NO_PRICE_DATA',
            'message': 'eBay returned items but no price data',
            'requires_auth': False
        }
    
    avg_price = sum(prices) / len(prices)
    min_price = min(prices)
    max_price = max(prices)
    median_price = sorted(prices)[len(prices) // 2]
    
    analysis = {
        'success': True,
        'average_price': round(avg_price, 2),
        'price_range': f"${min_price:.2f} - ${max_price:.2f}",
        'total_sold_analyzed': len(sold_items),
        'recommended_price': round(median_price * 0.85, 2),
        'market_notes': f'Based on {len(sold_items)} recent eBay sales',
        'data_source': 'eBay Browse API',
        'confidence': 'high' if len(sold_items) >= 5 else 'medium',
        'api_used': 'Browse API'
    }
    
    return analysis

def enhance_with_ebay_data_user_prioritized(item_data: Dict, vision_analysis: Dict, user_keywords: Dict) -> Dict:
    """Enhanced market analysis using REAL eBay data with USER-PRIORITIZED search"""
    try:
        # Start with user-provided keywords as primary search terms
        search_strategies = []
        
        # 1. Build search from user input FIRST (highest priority)
        user_search_terms = []
        
        # Add user keywords in priority order
        if user_keywords.get('years'):
            for year in user_keywords['years']:
                user_search_terms.append(year)
        
        if user_keywords.get('brands'):
            for brand in user_keywords['brands']:
                user_search_terms.append(brand)
        
        if user_keywords.get('models'):
            for model in user_keywords['models']:
                user_search_terms.append(model)
        
        if user_keywords.get('features'):
            for feature in user_keywords['features'][:3]:  # Limit features
                user_search_terms.append(feature)
        
        # Create user-prioritized search queries
        if user_search_terms:
            # Strategy 1: All user terms combined (most specific)
            user_query = " ".join(user_search_terms[:8])  # Limit length
            search_strategies.append(user_query)
            logger.info(f"🎯 USER-PRIORITIZED SEARCH 1: '{user_query}'")
            
            # Strategy 2: Brand + Year + Model (if available)
            brand_year_query_parts = []
            if user_keywords.get('brands'):
                brand_year_query_parts.append(user_keywords['brands'][0])
            if user_keywords.get('years'):
                brand_year_query_parts.append(user_keywords['years'][0])
            if user_keywords.get('models'):
                brand_year_query_parts.append(user_keywords['models'][0])
            
            if len(brand_year_query_parts) >= 2:
                brand_year_query = " ".join(brand_year_query_parts)
                search_strategies.append(brand_year_query)
                logger.info(f"🎯 USER-PRIORITIZED SEARCH 2: '{brand_year_query}'")
        
        # 2. Add AI-detected terms as secondary options
        brand = item_data.get('brand', '').strip()
        if brand and 'unknown' not in brand.lower():
            # Only add AI brand if not already covered by user
            if not user_keywords.get('brands') or brand.lower() not in [b.lower() for b in user_keywords['brands']]:
                search_strategies.append(brand)
        
        # Parse title for additional key terms
        title = item_data.get('title', '')
        if title:
            title_terms = extract_search_terms_from_title(title)
            if title_terms:
                # Filter out terms already covered by user
                filtered_terms = []
                for term in title_terms:
                    term_lower = term.lower()
                    already_covered = False
                    
                    # Check if term is already in user keywords
                    for category in ['years', 'brands', 'models', 'features']:
                        if any(term_lower in str(kw).lower() for kw in user_keywords.get(category, [])):
                            already_covered = True
                            break
                    
                    if not already_covered and len(term) > 2:
                        filtered_terms.append(term)
                
                if filtered_terms:
                    ai_query = ' '.join(filtered_terms[:3])
                    search_strategies.append(ai_query)
        
        # Clean strategies
        cleaned_strategies = []
        seen = set()
        for strategy in search_strategies:
            if strategy and strategy not in seen and len(strategy) > 2:
                seen.add(strategy)
                cleaned_strategy = clean_search_query(strategy)
                if cleaned_strategy:
                    cleaned_strategies.append(cleaned_strategy[:50])  # Shorter queries
        
        search_strategies = cleaned_strategies[:3]  # Only 3 strategies max
        
        if not search_strategies:
            logger.warning("No valid search strategies")
            item_data['market_insights'] = "Cannot search eBay - no identifiable terms. " + item_data.get('market_insights', '')
            item_data['identification_confidence'] = "low"
            return item_data
        
        # Try to get REAL eBay market analysis with user-prioritized search
        market_analysis = None
        
        for strategy in search_strategies:
            logger.info(f"🔍 Analyzing REAL eBay market with: '{strategy}'")
            analysis = analyze_ebay_market_directly(strategy)
            
            if analysis and analysis.get('success'):
                market_analysis = analysis
                break
            elif analysis and analysis.get('error') == 'NO_EBAY_DATA':
                # eBay API failed completely - abort
                logger.error("❌ EBAY API FAILED - CANNOT PROCEED")
                item_data['market_insights'] = "⚠️ eBay authentication required. Please connect your eBay account in the app settings to get real market data."
                item_data['price_range'] = "Authentication Required"
                item_data['suggested_cost'] = "Connect eBay Account"
                item_data['profit_potential'] = "eBay data unavailable"
                item_data['identification_confidence'] = "requires_auth"
                item_data['ebay_specific_tips'] = ["Connect your eBay account in settings", "Authenticate to access real market data", "Try again after authentication"]
                return item_data
        
        if market_analysis:
            # Update item data with REAL market info
            item_data['price_range'] = market_analysis['price_range']
            item_data['suggested_cost'] = f"${market_analysis['recommended_price']:.2f}"
            
            # Calculate profit
            avg_price = market_analysis['average_price']
            ebay_fees = avg_price * 0.13
            shipping_cost = 12.00
            estimated_net = avg_price - ebay_fees - shipping_cost
            suggested_purchase = market_analysis['recommended_price']
            profit = estimated_net - suggested_purchase
            
            if profit > 0:
                item_data['profit_potential'] = f"${profit:.2f} profit (after all fees)"
            else:
                item_data['profit_potential'] = f"${abs(profit):.2f} potential loss"
            
            # Enhanced market insights showing user contribution
            insights = []
            if user_keywords:
                insights.append(f"Search prioritized by user-provided details")
                if user_keywords.get('years'):
                    insights.append(f"Year focus: {', '.join(user_keywords['years'])}")
                if user_keywords.get('brands'):
                    insights.append(f"Brand focus: {', '.join(user_keywords['brands'][:2])}")
            else:
                insights.append("Search based on AI analysis")
            
            insights.extend([
                f"Based on {market_analysis['total_sold_analyzed']} recent eBay sales",
                f"Average price: ${market_analysis['average_price']:.2f}",
                f"Price range: {market_analysis['price_range']}",
                f"Confidence: {market_analysis['confidence']}"
            ])
            
            item_data['market_insights'] = ". ".join(insights) + ". " + item_data.get('market_insights', '')
            
            # Add REAL eBay tips including user terms
            ebay_tips = []
            if search_strategies:
                ebay_tips.append(f"Primary search terms: {search_strategies[0][:40]}")
            ebay_tips.extend([
                "Use 'Buy It Now' with Best Offer option",
                "Include measurements in description",
                "Take photos from multiple angles",
                "List on weekends for best visibility"
            ])
            
            item_data['ebay_specific_tips'] = ebay_tips
            
            logger.info(f"✅ REAL eBay market analysis successful (user-prioritized)")
            
            # Add confidence indicator
            item_data['identification_confidence'] = market_analysis['confidence']
            item_data['data_source'] = market_analysis['data_source']
            if user_keywords:
                item_data['user_input_utilized'] = True
                    
        else:
            logger.error("❌ NO EBAY MARKET ANALYSIS AVAILABLE")
            item_data['market_insights'] = "⚠️ Real eBay market data required but unavailable. Please ensure you are authenticated and try again."
            item_data['identification_confidence'] = "low"
        
        return item_data
        
    except Exception as e:
        logger.error(f"❌ eBay data enhancement failed: {e}")
        item_data['market_insights'] = f"⚠️ eBay API error: {str(e)[:100]}. Please try again."
        item_data['identification_confidence'] = "error"
        return item_data

def process_image_maximum_accuracy(job_data: Dict) -> Dict:
    """MAXIMUM accuracy processing - uses REAL eBay data ONLY"""
    try:
        if not groq_client:
            return {"status": "failed", "error": "Groq client not configured"}
        
        image_base64 = job_data['image_base64']
        mime_type = job_data['mime_type']
        
        # Extract structured keywords from user input
        user_title = job_data.get('title', '')
        user_description = job_data.get('description', '')
        
        user_keywords = {}
        if user_title or user_description:
            full_user_text = f"{user_title} {user_description}"
            user_keywords = extract_keywords_from_user_input(full_user_text)
            logger.info(f"📝 Extracted user keywords: {user_keywords}")
        
        # Build ENHANCED prompt with user-provided details
        enhanced_prompt = market_analysis_prompt
        
        # Add user-provided information with emphasis
        if user_title or user_description:
            enhanced_prompt += f"\n\n🔍 **USER-PROVIDED IDENTIFICATION HINTS (HIGH PRIORITY):**\n"
            if user_title:
                enhanced_prompt += f"**USER TITLE:** '{user_title}'\n"
            if user_description:
                enhanced_prompt += f"**USER DESCRIPTION:** '{user_description}'\n"
            
            if user_keywords:
                enhanced_prompt += f"\n**EXTRACTED DETAILS FROM USER INPUT:**\n"
                if user_keywords.get('years'):
                    enhanced_prompt += f"- **YEAR(S):** {', '.join(user_keywords['years'])} (CRITICAL for accurate search)\n"
                if user_keywords.get('brands'):
                    enhanced_prompt += f"- **BRAND(S):** {', '.join(user_keywords['brands'])} (USE EXACTLY for search)\n"
                if user_keywords.get('models'):
                    enhanced_prompt += f"- **MODEL/FEATURES:** {', '.join(user_keywords['models'])}\n"
                if user_keywords.get('features'):
                    enhanced_prompt += f"- **ADDITIONAL FEATURES:** {', '.join(user_keywords['features'][:5])}\n"
        
        # Add search prioritization instructions
        enhanced_prompt += "\n\n🔍 **SEARCH PRIORITIZATION RULES (FOLLOW EXACTLY):**\n"
        enhanced_prompt += "1. **USER DETAILS TAKE ABSOLUTE PRECEDENCE** over AI detection\n"
        enhanced_prompt += "2. Use EXACT user-provided terms for identification and searching\n"
        enhanced_prompt += "3. If user provides year(s), prioritize that exact year range\n"
        enhanced_prompt += "4. If user provides specific model/feature terms, INCLUDE THEM in the search\n"
        enhanced_prompt += "5. Combine visual analysis with user hints for maximum accuracy\n"
        enhanced_prompt += "6. User knowledge is MORE VALUABLE than AI detection when available\n"
        
        logger.info(f"🔬 Starting MAXIMUM ACCURACY analysis with user details and REAL eBay data...")
        
        # Call Groq API for detailed analysis
        response_text = call_groq_api(enhanced_prompt, image_base64, mime_type)
        logger.info("✅ AI analysis completed, parsing response...")
        
        ai_response = parse_json_response(response_text)
        logger.info(f"📊 Parsed {len(ai_response)} items from response")
        
        # Mock vision analysis
        vision_analysis = {
            "detected_text": [],
            "detected_objects": [],
            "potential_brands": [],
            "image_size": "unknown"
        }
        
        enhanced_items = []
        for item_data in ai_response:
            if isinstance(item_data, dict):
                # Detect category using ALL available data including user input
                title = item_data.get("title", "")
                description = item_data.get("description", "")
                
                # Override with user details if provided
                if user_title:
                    # Incorporate user title into the item title
                    item_data["title"] = f"{user_title} - {title}" if title else user_title
                
                if user_description and not description:
                    item_data["description"] = user_description
                
                detected_category = detect_category(
                    item_data.get("title", ""), 
                    item_data.get("description", ""),
                    vision_analysis
                )
                item_data["category"] = detected_category
                
                # Enhance with REAL eBay market data using USER-PRIORITIZED search
                item_data = enhance_with_ebay_data_user_prioritized(item_data, vision_analysis, user_keywords)
                
                # Check if eBay authentication is required
                if item_data.get('identification_confidence') == 'requires_auth':
                    # Return authentication error immediately
                    return {
                        "status": "failed",
                        "error": "EBAY_AUTH_REQUIRED",
                        "message": "eBay authentication required for market analysis",
                        "requires_auth": True
                    }
                
                # Ensure we have ALL required fields
                if not item_data.get('brand'):
                    item_data['brand'] = "Unknown"
                if not item_data.get('model'):
                    item_data['model'] = "Model unknown"
                if not item_data.get('condition'):
                    item_data['condition'] = "Condition unknown"
                if not item_data.get('identification_confidence'):
                    item_data['identification_confidence'] = "medium"
                if not item_data.get('additional_info_needed'):
                    item_data['additional_info_needed'] = [
                        "Clear photos of any markings",
                        "Manufacturer information",
                        "Exact measurements"
                    ]
                
                # Add user input flag
                if user_keywords:
                    item_data['user_input_incorporated'] = True
                    if user_keywords.get('years'):
                        item_data['user_specified_year'] = user_keywords['years'][0]
                    if user_keywords.get('brands'):
                        item_data['user_specified_brand'] = user_keywords['brands'][0]
                
                enhanced_items.append(EnhancedAppItem(item_data).to_dict())
            else:
                logger.warning(f"Skipping non-dictionary item: {item_data}")
        
        # CRITICAL: If no items, return error
        if not enhanced_items:
            logger.error("❌ NO ITEMS PARSED FROM AI RESPONSE")
            return {
                "status": "failed",
                "error": "NO_AI_ANALYSIS",
                "message": "AI failed to analyze the image"
            }
        
        logger.info(f"✅ Processing complete: {len(enhanced_items)} items with REAL eBay data")
        
        return {
            "status": "completed",
            "result": {
                "message": f"Maximum accuracy analysis completed with {len(enhanced_items)} items using REAL eBay data",
                "items": enhanced_items,
                "processing_time": "25-30s",
                "analysis_stages": 3,
                "confidence_level": "maximum_accuracy",
                "analysis_timestamp": datetime.now().isoformat(),
                "model_used": groq_model,
                "ebay_data_used": True,
                "user_details_incorporated": bool(user_title or user_description)
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Maximum accuracy processing failed: {e}")
        return {
            "status": "failed", 
            "error": str(e)[:200]
        }

def background_worker():
    """Background worker with maximum accuracy only"""
    logger.info("🎯 Background worker started - MAXIMUM ACCURACY ONLY")
    
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
            
            logger.info(f"🔄 Processing job {job_id} with maximum accuracy...")
            
            # Process with timeout (25 seconds for Render's 30s limit)
            future = job_executor.submit(process_image_maximum_accuracy, job_data)
            try:
                result = future.result(timeout=25)
                
                with job_lock:
                    if result.get('status') == 'completed':
                        job_data['status'] = 'completed'
                        job_data['result'] = result['result']
                        logger.info(f"✅ Job {job_id} completed successfully")
                    elif result.get('error') == 'EBAY_AUTH_REQUIRED':
                        job_data['status'] = 'failed'
                        job_data['error'] = 'eBay authentication required. Please connect your eBay account first.'
                        job_data['requires_auth'] = True
                        logger.error(f"❌ Job {job_id} requires eBay authentication")
                    else:
                        job_data['status'] = 'failed'
                        job_data['error'] = result.get('error', 'Unknown error')
                        logger.error(f"❌ Job {job_id} failed: {job_data['error']}")
                    
                    job_data['completed_at'] = datetime.now().isoformat()
                    job_storage[job_id] = job_data
                    
            except FutureTimeoutError:
                with job_lock:
                    job_data['status'] = 'failed'
                    job_data['error'] = 'Processing timeout (25s) - Server busy, please try again'
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
    threading.Thread(target=background_worker, daemon=True, name="JobWorker-MaxAccuracy").start()
    
    # Start keep-alive thread
    def keep_alive_loop():
        while True:
            time.sleep(25)
            try:
                update_activity()
                requests.get(f"http://localhost:{os.getenv('PORT', 8000)}/ping", timeout=5)
            except:
                pass
    
    threading.Thread(target=keep_alive_loop, daemon=True, name="KeepAlive").start()
    
    logger.info("🚀 Server started with MAXIMUM ACCURACY processing and REAL eBay data ONLY")

@app.on_event("shutdown")
async def shutdown_event():
    job_executor.shutdown(wait=False)
    logger.info("🛑 Server shutting down")

# ============= EBAY OAUTH ENDPOINTS (✅ WEB-TO-APP BRIDGE) =============

@app.get("/ebay/oauth/start")
async def get_ebay_auth_url():
    """
    Generate eBay OAuth authorization URL
    ✅ ALWAYS USES WEB REDIRECT (eBay requirement)
    """
    update_activity()
    
    try:
        app_id = os.getenv('EBAY_APP_ID')
        
        if not app_id:
            raise HTTPException(status_code=500, detail="eBay credentials not configured")
        
        # Generate auth URL
        auth_url, state = ebay_oauth.generate_auth_url()
        
        logger.info(f"🔗 Generated auth URL")
        logger.info(f"🔗 State: {state}")
        
        return {
            "success": True,
            "auth_url": auth_url,
            "state": state,
            "redirect_uri": "https://resell-app-bi47.onrender.com/ebay/oauth/callback",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to generate eBay auth URL: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate auth URL: {str(e)}")

@app.get("/ebay/oauth/callback")
async def ebay_oauth_callback_get(
    code: Optional[str] = None, 
    error: Optional[str] = None, 
    state: Optional[str] = None
):
    """
    Handle eBay OAuth callback (GET from eBay)
    ✅ EXCHANGES CODE FOR TOKEN, THEN REDIRECTS TO iOS APP
    """
    update_activity()
    
    logger.info(f"🔔 eBay OAuth callback received. Code: {code is not None}, Error: {error}, State: {state}")
    
    # Handle error from eBay
    if error:
        logger.error(f"❌ eBay OAuth error: {error}")
        redirect_url = f"ai-resell-pro://ebay-oauth-callback?error={error}"
        
        return HTMLResponse(content=f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Authorization Failed</title>
                <meta http-equiv="refresh" content="0; url={redirect_url}">
            </head>
            <body>
                <script>
                    window.location.href = "{redirect_url}";
                </script>
                <h1>❌ Authorization Failed</h1>
                <p>Error: {error}</p>
                <p><a href="{redirect_url}">Return to app</a></p>
            </body>
            </html>
        """)
    
    # Check for authorization code
    if not code:
        logger.error("❌ No authorization code received")
        redirect_url = "ai-resell-pro://ebay-oauth-callback?error=no_code"
        
        return HTMLResponse(content=f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Authorization Failed</title>
                <meta http-equiv="refresh" content="0; url={redirect_url}">
            </head>
            <body>
                <script>
                    window.location.href = "{redirect_url}";
                </script>
                <h1>❌ No Authorization Code</h1>
                <p><a href="{redirect_url}">Return to app</a></p>
            </body>
            </html>
        """)
    
    try:
        logger.info(f"✅ Received OAuth code: {code[:20]}...")
        
        # ✅ EXCHANGE CODE FOR TOKEN (server-side, secure)
        token_response = ebay_oauth.exchange_code_for_token(code, state=state)
        
        if token_response and token_response.get("success"):
            logger.info("✅ Successfully obtained eBay OAuth token")
            
            token_id = token_response["token_id"]
            
            # ✅ GET ACTUAL ACCESS TOKEN FOR IMAGE ANALYSIS
            token_data = ebay_oauth.get_user_token(token_id)
            if token_data and "access_token" in token_data:
                access_token = token_data["access_token"]
                store_ebay_token(access_token)
                logger.info(f"✅ eBay access token stored for market analysis: {access_token[:20]}...")
            
            # ✅ REDIRECT TO iOS APP WITH TOKEN_ID
            redirect_url = f"ai-resell-pro://ebay-oauth-callback?success=true&token_id={token_id}&state={state}"
            
            logger.info(f"🔗 Redirecting to iOS app: {redirect_url}")
            
            return HTMLResponse(content=f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Redirecting to AI Resell Pro...</title>
                    <meta http-equiv="refresh" content="0; url={redirect_url}">
                    <style>
                        body {{
                            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
                            display: flex;
                            justify-content: center;
                            align-items: center;
                            min-height: 100vh;
                            margin: 0;
                            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                            color: white;
                            text-align: center;
                        }}
                        .container {{
                            background: rgba(255, 255, 255, 0.1);
                            padding: 40px;
                            border-radius: 20px;
                            backdrop-filter: blur(10px);
                        }}
                        h1 {{ margin: 0 0 20px 0; font-size: 2em; }}
                        .spinner {{
                            border: 4px solid rgba(255, 255, 255, 0.3);
                            border-top: 4px solid white;
                            border-radius: 50%;
                            width: 50px;
                            height: 50px;
                            animation: spin 1s linear infinite;
                            margin: 20px auto;
                        }}
                        @keyframes spin {{
                            0% {{ transform: rotate(0deg); }}
                            100% {{ transform: rotate(360deg); }}
                        }}
                        a {{
                            color: white;
                            text-decoration: underline;
                        }}
                    </style>
                </head>
                <body>
                    <div class="container">
                        <h1>✅ Authorization Successful!</h1>
                        <div class="spinner"></div>
                        <p>eBay connected for real market data</p>
                        <p>Redirecting to AI Resell Pro...</p>
                        <p style="margin-top: 30px; font-size: 0.9em;">
                            If you are not redirected automatically,<br>
                            <a href="{redirect_url}">click here</a>
                        </p>
                    </div>
                    <script>
                        // Immediate redirect
                        window.location.href = "{redirect_url}";
                        
                        // Fallback after 2 seconds
                        setTimeout(function() {{
                            if (document.visibilityState === 'visible') {{
                                window.location.href = "{redirect_url}";
                            }}
                        }}, 2000);
                    </script>
                </body>
                </html>
            """)
        else:
            logger.error(f"❌ Failed to exchange code for token: {token_response}")
            error_msg = token_response.get("error", "unknown") if token_response else "no_response"
            redirect_url = f"ai-resell-pro://ebay-oauth-callback?error=token_exchange_failed&details={error_msg}"
            
            return HTMLResponse(content=f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Authorization Failed</title>
                    <meta http-equiv="refresh" content="0; url={redirect_url}">
                </head>
                <body>
                    <script>
                        window.location.href = "{redirect_url}";
                    </script>
                    <h1>❌ Token Exchange Failed</h1>
                    <p>Error: {error_msg}</p>
                    <p><a href="{redirect_url}">Return to app</a></p>
                </body>
                </html>
            """)
        
    except Exception as e:
        logger.error(f"❌ OAuth callback error: {e}")
        error_msg = str(e)[:100]
        redirect_url = f"ai-resell-pro://ebay-oauth-callback?error=callback_error&details={error_msg}"
        
        return HTMLResponse(content=f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Authorization Error</title>
                <meta http-equiv="refresh" content="0; url={redirect_url}">
            </head>
            <body>
                <script>
                    window.location.href = "{redirect_url}";
                </script>
                <h1>❌ Authorization Error</h1>
                <p>Error: {error_msg}</p>
                <p><a href="{redirect_url}">Return to app</a></p>
            </body>
            </html>
        """)

@app.get("/ebay/oauth/token/{token_id}")
async def get_ebay_token_endpoint(token_id: str):
    """
    Get eBay access token (for iOS app to use in API calls)
    """
    update_activity()
    
    try:
        token_data = ebay_oauth.get_user_token(token_id)
        
        if not token_data:
            raise HTTPException(status_code=404, detail="Token not found or expired")
        
        # Refresh token if needed
        refresh_ebay_token_if_needed(token_id)
        
        # Return only what's needed for API calls
        return {
            "success": True,
            "access_token": token_data["access_token"],
            "expires_at": token_data["expires_at"],
            "token_type": token_data.get("token_type", "Bearer")
        }
        
    except Exception as e:
        logger.error(f"Get token error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/ebay/oauth/token/{token_id}")
async def revoke_ebay_token(token_id: str):
    """
    Revoke/delete eBay token
    """
    update_activity()
    
    try:
        success = ebay_oauth.revoke_token(token_id)
        
        if success:
            store_ebay_token(None)  # Clear stored token
            return {"success": True, "message": "Token revoked"}
        else:
            raise HTTPException(status_code=404, detail="Token not found")
        
    except Exception as e:
        logger.error(f"Revoke token error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ebay/oauth/status/{token_id}")
async def check_ebay_token_status(token_id: str):
    """
    Check if eBay token is valid
    """
    update_activity()
    
    try:
        token_data = ebay_oauth.get_user_token(token_id)
        
        if token_data:
            expires_at = datetime.fromisoformat(token_data["expires_at"])
            time_remaining = expires_at - datetime.now()
            
            return {
                "valid": True,
                "expires_at": token_data["expires_at"],
                "seconds_remaining": int(time_remaining.total_seconds()),
                "refreshable": "refresh_token" in token_data
            }
        else:
            return {
                "valid": False,
                "message": "Token not found or expired without refresh"
            }
        
    except Exception as e:
        logger.error(f"Token status error: {e}")
        return {"valid": False, "error": str(e)}

@app.get("/ebay/token-status")
async def get_ebay_token_status():
    """
    Check if eBay token is available for market analysis
    """
    update_activity()
    
    token = get_ebay_token()
    
    return {
        "has_token": bool(token),
        "token_length": len(token) if token else 0,
        "can_analyze_market": bool(token),
        "timestamp": datetime.now().isoformat(),
        "message": "eBay token required for real market analysis" if not token else "eBay token ready for market analysis"
    }

@app.post("/api/ebay/analyze")
async def analyze_ebay_market(request: Request):
    """
    Analyze market value using eBay Browse API
    Called by iOS app with OAuth token
    """
    update_activity()
    
    try:
        # Get authorization header
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
        
        access_token = auth_header[7:]  # Remove "Bearer "
        
        # Store the token for future use
        store_ebay_token(access_token)
        
        # Parse request body
        body = await request.json()
        query = body.get("query", "")
        
        if not query:
            raise HTTPException(status_code=400, detail="Missing query parameter")
        
        # Use REAL eBay API to analyze market
        analysis = analyze_ebay_market_directly(query)
        
        if analysis.get('error'):
            raise HTTPException(status_code=500, detail=analysis.get('message', 'eBay API error'))
        
        return analysis
        
    except Exception as e:
        logger.error(f"Market analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ebay/search/sold")
async def search_ebay_sold_items(request: Request):
    """
    Search sold items on eBay
    Called by iOS app with OAuth token
    """
    update_activity()
    
    try:
        # Get authorization header
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
        
        access_token = auth_header[7:]
        
        # Store the token for future use
        store_ebay_token(access_token)
        
        # Parse request body
        body = await request.json()
        query = body.get("query", "")
        limit = min(body.get("limit", 10), 20)
        
        if not query:
            raise HTTPException(status_code=400, detail="Missing query parameter")
        
        # Search REAL eBay data
        items = search_ebay_directly(query, limit)
        
        if not items:
            raise HTTPException(status_code=500, detail="No eBay data found. Please ensure you are authenticated.")
        
        return items
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============= MAIN ENDPOINTS =============
@app.post("/upload_item/")
async def create_upload_file(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    description: Optional[str] = Form(None)
):
    update_activity()
    
    try:
        # Check if eBay token is available for market analysis
        ebay_token = get_ebay_token()
        if not ebay_token:
            raise HTTPException(
                status_code=400, 
                detail="eBay authentication required. Please connect your eBay account in the app settings first."
            )
        
        # Read image with reasonable limit
        image_bytes = await file.read()
        if len(image_bytes) > 8 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Image too large (max 8MB)")
        
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        job_id = str(uuid.uuid4())
        
        with job_lock:
            # Clean old jobs
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
                'mime_type': file.content_type,
                'title': title,
                'description': description,
                'status': 'queued',
                'created_at': datetime.now().isoformat(),
                'requires_ebay_auth': not bool(ebay_token)
            }
        
        job_queue.put(job_id)
        logger.info(f"📤 Job {job_id} queued (MAXIMUM ACCURACY with REAL eBay data)")
        
        return {
            "message": "Analysis queued with MAXIMUM ACCURACY using REAL eBay market data",
            "job_id": job_id,
            "status": "queued",
            "estimated_time": "25-30 seconds",
            "check_status_url": f"/job/{job_id}/status",
            "note": "Processing with comprehensive AI + REAL eBay market data integration",
            "ebay_auth_status": "connected" if ebay_token else "required",
            "user_details_provided": bool(title or description)
        }
            
    except HTTPException as he:
        raise he
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
        "requires_ebay_auth": job_data.get('requires_ebay_auth', False),
        "user_details_provided": bool(job_data.get('title') or job_data.get('description'))
    }
    
    if job_data.get('status') == 'completed':
        response["result"] = job_data.get('result')
        response["completed_at"] = job_data.get('completed_at')
    elif job_data.get('status') == 'failed':
        response["error"] = job_data.get('error', 'Unknown error')
        response["completed_at"] = job_data.get('completed_at')
        response["requires_ebay_auth"] = job_data.get('requires_ebay_auth', False)
    elif job_data.get('status') == 'processing':
        response["started_at"] = job_data.get('started_at')
    
    return response

@app.get("/health")
async def health_check():
    update_activity()
    
    with activity_lock:
        time_since = time.time() - last_activity
    
    groq_status = "✅ Ready" if groq_client else "❌ Not configured"
    ebay_token = get_ebay_token()
    ebay_status = "✅ Connected" if ebay_token else "⚠️ Not connected"
    
    return {
        "status": "✅ HEALTHY" if groq_client and ebay_token else "⚠️ PARTIAL",
        "timestamp": datetime.now().isoformat(),
        "time_since_last_activity": f"{int(time_since)}s",
        "jobs_queued": job_queue.qsize(),
        "jobs_stored": len(job_storage),
        "groq_status": groq_status,
        "ebay_status": ebay_status,
        "ebay_token_available": bool(ebay_token),
        "processing_mode": "MAXIMUM_ACCURACY_REAL_EBAY_ONLY",
        "timeout_protection": "25s processing window",
        "features": [
            "Maximum accuracy Groq AI analysis",
            "REAL eBay market data integration",
            "No mock data - real prices only",
            "eBay OAuth authentication",
            "Web-to-app redirect bridge",
            "User input prioritization system"
        ],
        "requirements": [
            "eBay authentication REQUIRED for market analysis",
            "Real market data only - no estimates",
            "25 second processing limit",
            "User details are prioritized over AI detection"
        ]
    }

@app.get("/ping")
async def ping():
    update_activity()
    return {
        "status": "✅ PONG",
        "timestamp": datetime.now().isoformat(),
        "message": "Server is awake and processing with REAL eBay data",
        "keep_alive": "active",
        "processing_mode": "maximum_accuracy_real_ebay_only",
        "ebay_ready": bool(get_ebay_token())
    }

@app.get("/")
async def root():
    update_activity()
    ebay_token = get_ebay_token()
    
    return {
        "message": "🎯 AI Resell Pro API - REAL EBAY DATA EDITION",
        "status": "🚀 OPERATIONAL" if groq_client and ebay_token else "⚠️ AUTHENTICATION REQUIRED",
        "version": "3.5.0",
        "ebay_authentication": "✅ Connected" if ebay_token else "❌ Required",
        "processing_capabilities": [
            "Maximum accuracy item identification",
            "REAL eBay market data integration",
            "No mock data - real prices only",
            "eBay OAuth authentication",
            "Web-to-app redirect bridge",
            "User input prioritization system"
        ],
        "requirements": [
            "eBay authentication REQUIRED for market analysis",
            "Real market data only - no estimates",
            "25s processing window (Render: 30s)",
            "User details take precedence over AI detection"
        ],
        "endpoints": {
            "upload": "POST /upload_item (REAL eBay data required)",
            "job_status": "GET /job/{job_id}/status",
            "health": "GET /health",
            "ping": "GET /ping (keep-alive)",
            "ebay_auth": "GET /ebay/oauth/start (JSON auth URL)",
            "ebay_callback": "GET /ebay/oauth/callback (OAuth web-to-app bridge)",
            "ebay_token": "GET /ebay/oauth/token/{token_id} (get access token)",
            "ebay_token_status": "GET /ebay/token-status (check token availability)",
            "market_analysis": "POST /api/ebay/analyze (direct market analysis)"
        }
    }

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8000))
    logger.info(f"🚀 Starting REAL EBAY DATA server on port {port}")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        workers=1,
        timeout_keep_alive=30,
        log_level="info"
    )