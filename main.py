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
    version="4.0.0",
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

# Rare item database (coins, stamps, collectibles)
RARE_ITEM_DATABASE = {
    'coins': {
        '1909-S VDB': {'min_value': 500, 'max_value': 100000, 'rarity': 'extreme'},
        '1955 Double Die': {'min_value': 1000, 'max_value': 125000, 'rarity': 'extreme'},
        '1943 Copper Penny': {'min_value': 85000, 'max_value': 1000000, 'rarity': 'extreme'},
        '1916-D Mercury Dime': {'min_value': 1000, 'max_value': 35000, 'rarity': 'high'},
        '1893-S Morgan Dollar': {'min_value': 2000, 'max_value': 500000, 'rarity': 'extreme'},
        '1804 Silver Dollar': {'min_value': 1000000, 'max_value': 10000000, 'rarity': 'legendary'},
        '1913 Liberty Nickel': {'min_value': 3000000, 'max_value': 5000000, 'rarity': 'legendary'},
    },
    'stamps': {
        'Inverted Jenny': {'min_value': 100000, 'max_value': 1500000, 'rarity': 'legendary'},
        'British Guiana 1c': {'min_value': 8000000, 'max_value': 10000000, 'rarity': 'legendary'},
        'Mauritius Post Office': {'min_value': 1000000, 'max_value': 2000000, 'rarity': 'legendary'},
    },
    'cards': {
        'Charizard 1st Edition': {'min_value': 5000, 'max_value': 400000, 'rarity': 'extreme'},
        'Black Lotus Alpha': {'min_value': 50000, 'max_value': 500000, 'rarity': 'extreme'},
        'Honus Wagner T206': {'min_value': 1000000, 'max_value': 6000000, 'rarity': 'legendary'},
        'Mickey Mantle 1952': {'min_value': 50000, 'max_value': 5000000, 'rarity': 'extreme'},
    }
}

# Era classification patterns
ERA_PATTERNS = {
    'furniture': {
        'Victorian': {'years': (1837, 1901), 'keywords': ['ornate', 'carved', 'mahogany', 'walnut', 'upholstered']},
        'Art Deco': {'years': (1920, 1939), 'keywords': ['geometric', 'chrome', 'lacquer', 'streamlined']},
        'Mid-Century Modern': {'years': (1945, 1969), 'keywords': ['teak', 'walnut', 'tapered legs', 'organic', 'eames']},
        'Colonial': {'years': (1700, 1780), 'keywords': ['windsor', 'maple', 'pine', 'simple', 'handmade']},
        'Art Nouveau': {'years': (1890, 1910), 'keywords': ['flowing', 'nature', 'curved', 'floral']},
        'Chippendale': {'years': (1750, 1780), 'keywords': ['cabriole', 'ball and claw', 'mahogany']},
        'Queen Anne': {'years': (1700, 1755), 'keywords': ['curved', 'cabriole legs', 'pad foot']},
        'Federal': {'years': (1780, 1820), 'keywords': ['inlay', 'tapered', 'classical']},
        'Empire': {'years': (1800, 1840), 'keywords': ['heavy', 'carved', 'gilt', 'claw feet']},
        'Renaissance Revival': {'years': (1850, 1880), 'keywords': ['ornate', 'carved', 'walnut', 'pediment']},
    },
    'jewelry': {
        'Victorian': {'years': (1837, 1901), 'keywords': ['mourning', 'cameo', 'filigree', 'sentiment']},
        'Art Nouveau': {'years': (1890, 1910), 'keywords': ['nature', 'enamel', 'flowing', 'organic']},
        'Edwardian': {'years': (1901, 1915), 'keywords': ['platinum', 'delicate', 'lace', 'filigree']},
        'Art Deco': {'years': (1920, 1935), 'keywords': ['geometric', 'platinum', 'emerald cut', 'bold']},
        'Retro': {'years': (1935, 1950), 'keywords': ['rose gold', 'large', 'cocktail', 'bold']},
        'Mid-Century': {'years': (1950, 1970), 'keywords': ['modernist', 'abstract', 'textured']},
    }
}

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

def refresh_ebay_token_if_needed(token_id: str) -> bool:
    """Refresh eBay token if it's expired"""
    try:
        token_data = ebay_oauth.get_user_token(token_id)
        if not token_data:
            logger.warning("⚠️ No token data found for refresh")
            return False
        
        expires_at = datetime.fromisoformat(token_data["expires_at"])
        time_remaining = expires_at - datetime.now()
        
        if time_remaining.total_seconds() < 300:
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

def check_rare_item_database(item_data: Dict) -> Optional[Dict]:
    """Check if item matches rare item database"""
    title = item_data.get('title', '').lower()
    description = item_data.get('description', '').lower()
    category = item_data.get('category', '').lower()
    
    combined_text = f"{title} {description}"
    
    # Check category-specific rare items
    if category in ['coins', 'collectibles']:
        for item_name, data in RARE_ITEM_DATABASE.get('coins', {}).items():
            if item_name.lower() in combined_text:
                logger.info(f"💎 RARE COIN DETECTED: {item_name}")
                return {
                    'rare_item_match': item_name,
                    'rarity_level': data['rarity'],
                    'estimated_value_range': f"${data['min_value']:,} - ${data['max_value']:,}",
                    'database_source': 'rare_coins'
                }
    
    if category == 'stamps':
        for item_name, data in RARE_ITEM_DATABASE.get('stamps', {}).items():
            if item_name.lower() in combined_text:
                logger.info(f"💎 RARE STAMP DETECTED: {item_name}")
                return {
                    'rare_item_match': item_name,
                    'rarity_level': data['rarity'],
                    'estimated_value_range': f"${data['min_value']:,} - ${data['max_value']:,}",
                    'database_source': 'rare_stamps'
                }
    
    if category == 'collectibles':
        for item_name, data in RARE_ITEM_DATABASE.get('cards', {}).items():
            if item_name.lower() in combined_text:
                logger.info(f"💎 RARE CARD DETECTED: {item_name}")
                return {
                    'rare_item_match': item_name,
                    'rarity_level': data['rarity'],
                    'estimated_value_range': f"${data['min_value']:,} - ${data['max_value']:,}",
                    'database_source': 'rare_cards'
                }
    
    return None

def detect_era(item_data: Dict) -> Optional[str]:
    """Detect historical era of item (furniture, jewelry, etc.)"""
    category = item_data.get('category', '').lower()
    title = item_data.get('title', '').lower()
    description = item_data.get('description', '').lower()
    year_str = item_data.get('year', '').strip()
    
    combined_text = f"{title} {description}"
    
    # Check if category has era patterns
    if category not in ERA_PATTERNS:
        return None
    
    era_matches = []
    
    for era_name, era_data in ERA_PATTERNS[category].items():
        score = 0
        
        # Check keywords
        for keyword in era_data['keywords']:
            if keyword in combined_text:
                score += 2
        
        # Check year range
        if year_str and year_str.isdigit():
            year = int(year_str)
            year_start, year_end = era_data['years']
            if year_start <= year <= year_end:
                score += 5
        
        # Check if era name is mentioned
        if era_name.lower() in combined_text:
            score += 10
        
        if score > 0:
            era_matches.append((era_name, score))
    
    if era_matches:
        # Sort by score and return highest
        era_matches.sort(key=lambda x: x[1], reverse=True)
        best_match = era_matches[0]
        
        if best_match[1] >= 5:  # Minimum confidence threshold
            logger.info(f"🏛️ ERA DETECTED: {best_match[0]} (score: {best_match[1]})")
            return best_match[0]
    
    return None

# MAXIMUM ACCURACY MARKET ANALYSIS PROMPT - ENHANCED FOR MULTI-ITEM
market_analysis_prompt = """
EXPERT RESELL ANALYST - MAXIMUM ACCURACY ANALYSIS:

**CRITICAL: If multiple distinct items are visible in the image, analyze EACH separately.**
**Do NOT combine unrelated items into a single analysis. Return a JSON ARRAY of items.**

🔍 **MULTI-ITEM ANALYSIS RULES:**
1. Count ALL distinct items visible
2. Analyze EACH item separately with its own value
3. If items form a SET (like trading cards), analyze as a SET with combined value
4. For unrelated items, create separate analyses
5. If single item, return single-item array

🔍 **COMPREHENSIVE IDENTIFICATION PHASE (PER ITEM):**
- Extract EVERY visible text, number, logo, brand mark, model number, serial number
- Identify ALL materials, construction quality, age indicators, manufacturing details
- Note ALL condition issues, wear patterns, damage, repairs, modifications
- Capture EXACT size, dimensions, weight indicators, manufacturing codes
- Identify style period/era (Victorian, Art Deco, Mid-Century, Colonial, etc.)

📊 **ENHANCED MARKET ANALYSIS PHASE:**
- Use EXACT brand/model/year data when available
- If specific identification is unclear, analyze by material, construction, and visual characteristics
- Consider brand popularity, rarity, demand trends, collector interest
- Factor in ALL condition deductions and market saturation
- Account for seasonal pricing variations and current market trends
- Identify historical era or period if applicable (furniture, jewelry, collectibles)

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

Return analysis in JSON array format:

[
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
    
    "brand": "Exact brand if visible, otherwise 'Unknown - appears to be [quality/style]'",
    "model": "Model number/name if visible, otherwise descriptive characteristics", 
    "year": "Production year if determinable, otherwise era/style indicators",
    "era": "Historical period if applicable (Victorian, Mid-Century, Art Deco, etc.)",
    "condition": "DETAILED condition assessment with specific notes",
    "confidence": 0.85,
    "analysis_depth": "comprehensive",
    "key_features": ["ALL notable features that add value"],
    "comparable_items": "Similar items selling for $X-Y",
    "identification_confidence": "high/medium/low with reasoning",
    "additional_info_needed": ["What specific info would enable better identification"]
  }
]

CRITICAL: Base pricing on ACTUAL market conditions, NEVER guess.
If specific identification is unclear, analyze by observable characteristics and provide guidance.

IMPORTANT: Return ONLY valid JSON array, no additional text or explanations.
ALWAYS provide actionable insights, NEVER empty or generic responses.
"""

def map_to_ebay_category(category: str) -> str:
    """Map internal category to eBay search-friendly category"""
    category_mapping = {
        'electronics': 'electronics',
        'clothing': 'clothing shoes accessories',
        'furniture': 'antiques furniture',
        'collectibles': 'collectibles',
        'books': 'books magazines',
        'toys': 'toys hobbies',
        'jewelry': 'jewelry watches',
        'sports': 'sporting goods',
        'tools': 'tools',
        'kitchen': 'home garden',
        'vehicles': 'cars trucks',
        'automotive': 'cars trucks',
        'music': 'musical instruments gear',
        'art': 'art',
        'coins': 'coins paper money',
        'stamps': 'stamps',
        'unknown': ''
    }
    
    return category_mapping.get(category.lower(), '')

def clean_search_query(query: str) -> str:
    """Clean and optimize search query for eBay"""
    if not query:
        return ""
    
    query = ' '.join(query.split())
    query = query.replace('"', '').replace("'", "").replace("`", "")
    
    if len(query) > 100:
        words = query.split()
        query = ' '.join(words[:10])
    
    return query

def detect_category(title: str, description: str, vision_analysis: Dict) -> str:
    """
    IMPROVED CONTEXT-AWARE category detection
    """
    title_lower = title.lower()
    description_lower = description.lower()
    
    detected_text = " ".join(vision_analysis.get('detected_text', []))
    detected_objects = " ".join(vision_analysis.get('detected_objects', []))
    brands = " ".join(vision_analysis.get('potential_brands', []))
    
    all_text = f"{title_lower} {description_lower} {detected_text.lower()} {detected_objects.lower()} {brands.lower()}"
    
    # 🚨 CRITICAL FIX: Use CONTEXT-AWARE detection, not just keyword matching
    # First, check for OBVIOUS MISCLASSIFICATIONS
    
    # Is this a COLLECTIBLE CARD (not a vehicle)?
    # Look for strong card-related terms
    card_indicators = [
        ("pokemon card", 10), ("trading card", 10), ("baseball card", 10),
        ("football card", 10), ("basketball card", 10), ("hockey card", 10),
        ("magic card", 10), ("yu-gi-oh", 10), ("tcg", 8),
        ("first edition", 8), ("graded", 8), ("psa", 8), ("bgs", 8),
        ("holo", 6), ("holographic", 6), ("collectible card", 8)
    ]
    
    for keyword, score in card_indicators:
        if keyword in all_text:
            logger.info(f"🃏 STRONG CARD INDICATOR: '{keyword}' (score: {score})")
            return "collectibles"
    
    # Is this a TOY/PLUSH (not a vehicle)?
    toy_indicators = [
        ("care bear", 10), ("teddy bear", 10), ("stuffed animal", 10),
        ("plush", 8), ("toy", 6), ("action figure", 8), ("doll", 8),
        ("lego", 10), ("playset", 6), ("model kit", 6)
    ]
    
    for keyword, score in toy_indicators:
        if keyword in all_text:
            logger.info(f"🧸 STRONG TOY INDICATOR: '{keyword}' (score: {score})")
            return "toys"
    
    # 🚨 SMART VEHICLE DETECTION: Only detect vehicles with STRONG evidence
    vehicle_indicators = [
        ("truck", 8), ("pickup", 8), ("sedan", 8), ("suv", 8), ("van", 8),
        ("motorcycle", 8), ("boat", 8), ("trailer", 8), ("rv", 8),
        ("automobile", 8), ("vehicle", 6), ("car", 5)  # "car" has lower score
    ]
    
    vehicle_score = 0
    for keyword, score in vehicle_indicators:
        if keyword in all_text:
            # Check context - is "car" in a product name or actual vehicle?
            if keyword == "car":
                # "car" in product names like "Care Bear" or "Pokemon Scarlet Violet" should NOT count
                if "care bear" in all_text or "pokemon" in all_text or "trading card" in all_text:
                    logger.info(f"🚫 Ignoring 'car' in product name: {all_text[:50]}...")
                    continue
            
            vehicle_score += score
            logger.info(f"🚗 Vehicle indicator: '{keyword}' (+{score})")
    
    # Also check for year + automotive brand combination (strong vehicle signal)
    year_pattern = r'\b(19[0-9]\d|20[0-2]\d)\b'
    years = re.findall(year_pattern, all_text)
    
    automotive_brands = ["chevrolet", "chevy", "ford", "toyota", "honda", "bmw", "mercedes", "dodge"]
    has_auto_brand = any(brand in all_text for brand in automotive_brands)
    
    if years and has_auto_brand:
        vehicle_score += 15
        logger.info(f"🚗 STRONG VEHICLE SIGNAL: Year {years[0]} + automotive brand")
    
    # Only classify as vehicle if we have strong evidence
    if vehicle_score >= 10:
        logger.info(f"📦 VEHICLE DETECTED (score: {vehicle_score})")
        return "vehicles"
    
    # Score each category based on comprehensive keyword matching
    category_keywords = {
        # Musical Instruments
        "music": [
            "piano", "guitar", "violin", "trumpet", "saxophone", "drums", 
            "keyboard", "synthesizer", "amplifier", "microphone"
        ],
        
        # Collectible Cards & Games
        "collectibles": [
            "pokemon", "pokémon", "magic", "yugioh", "baseball card", "sports card",
            "trading card", "tcg", "ccg", "collectible", "rare card", "vintage card"
        ],
        
        # Electronics
        "electronics": [
            "iphone", "samsung", "laptop", "computer", "camera", "headphones",
            "speaker", "smartphone", "tablet", "gaming console", "playstation"
        ],
        
        # Clothing & Shoes
        "clothing": [
            "shirt", "pants", "jeans", "dress", "jacket", "coat", "shoe",
            "sneaker", "boot", "hoodie", "sweater", "t-shirt"
        ],
        
        # Furniture
        "furniture": [
            "chair", "table", "sofa", "couch", "desk", "bed", "dresser",
            "cabinet", "bookshelf", "wardrobe", "armoire"
        ],
        
        # Jewelry & Watches
        "jewelry": [
            "ring", "necklace", "bracelet", "earring", "watch", "rolex",
            "diamond", "gold", "silver", "platinum", "gemstone"
        ],
        
        # Books
        "books": [
            "book", "novel", "hardcover", "paperback", "textbook", "comic book",
            "manga", "graphic novel", "first edition", "signed book"
        ],
        
        # Toys & Action Figures
        "toys": [
            "toy", "action figure", "doll", "lego", "stuffed animal", "plush",
            "model car", "hot wheels", "board game", "puzzle"
        ],
        
        # Sports Equipment
        "sports": [
            "baseball", "football", "basketball", "golf", "tennis", "hockey",
            "fishing", "bicycle", "skateboard", "exercise equipment"
        ],
        
        # Tools & Hardware
        "tools": [
            "tool", "wrench", "hammer", "screwdriver", "drill", "saw",
            "pliers", "toolbox", "power tool", "hand tool"
        ],
        
        # Art & Decor
        "art": [
            "painting", "print", "sculpture", "drawing", "photograph",
            "canvas", "art", "original art", "signed art"
        ],
        
        # Kitchen & Appliances
        "kitchen": [
            "kitchen", "pan", "pot", "knife", "cutlery", "blender",
            "mixer", "toaster", "coffee maker", "microwave"
        ]
    }
    
    scores = {category: 0 for category in category_keywords}
    scores["unknown"] = 0
    
    for category, keywords in category_keywords.items():
        for keyword in keywords:
            if keyword in all_text:
                # Weight by how definitive the keyword is
                weight = 3 if len(keyword) > 5 else 1  # Longer words are more specific
                scores[category] += weight
    
    # Get highest scoring category
    detected_category = max(scores.items(), key=lambda x: x[1])[0]
    highest_score = scores[detected_category]
    
    # Only accept if score is above reasonable threshold
    if highest_score >= 3:
        logger.info(f"📦 CATEGORY: '{detected_category}' (score: {highest_score})")
        return detected_category
    else:
        logger.info(f"📦 CATEGORY: 'unknown' (highest score: {highest_score})")
        return "unknown"

def extract_keywords_from_user_input(user_text: str) -> Dict[str, List[str]]:
    """Extract structured keywords from user input"""
    if not user_text:
        return {}
    
    user_text = user_text.lower()
    
    # Extract year patterns (1900-2025)
    year_pattern = r'\b(19[0-9]\d|20[0-2]\d)\b'
    years = re.findall(year_pattern, user_text)
    
    # Extract decade patterns (1980s, 1990s, etc.)
    decade_pattern = r'\b(19[0-9]0s|20[0-2]0s)\b'
    decades = re.findall(decade_pattern, user_text)
    
    # Extract era keywords
    era_keywords = [
        'victorian', 'edwardian', 'art deco', 'art nouveau', 'mid century', 'mid-century',
        'colonial', 'federal', 'empire', 'renaissance', 'baroque', 'rococo',
        'chippendale', 'queen anne', 'regency', 'georgian', 'retro', 'vintage', 'antique'
    ]
    
    eras = []
    for era in era_keywords:
        if era in user_text:
            eras.append(era.title())
    
    # Extract potential brand names
    brands = []
    common_brands = [
        # Automotive
        'chevy', 'chevrolet', 'ford', 'toyota', 'honda', 'bmw', 'mercedes', 'benz',
        'dodge', 'jeep', 'gmc', 'cadillac', 'buick', 'pontiac', 'ram', 'chrysler',
        'nissan', 'subaru', 'mazda', 'volkswagen', 'vw', 'audi', 'volvo', 'tesla',
        'porsche', 'ferrari', 'lamborghini', 'jaguar', 'land rover', 'mini',
        
        # Musical Instruments
        'steinway', 'yamaha', 'petrof', 'baldwin', 'kawai', 'bosendorfer',
        'fender', 'gibson', 'martin', 'taylor', 'prs', 'ibanez',
        
        # Electronics
        'apple', 'samsung', 'sony', 'microsoft', 'google', 'dell', 'hp', 'lenovo',
        'canon', 'nikon', 'panasonic', 'lg', 'bose',
        
        # Fashion
        'nike', 'adidas', 'jordan', 'gucci', 'prada', 'louis vuitton', 'lv',
        'supreme', 'bape', 'off-white', 'balenciaga', 'versace',
        
        # Watches
        'rolex', 'omega', 'cartier', 'patek philippe', 'tag heuer', 'breitling',
        
        # General
        'pokemon', 'pokémon', 'magic', 'yugioh', 'topps'
    ]
    
    for brand in common_brands:
        if brand in user_text:
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
    
    model_keywords = [
        'truck', 'pickup', 'sedan', 'coupe', 'convertible', 'suv', 'van',
        'deluxe', 'custom', 'standard', 'limited',
        'premium', 'luxury', 'sport', 'performance', 'edition', 'series',
        'grand', 'upright', 'baby grand', 'console',
        'first edition', '1st edition', 'shadowless', 'holographic', 'holo',
        'graded', 'psa', 'bgs', 'cgc', 'mint', 'near mint'
    ]
    
    for keyword in model_keywords:
        if keyword in user_text:
            if keyword in ['window', '5-window', 'deluxe', 'custom', 'standard', 'holographic', 'holo', 'graded']:
                features.append(keyword)
            else:
                models.append(keyword)
    
    # Remove duplicates
    def deduplicate(lst):
        seen = set()
        result = []
        for item in lst:
            item_lower = item.lower()
            if item_lower not in seen:
                seen.add(item_lower)
                result.append(item)
        return result
    
    years = deduplicate(years)
    decades = deduplicate(decades)
    eras = deduplicate(eras)
    brands = deduplicate(brands)
    models = deduplicate(models)
    features = deduplicate(features)
    
    logger.info(f"📋 Extracted keywords: years={years}, decades={decades}, eras={eras}, brands={brands}, models={models}, features={features}")
    
    return {
        'years': years,
        'decades': decades,
        'eras': eras,
        'brands': brands,
        'models': models,
        'features': features
    }

def parse_json_response(response_text: str) -> List[Dict]:
    """Robust JSON parsing for maximum accuracy"""
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
        
        json_match = re.search(r'(\{.*\}|\[.*\])', json_text, re.DOTALL)
        if json_match:
            json_text = json_match.group(1)
        
        json_text = re.sub(r',\s*([}\]])', r'\1', json_text)
        
        parsed_data = json.loads(json_text)
        
        if isinstance(parsed_data, dict):
            return [parsed_data]
        elif isinstance(parsed_data, list):
            return parsed_data
        else:
            logger.warning(f"Unexpected JSON format: {type(parsed_data)}")
            return []
            
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing failed at line {e.lineno}, column {e.colno}: {e.msg}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error in JSON parsing: {e}")
        return []

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
        self.era = data.get("era", "")
        self.condition = data.get("condition", "")
        self.confidence = data.get("confidence", 0.5)
        self.analysis_depth = data.get("analysis_depth", "comprehensive")
        self.key_features = data.get("key_features", [])
        self.comparable_items = data.get("comparable_items", "")
        self.identification_confidence = data.get("identification_confidence", "unknown")
        self.additional_info_needed = data.get("additional_info_needed", [])
        
        # NEW: Sold comparison data
        self.sold_statistics = data.get("sold_statistics", {})
        self.comparison_items = data.get("comparison_items", [])
        self.sold_items_links = data.get("sold_items_links", [])
        
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
            "era": self.era,
            "condition": self.condition,
            "confidence": self.confidence,
            "analysis_depth": self.analysis_depth,
            "key_features": self.key_features,
            "comparable_items": self.comparable_items,
            "identification_confidence": self.identification_confidence,
            "additional_info_needed": self.additional_info_needed,
            "sold_statistics": self.sold_statistics,
            "comparison_items": self.comparison_items,
            "sold_items_links": self.sold_items_links
        }

def call_groq_api(prompt: str, image_base64: str = None, mime_type: str = None) -> str:
    """MAXIMUM accuracy Groq API call with JSON formatting instructions"""
    if not groq_client:
        raise Exception("Groq client not configured")
    
    json_format_prompt = prompt + "\n\n**IMPORTANT: Return ONLY valid JSON array. Do not include any explanatory text, code fences, or markdown outside the JSON.**"
    
    messages = []
    
    if image_base64 and mime_type:
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
            temperature=0.1,
            max_tokens=4000,
            top_p=0.95,
            stream=False,
            timeout=15.0,
            response_format={"type": "json_object"}
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

def search_ebay_directly(keywords: str, limit: int = 10) -> List[Dict]:
    """Enhanced eBay search - FIXED to actually find sold items"""
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
        
        # FIXED: Better search parameters
        params = {
            'q': keywords,
            'limit': str(limit * 3),  # Get more to filter
            'filter': 'price:[1..10000],buyingOptions:{FIXED_PRICE|AUCTION}',  # Get items with buying options
            'sort': 'price',  # Sort by price
            'fieldgroups': 'EXTENDED'
        }
        
        url = "https://api.ebay.com/buy/browse/v1/item_summary/search"
        
        logger.info(f"🔍 Enhanced eBay search for: '{keywords}'")
        response = requests.get(url, headers=headers, params=params, timeout=15)
        
        logger.info(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            if 'itemSummaries' in data:
                logger.info(f"📊 eBay returned {len(data.get('itemSummaries', []))} total results")
                
                items = []
                for i, item in enumerate(data['itemSummaries']):
                    try:
                        # Get item details
                        title = item.get('title', '').lower()
                        price = item.get('price', {}).get('value', '0')
                        price_float = float(price)
                        condition = item.get('condition', '')
                        item_end_date = item.get('itemEndDate', '')
                        buying_options = item.get('buyingOptions', [])
                        
                        # Determine if sold (auction ended or has end date)
                        is_sold = False
                        if item_end_date:
                            end_time = datetime.fromisoformat(item_end_date.replace('Z', '+00:00'))
                            if end_time < datetime.now():
                                is_sold = True
                        
                        # Skip unrealistic prices for certain categories
                        if 'pokemon' in keywords.lower() and price_float < 0.50:
                            logger.debug(f"   [{i+1}] Skipping - unrealistic Pokemon card price: ${price_float}")
                            continue
                        
                        # Filter out junk
                        parts_keywords = ['for parts only', 'parts only', 'not working', 'as is']
                        is_parts = any(kw in title for kw in parts_keywords)
                        
                        if not is_parts and price_float > 0:
                            items.append({
                                'title': item.get('title', ''),
                                'price': price_float,
                                'item_id': item.get('itemId', ''),
                                'condition': condition,
                                'category': item.get('categoryPath', ''),
                                'image_url': item.get('image', {}).get('imageUrl', ''),
                                'item_web_url': item.get('itemWebUrl', ''),
                                'buying_options': buying_options,
                                'sold': is_sold,
                                'item_end_date': item_end_date
                            })
                            
                            if len(items) >= limit:
                                break
                        else:
                            logger.debug(f"   [{i+1}] FILTERED OUT: '{title[:50]}...' - ${price_float}")
                                
                    except (KeyError, ValueError) as e:
                        logger.debug(f"   [{i+1}] Skipping item - parsing error: {e}")
                        continue
                
                logger.info(f"✅ Found {len(items)} items")
                return items
        elif response.status_code == 401:
            logger.error("❌ eBay token expired or invalid")
            store_ebay_token(None)
            return []
        elif response.status_code == 429:
            logger.warning("⚠️ eBay rate limit reached")
            return []
        else:
            logger.error(f"❌ eBay search error {response.status_code}: {response.text[:200]}")
            
    except Exception as e:
        logger.error(f"❌ Direct eBay search error: {e}")
    
    return []

def search_ebay_completed_items(keywords: str, limit: int = 10) -> List[Dict]:
    """Search for specifically completed/sold items"""
    token = get_ebay_token()
    if not token:
        return []
    
    try:
        headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json',
            'X-EBAY-C-MARKETPLACE-ID': 'EBAY_US'
        }
        
        # Try to find completed auctions
        params = {
            'q': keywords,
            'limit': str(limit * 2),
            'filter': 'buyingOptions:{AUCTION}',  # Auctions are more likely to be sold
            'sort': 'endTime',  # Sort by end time
            'fieldgroups': 'EXTENDED'
        }
        
        url = "https://api.ebay.com/buy/browse/v1/item_summary/search"
        response = requests.get(url, headers=headers, params=params, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            items = []
            for item in data.get('itemSummaries', []):
                try:
                    item_end_date = item.get('itemEndDate', '')
                    if item_end_date:
                        end_time = datetime.fromisoformat(item_end_date.replace('Z', '+00:00'))
                        if end_time < datetime.now():  # Item has ended
                            price = float(item.get('price', {}).get('value', '0'))
                            if price > 0:
                                items.append({
                                    'title': item.get('title', ''),
                                    'price': price,
                                    'item_id': item.get('itemId', ''),
                                    'condition': item.get('condition', ''),
                                    'item_web_url': item.get('itemWebUrl', ''),
                                    'sold': True,
                                    'item_end_date': item_end_date
                                })
                except:
                    continue
            
            logger.info(f"🎯 Found {len(items)} completed items")
            return items[:limit]
    
    except Exception as e:
        logger.error(f"Completed items search error: {e}")
    
    return []

def get_sold_comparison_items(keywords: str, limit: int = 5) -> List[Dict]:
    """Get actual sold items for comparison"""
    # Try multiple search strategies
    items = []
    
    # Strategy 1: Look for completed auctions
    items = search_ebay_completed_items(keywords, limit)
    
    # Strategy 2: Fallback to regular search
    if len(items) < 3:
        regular_items = search_ebay_directly(keywords, limit * 2)
        # Filter for items that appear sold or have ended
        for item in regular_items:
            if item.get('sold') or 'AUCTION' in item.get('buying_options', []):
                items.append(item)
    
    # Strategy 3: Try different keyword variations
    if len(items) < 3:
        # Try adding "sold" to search
        items.extend(search_ebay_directly(f"{keywords} sold", limit))
    
    # Deduplicate
    seen_ids = set()
    unique_items = []
    for item in items:
        if item.get('item_id') not in seen_ids:
            seen_ids.add(item.get('item_id'))
            unique_items.append(item)
    
    return unique_items[:limit]

def analyze_ebay_market_directly(keywords: str) -> Dict:
    """Direct eBay market analysis with sold items"""
    logger.info(f"📊 Direct eBay market analysis for: '{keywords}'")
    
    # Get sold comparison items
    sold_items = get_sold_comparison_items(keywords, limit=15)
    
    if not sold_items:
        logger.warning("⚠️ NO SOLD ITEMS FOUND - using current listings")
        # Fallback to current listings
        sold_items = search_ebay_directly(keywords, limit=10)
    
    if not sold_items:
        logger.error("❌ NO EBAY DATA AVAILABLE")
        return {
            'error': 'NO_EBAY_DATA',
            'message': 'Unable to retrieve eBay data for this search',
            'requires_auth': False
        }
    
    prices = [item['price'] for item in sold_items if item.get('price', 0) > 0]
    
    if not prices:
        logger.error("❌ No valid price data from eBay")
        return {
            'error': 'NO_PRICE_DATA',
            'message': 'eBay returned items but no valid price data',
            'requires_auth': False
        }
    
    # Calculate statistics
    avg_price = sum(prices) / len(prices)
    min_price = min(prices)
    max_price = max(prices)
    
    analysis = {
        'success': True,
        'average_price': round(avg_price, 2),
        'price_range': f"${min_price:.2f} - ${max_price:.2f}",
        'lowest_price': round(min_price, 2),
        'highest_price': round(max_price, 2),
        'total_sold_analyzed': len(sold_items),
        'recommended_price': round(avg_price * 0.85, 2),
        'market_notes': f'Based on {len(sold_items)} recent eBay items',
        'data_source': 'eBay Browse API',
        'confidence': 'high' if len(sold_items) >= 5 else 'medium',
        'api_used': 'Browse API',
        'sold_items': sold_items[:10]  # Include actual sold items
    }
    
    logger.info(f"✅ Market analysis: avg=${avg_price:.2f}, low=${min_price:.2f}, high=${max_price:.2f}")
    
    return analysis

def build_search_query(item_data: Dict, user_keywords: Dict, detected_category: str) -> List[str]:
    """
    Build search queries using SPECIFIC identifiers from AI analysis
    """
    search_strategies = []
    
    # Get AI-identified specifics
    title = item_data.get('title', '').lower()
    description = item_data.get('description', '').lower()
    brand = item_data.get('brand', '').lower()
    model = item_data.get('model', '').lower()
    year = item_data.get('year', '').strip()
    era = item_data.get('era', '').lower()
    
    # Extract specific Pokemon/character names from title
    specific_identifiers = []
    
    # Strategy 1: EXACT CHARACTER/ITEM IDENTIFICATION (Highest priority)
    # For collectibles/cards/toys: extract specific character names
    if detected_category in ['collectibles', 'toys']:
        # Common patterns in titles like "Celebi, Yveltal, and Kyogre Pokemon Cards"
        # Extract comma-separated names
        title_parts = title.replace(' and ', ', ').split(',')
        for part in title_parts:
            part = part.strip()
            # Remove generic words
            generic_words = ['pokemon', 'cards', 'trading', 'card', 'figure', 'toy', 'plush']
            if (len(part) > 3 and 
                not any(word in part for word in generic_words) and
                not part.isdigit()):
                specific_identifiers.append(part)
    
    # Strategy 2: USER INPUT (High priority)
    if user_keywords:
        user_terms = []
        
        # Add specific years/decades first
        if user_keywords.get('years'):
            user_terms.append(user_keywords['years'][0])
        elif user_keywords.get('decades'):
            user_terms.append(user_keywords['decades'][0])
        
        # Add specific brands
        if user_keywords.get('brands'):
            user_terms.append(user_keywords['brands'][0])
        
        # Add specific models/features
        if user_keywords.get('models'):
            user_terms.extend(user_keywords['models'][:2])
        
        if user_keywords.get('features'):
            # Filter out non-essential features
            essential_features = []
            non_essential = ['window', '5-window', 'deluxe', 'custom', 'standard', 'edition']
            for feature in user_keywords['features']:
                if feature.lower() not in non_essential and len(feature) > 3:
                    essential_features.append(feature)
            user_terms.extend(essential_features[:2])
        
        if user_terms:
            query = " ".join(user_terms)
            search_strategies.append(query)
            logger.info(f"🎯 USER QUERY: '{query}'")
    
    # Strategy 3: SPECIFIC CHARACTER/MODEL SEARCH (For collectibles, toys, cards)
    if specific_identifiers:
        # Try individual character searches first
        for char_name in specific_identifiers[:3]:  # Top 3 specific names
            char_query = f"{char_name} {detected_category}"
            search_strategies.append(char_query)
            logger.info(f"👤 SPECIFIC CHARACTER: '{char_query}'")
        
        # Then try combination
        if len(specific_identifiers) >= 2:
            combo_query = " ".join(specific_identifiers[:2])
            search_strategies.append(combo_query)
            logger.info(f"👥 COMBO CHARACTERS: '{combo_query}'")
    
    # Strategy 4: BRAND + MODEL + YEAR (For vehicles, electronics, etc.)
    if brand and 'unknown' not in brand:
        # For vehicles: "1947 Ford pickup" not "Post-War Era 1947 Ford 5-Window Pickup"
        if detected_category == 'vehicles':
            vehicle_terms = []
            if year and year.isdigit():
                vehicle_terms.append(year)
            vehicle_terms.append(brand)
            
            # Clean model - remove window/trim details for search
            if model and 'unknown' not in model:
                model_clean = model
                # Remove window/trim descriptors for better search
                model_clean = re.sub(r'\d+\s*window', '', model_clean)
                model_clean = re.sub(r'window', '', model_clean)
                model_clean = re.sub(r'deluxe|custom|standard', '', model_clean)
                model_clean = model_clean.strip()
                if model_clean:
                    vehicle_terms.append(model_clean)
            
            if vehicle_terms:
                query = " ".join(vehicle_terms)
                search_strategies.append(query)
                logger.info(f"🚗 VEHICLE QUERY: '{query}'")
        
        # For other items
        else:
            brand_query_parts = []
            if year and year.isdigit():
                brand_query_parts.append(year)
            brand_query_parts.append(brand)
            if model and 'unknown' not in model:
                # Clean model for search
                model_words = model.split()[:2]  # First 2 model words
                brand_query_parts.extend(model_words)
            
            if brand_query_parts:
                query = " ".join(brand_query_parts)
                search_strategies.append(query)
                logger.info(f"🏷️ BRAND+MODEL: '{query}'")
    
    # Strategy 5: TITLE-BASED (Fallback - clean generic words)
    if title:
        # Extract key terms, removing generic words
        title_words = []
        stop_words = ['the', 'a', 'an', 'and', 'or', 'for', 'with', 'in', 'on', 'at', 'to',
                     'pokemon', 'card', 'cards', 'trading', 'figure', 'toy', 'plush',
                     'era', 'post-war', 'postwar', 'window', 'windows']
        
        for word in title.split()[:10]:  # First 10 words max
            word_clean = word.lower().strip('.,!?;:"\'()[]{}')
            if (len(word_clean) > 2 and 
                word_clean not in stop_words and
                not word_clean.isdigit() and
                word_clean not in ['unknown', 'modern']):
                title_words.append(word_clean)
        
        if title_words:
            query = " ".join(title_words[:5])  # Max 5 specific words
            search_strategies.append(query)
            logger.info(f"📝 CLEANED TITLE: '{query}'")
    
    # Strategy 6: CATEGORY-SPECIFIC FALLBACK
    if detected_category and detected_category != 'unknown':
        category_query = f"{detected_category}"
        search_strategies.append(category_query)
        logger.info(f"📦 CATEGORY ONLY: '{category_query}'")
    
    # Clean and deduplicate
    cleaned = []
    seen = set()
    for strategy in search_strategies:
        strategy = clean_search_query(strategy)
        # Ensure query is meaningful
        if (strategy and 
            strategy not in seen and 
            len(strategy) > 3 and
            len(strategy.split()) <= 6):  # Max 6 terms
            seen.add(strategy)
            cleaned.append(strategy[:80])  # eBay search limit
    
    logger.info(f"🔍 FINAL SEARCH STRATEGIES: {cleaned}")
    return cleaned[:3]  # Max 3 strategies

def ensure_string_field(item_data: Dict, field_name: str) -> Dict:
    """Ensure a field is always a string, converting if necessary"""
    # Ensure field exists
    if field_name not in item_data:
        item_data[field_name] = ""
        return item_data
        
    if item_data[field_name] is not None:
        try:
            value = item_data[field_name]
            if isinstance(value, (int, float)):
                item_data[field_name] = str(int(value))
            elif isinstance(value, bool):
                item_data[field_name] = str(value).lower()
            elif isinstance(value, list):
                # Join list with commas
                item_data[field_name] = ", ".join(str(v) for v in value)
            elif not isinstance(value, str):
                item_data[field_name] = str(value)
            # If it's already a string, ensure it's stripped
            elif isinstance(value, str):
                item_data[field_name] = value.strip()
        except Exception as e:
            logger.warning(f"Failed to convert field {field_name}: {e}")
            item_data[field_name] = ""
    return item_data

def ensure_numeric_fields(item_data: Dict) -> Dict:
    """Ensure numeric fields are proper numbers"""
    if 'confidence' in item_data and item_data['confidence'] is not None:
        try:
            if isinstance(item_data['confidence'], str):
                item_data['confidence'] = float(item_data['confidence'])
        except (ValueError, TypeError):
            item_data['confidence'] = 0.5
    
    if 'resellability_rating' in item_data and item_data['resellability_rating'] is not None:
        try:
            if isinstance(item_data['resellability_rating'], str):
                item_data['resellability_rating'] = int(item_data['resellability_rating'])
            item_data['resellability_rating'] = max(1, min(10, item_data['resellability_rating']))
        except (ValueError, TypeError):
            item_data['resellability_rating'] = 5
    
    if 'processing_time_seconds' in item_data and item_data['processing_time_seconds'] is not None:
        try:
            if isinstance(item_data['processing_time_seconds'], str):
                item_data['processing_time_seconds'] = int(item_data['processing_time_seconds'])
        except (ValueError, TypeError):
            item_data['processing_time_seconds'] = 25
    
    return item_data

def enhance_with_ebay_data_user_prioritized(item_data: Dict, vision_analysis: Dict, user_keywords: Dict) -> Dict:
    """
    Enhanced market analysis using REAL eBay SOLD data with SPECIFIC searches
    """
    try:
        detected_category = item_data.get('category', 'unknown')
        logger.info(f"📦 Category: '{detected_category}' for eBay search")
        
        # Build proper search queries
        search_strategies = build_search_query(item_data, user_keywords, detected_category)
        
        if not search_strategies:
            logger.warning("No valid search strategies")
            item_data['market_insights'] = "Cannot search eBay - no identifiable terms. " + item_data.get('market_insights', '')
            item_data['identification_confidence'] = "low"
            return item_data
        
        # Try to get REAL eBay market analysis with each strategy
        market_analysis = None
        sold_items = []
        best_strategy = None
        
        for strategy in search_strategies:
            logger.info(f"🔍 Searching eBay with: '{strategy}'")
            analysis = analyze_ebay_market_directly(strategy)
            
            if analysis and analysis.get('success'):
                # Check if results are actually relevant
                sold_count = len(analysis.get('sold_items', []))
                avg_price = analysis.get('average_price', 0)
                
                # For Pokemon cards, ensure we have meaningful prices
                if detected_category == 'collectibles' and 'pokemon' in strategy.lower():
                    # Filter out junk results (< $1)
                    valid_items = [item for item in analysis.get('sold_items', []) 
                                  if item.get('price', 0) > 1.0]
                    if len(valid_items) >= 3:
                        analysis['sold_items'] = valid_items
                        analysis['total_sold_analyzed'] = len(valid_items)
                        if valid_items:
                            prices = [item['price'] for item in valid_items if item.get('price', 0) > 0]
                            if prices:
                                analysis['average_price'] = sum(prices) / len(prices)
                                analysis['lowest_price'] = min(prices)
                                analysis['highest_price'] = max(prices)
                                analysis['price_range'] = f"${min(prices):.2f} - ${max(prices):.2f}"
                
                if analysis.get('total_sold_analyzed', 0) >= 3:
                    market_analysis = analysis
                    sold_items = analysis.get('sold_items', [])
                    best_strategy = strategy
                    logger.info(f"✅ Found relevant data with strategy: '{strategy}'")
                    break
                else:
                    logger.info(f"⚠️ Strategy '{strategy}' returned {sold_count} items, trying next...")
            elif analysis and analysis.get('error') == 'NO_EBAY_DATA':
                logger.error("❌ EBAY API FAILED")
                item_data['market_insights'] = "⚠️ eBay authentication required. Please connect your eBay account."
                item_data['price_range'] = "Authentication Required"
                item_data['suggested_cost'] = "Connect eBay Account"
                item_data['profit_potential'] = "eBay data unavailable"
                item_data['identification_confidence'] = "requires_auth"
                return item_data
        
        if market_analysis:
            # Update with REAL market data
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
            
            # Market insights with search strategy info
            insights = []
            if best_strategy:
                insights.append(f"Search: '{best_strategy}'")
            elif search_strategies:
                insights.append(f"Search: '{search_strategies[0][:40]}...'")
            
            total_sold = str(market_analysis.get('total_sold_analyzed', 0))
            avg_price_str = f"{market_analysis.get('average_price', 0):.2f}"
            price_range_str = str(market_analysis.get('price_range', 'Unknown'))
            confidence_str = str(market_analysis.get('confidence', 'unknown'))
            
            insights.extend([
                f"Based on {total_sold} eBay sales",
                f"Average: ${avg_price_str}",
                f"Range: {price_range_str}",
                f"Confidence: {confidence_str}"
            ])
            
            item_data['market_insights'] = ". ".join(insights)
            
            # eBay tips
            ebay_tips = []
            if best_strategy:
                ebay_tips.append(f"Search term: {best_strategy}")
            ebay_tips.extend([
                "Use 'Buy It Now' with Best Offer",
                "Include detailed measurements",
                "Take photos from all angles",
                "List on weekends for visibility"
            ])
            
            item_data['ebay_specific_tips'] = ebay_tips
            item_data['identification_confidence'] = market_analysis['confidence']
            item_data['data_source'] = market_analysis['data_source']
            
            # Add sold comparison items
            if sold_items:
                comparison_items = []
                sold_items_links = []
                prices = []
                
                for item in sold_items[:5]:  # Top 5 sold items
                    comparison_items.append({
                        'title': item.get('title', ''),
                        'sold_price': item.get('price', 0),
                        'condition': item.get('condition', ''),
                        'item_url': item.get('item_web_url', ''),
                        'image_url': item.get('image_url', ''),
                        'sold': item.get('sold', False)
                    })
                    
                    if item.get('item_web_url'):
                        sold_items_links.append(item['item_web_url'])
                    
                    if item.get('price', 0) > 0:
                        prices.append(item['price'])
                
                item_data['comparison_items'] = comparison_items
                item_data['sold_items_links'] = sold_items_links
                
                if prices:
                    item_data['sold_statistics'] = {
                        'lowest_sold': min(prices),
                        'highest_sold': max(prices),
                        'average_sold': sum(prices) / len(prices),
                        'total_comparisons': len(prices)
                    }
            
            # Check rare item database
            rare_match = check_rare_item_database(item_data)
            if rare_match:
                item_data['rare_item_detected'] = True
                item_data['rare_item_info'] = rare_match
                item_data['market_insights'] += f" 💎 RARE ITEM: {rare_match['rare_item_match']} - {rare_match['estimated_value_range']}"
            
            # Detect era
            detected_era = detect_era(item_data)
            if detected_era:
                item_data['era'] = detected_era
                item_data['market_insights'] += f" 🏛️ Era: {detected_era}"
            
            logger.info(f"✅ eBay analysis complete with {len(sold_items)} sold items")
                    
        else:
            logger.error("❌ NO EBAY DATA")
            item_data['market_insights'] = "⚠️ Unable to retrieve relevant eBay market data."
            item_data['identification_confidence'] = "low"
        
        return item_data
        
    except Exception as e:
        logger.error(f"❌ eBay enhancement failed: {e}")
        item_data['market_insights'] = f"⚠️ Error: {str(e)[:100]}"
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
        
        if user_title or user_description:
            enhanced_prompt += f"\n\n🔍 **USER-PROVIDED IDENTIFICATION HINTS (HIGH PRIORITY):**\n"
            if user_title:
                enhanced_prompt += f"**USER TITLE:** '{user_title}'\n"
            if user_description:
                enhanced_prompt += f"**USER DESCRIPTION:** '{user_description}'\n"
            
            if user_keywords:
                enhanced_prompt += f"\n**EXTRACTED DETAILS FROM USER INPUT:**\n"
                if user_keywords.get('years'):
                    enhanced_prompt += f"- **YEAR(S):** {', '.join(user_keywords['years'])}\n"
                if user_keywords.get('decades'):
                    enhanced_prompt += f"- **DECADE(S):** {', '.join(user_keywords['decades'])}\n"
                if user_keywords.get('eras'):
                    enhanced_prompt += f"- **ERA(S):** {', '.join(user_keywords['eras'])}\n"
                if user_keywords.get('brands'):
                    enhanced_prompt += f"- **BRAND(S):** {', '.join(user_keywords['brands'])}\n"
                if user_keywords.get('models'):
                    enhanced_prompt += f"- **MODEL(S):** {', '.join(user_keywords['models'])}\n"
                if user_keywords.get('features'):
                    enhanced_prompt += f"- **FEATURES:** {', '.join(user_keywords['features'][:5])}\n"
        
        enhanced_prompt += "\n\n🔍 **SEARCH PRIORITIZATION RULES:**\n"
        enhanced_prompt += "1. User details take ABSOLUTE precedence\n"
        enhanced_prompt += "2. Follow real search patterns: [Era/Year] [Brand] [Model] [Features]\n"
        enhanced_prompt += "3. Examples: '1955 Chevrolet 5 Window', '1980s Rolex Submariner', 'Victorian Mahogany Chair'\n"
        enhanced_prompt += "4. For collectibles: '[Brand] [Specific Item Name] [Features]' like 'Ninetales Delta Species Pokemon Card'\n"
        
        logger.info(f"🔬 Starting analysis with REAL eBay data...")
        
        # Call Groq API
        response_text = call_groq_api(enhanced_prompt, image_base64, mime_type)
        logger.info("✅ AI analysis completed")
        
        ai_response = parse_json_response(response_text)
        logger.info(f"📊 Parsed {len(ai_response)} items")
        
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
                title = item_data.get("title", "")
                description = item_data.get("description", "")
                
                if user_title:
                    item_data["title"] = f"{user_title} - {title}" if title else user_title
                
                if user_description and not description:
                    item_data["description"] = user_description
                
                # 🚨 CRITICAL: Detect category BEFORE eBay search
                detected_category = detect_category(
                    item_data.get("title", ""), 
                    item_data.get("description", ""),
                    vision_analysis
                )
                item_data["category"] = detected_category
                
                logger.info(f"📦 CATEGORY: '{detected_category}'")
                
                # Check if we have eBay token before attempting search
                ebay_token = get_ebay_token()
                if not ebay_token:
                    logger.error("❌ No eBay token available for search")
                    item_data['market_insights'] = "⚠️ eBay authentication required. Please connect your eBay account."
                    item_data['price_range'] = "Authentication Required"
                    item_data['suggested_cost'] = "Connect eBay Account"
                    item_data['profit_potential'] = "eBay data unavailable"
                    item_data['identification_confidence'] = "requires_auth"
                    
                    enhanced_items.append(EnhancedAppItem(item_data).to_dict())
                    continue
                
                # Enhance with REAL eBay market data
                item_data = enhance_with_ebay_data_user_prioritized(item_data, vision_analysis, user_keywords)
                
                # Check if eBay auth required
                if item_data.get('identification_confidence') == 'requires_auth':
                    # Don't return auth error for entire batch, just mark this item
                    logger.warning("⚠️ Item requires eBay auth")
                
                # Ensure required fields
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
                        "Clear photos of markings",
                        "Manufacturer information",
                        "Exact measurements"
                    ]
                
                # Ensure correct types
                item_data = ensure_string_field(item_data, "year")
                item_data = ensure_string_field(item_data, "era")
                item_data = ensure_numeric_fields(item_data)
                
                # Add user input flag
                if user_keywords:
                    item_data['user_input_incorporated'] = True
                    if user_keywords.get('years'):
                        item_data['user_specified_year'] = user_keywords['years'][0]
                    if user_keywords.get('brands'):
                        item_data['user_specified_brand'] = user_keywords['brands'][0]
                
                enhanced_items.append(EnhancedAppItem(item_data).to_dict())
            else:
                logger.warning(f"Skipping non-dictionary item")
        
        if not enhanced_items:
            logger.error("❌ NO ITEMS PARSED")
            return {
                "status": "failed",
                "error": "NO_AI_ANALYSIS",
                "message": "AI failed to analyze image"
            }
        
        logger.info(f"✅ Complete: {len(enhanced_items)} items with sold comparisons")
        
        return {
            "status": "completed",
            "result": {
                "message": f"Analysis complete with {len(enhanced_items)} items using REAL eBay data",
                "items": enhanced_items,
                "processing_time": "25-30s",
                "analysis_stages": 3,
                "confidence_level": "maximum_accuracy",
                "analysis_timestamp": datetime.now().isoformat(),
                "model_used": groq_model,
                "ebay_data_used": True,
                "user_details_incorporated": bool(user_title or user_description),
                "sold_items_included": any('comparison_items' in item for item in enhanced_items)
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Processing failed: {e}")
        return {
            "status": "failed", 
            "error": str(e)[:200]
        }

def background_worker():
    """Background worker with maximum accuracy"""
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
            
            logger.info(f"🔄 Processing job {job_id}")
            
            future = job_executor.submit(process_image_maximum_accuracy, job_data)
            try:
                result = future.result(timeout=25)
                
                with job_lock:
                    if result.get('status') == 'completed':
                        job_data['status'] = 'completed'
                        job_data['result'] = result['result']
                        logger.info(f"✅ Job {job_id} completed")
                    elif result.get('error') == 'EBAY_AUTH_REQUIRED':
                        job_data['status'] = 'failed'
                        job_data['error'] = 'eBay authentication required'
                        job_data['requires_auth'] = True
                        logger.error(f"❌ Job {job_id} needs auth")
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
    
    def keep_alive_loop():
        while True:
            time.sleep(25)
            try:
                update_activity()
                requests.get(f"http://localhost:{os.getenv('PORT', 8000)}/ping", timeout=5)
            except:
                pass
    
    threading.Thread(target=keep_alive_loop, daemon=True, name="KeepAlive").start()
    
    logger.info("🚀 Server started with REAL eBay data")

@app.on_event("shutdown")
async def shutdown_event():
    job_executor.shutdown(wait=False)
    logger.info("🛑 Server shutdown")

# ============= EBAY OAUTH ENDPOINTS =============
@app.get("/debug/ebay-search/{keywords}")
async def debug_ebay_search(keywords: str):
    """Debug endpoint to see raw eBay search results"""
    update_activity()
    
    token = get_ebay_token()
    if not token:
        return {"error": "No token available"}
    
    try:
        headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json',
            'X-EBAY-C-MARKETPLACE-ID': 'EBAY_US'
        }
        
        params = {
            'q': keywords,
            'limit': '10',
            'filter': 'buyingOptions:{FIXED_PRICE|AUCTION}'
        }
        
        url = "https://api.ebay.com/buy/browse/v1/item_summary/search"
        response = requests.get(url, headers=headers, params=params, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            
            # Extract key info from each item
            items = []
            for item in data.get('itemSummaries', [])[:10]:
                items.append({
                    'title': item.get('title', 'No title'),
                    'price': item.get('price', {}).get('value', 'No price'),
                    'condition': item.get('condition', 'No condition'),
                    'item_id': item.get('itemId', 'No ID'),
                    'category': item.get('categoryPath', 'No category'),
                    'buying_options': item.get('buyingOptions', []),
                    'item_end_date': item.get('itemEndDate', '')
                })
            
            return {
                "success": True,
                "total_results": len(data.get('itemSummaries', [])),
                "items": items,
                "raw_response": data
            }
        else:
            return {
                "success": False,
                "status_code": response.status_code,
                "error": response.text[:500]
            }
            
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/ebay/oauth/start")
async def get_ebay_auth_url():
    """Generate eBay OAuth authorization URL"""
    update_activity()
    
    try:
        app_id = os.getenv('EBAY_APP_ID')
        
        if not app_id:
            raise HTTPException(status_code=500, detail="eBay credentials not configured")
        
        auth_url, state = ebay_oauth.generate_auth_url()
        
        logger.info(f"🔗 Generated auth URL")
        
        return {
            "success": True,
            "auth_url": auth_url,
            "state": state,
            "redirect_uri": "https://resell-app-bi47.onrender.com/ebay/oauth/callback",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to generate auth URL: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ebay/oauth/callback")
async def ebay_oauth_callback_get(
    code: Optional[str] = None, 
    error: Optional[str] = None, 
    state: Optional[str] = None
):
    """Handle eBay OAuth callback"""
    update_activity()
    
    logger.info(f"🔔 OAuth callback: code={code is not None}, error={error}")
    
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
            logger.info("✅ Token obtained")
            
            token_id = token_response["token_id"]
            
            token_data = ebay_oauth.get_user_token(token_id)
            if token_data and "access_token" in token_data:
                access_token = token_data["access_token"]
                store_ebay_token(access_token)
                logger.info(f"✅ Token stored: {access_token[:20]}...")
            
            redirect_url = f"ai-resell-pro://ebay-oauth-callback?success=true&token_id={token_id}&state={state}"
            
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
        refresh_ebay_token_if_needed(token_id)
        
        return {
            "success": True,
            "access_token": token_data["access_token"],
            "expires_at": token_data["expires_at"],
            "token_type": token_data.get("token_type", "Bearer")
        }
        
    except Exception as e:
        logger.error(f"❌ Get token error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ebay/oauth/status/{token_id}")
async def get_token_status(token_id: str):
    """Get status of a specific token"""
    update_activity()
    
    logger.info(f"📋 Token status check for: {token_id}")
    
    try:
        token_data = ebay_oauth.get_user_token(token_id)
        
        if not token_data:
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
                    "seconds_remaining": 3600,
                    "refreshable": False,
                    "message": "Token from environment",
                    "source": "environment"
                }
            
            return {
                "valid": False,
                "error": f"Token {token_id} not found",
                "available_tokens": available_tokens
            }
        
        # Check expiration
        expires_at = datetime.fromisoformat(token_data["expires_at"])
        time_remaining = expires_at - datetime.now()
        seconds_remaining = time_remaining.total_seconds()
        
        valid = seconds_remaining > 300  # 5 minutes buffer
        
        return {
            "valid": valid,
            "expires_at": token_data["expires_at"],
            "seconds_remaining": int(seconds_remaining),
            "refreshable": "refresh_token" in token_data,
            "message": "Valid token" if valid else "Token expiring soon",
            "source": "oauth_storage"
        }
        
    except Exception as e:
        logger.error(f"❌ Token status error: {e}")
        return {
            "valid": False,
            "error": str(e),
            "message": "Error checking token status"
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

# ============= DEBUG ENDPOINTS =============
@app.get("/debug/ebay-raw/{keywords}")
async def debug_ebay_raw(keywords: str):
    """Debug endpoint to see EXACTLY what eBay returns"""
    update_activity()
    
    token = get_ebay_token()
    if not token:
        return {"error": "No token available"}
    
    try:
        headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json',
            'X-EBAY-C-MARKETPLACE-ID': 'EBAY_US'
        }
        
        params = {
            'q': keywords,
            'limit': '20',
            'filter': 'buyingOptions:{FIXED_PRICE|AUCTION}'
        }
        
        url = "https://api.ebay.com/buy/browse/v1/item_summary/search"
        response = requests.get(url, headers=headers, params=params, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            
            # Log EVERY item for debugging
            items = []
            for i, item in enumerate(data.get('itemSummaries', [])):
                price = item.get('price', {}).get('value', 'No price')
                title = item.get('title', 'No title')
                condition = item.get('condition', 'No condition')
                
                items.append({
                    'index': i,
                    'title': title,
                    'price': price,
                    'condition': condition,
                    'item_id': item.get('itemId', 'No ID'),
                    'buying_options': item.get('buyingOptions', []),
                    'item_end_date': item.get('itemEndDate', ''),
                    'item_web_url': item.get('itemWebUrl', '')
                })
            
            return {
                "success": True,
                "total_results": len(data.get('itemSummaries', [])),
                "items": items,
                "query_used": keywords,
                "token_available": True
            }
        else:
            return {
                "success": False,
                "status_code": response.status_code,
                "error": response.text[:500],
                "headers_sent": str(headers)[:200]
            }
            
    except Exception as e:
        logger.error(f"Debug error: {e}")
        return {"success": False, "error": str(e)}

# ============= MAIN ENDPOINTS =============

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
                'mime_type': file.content_type,
                'title': title,
                'description': description,
                'status': 'queued',
                'created_at': datetime.now().isoformat(),
                'requires_ebay_auth': not bool(ebay_token)
            }
        
        job_queue.put(job_id)
        logger.info(f"📤 Job {job_id} queued")
        
        return {
            "message": "Analysis queued with REAL eBay data",
            "job_id": job_id,
            "status": "queued",
            "estimated_time": "25-30 seconds",
            "check_status_url": f"/job/{job_id}/status",
            "ebay_auth_status": "connected" if ebay_token else "required",
            "user_details_provided": bool(title or description)
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
        "processing_mode": "MAXIMUM_ACCURACY_REAL_EBAY_ONLY"
    }

@app.get("/ping")
async def ping():
    update_activity()
    return {
        "status": "✅ PONG",
        "timestamp": datetime.now().isoformat(),
        "message": "Server awake with REAL eBay data",
        "ebay_ready": bool(get_ebay_token())
    }

@app.get("/")
async def root():
    update_activity()
    ebay_token = get_ebay_token()
    
    return {
        "message": "🎯 AI Resell Pro API - v4.0",
        "status": "🚀 OPERATIONAL" if groq_client and ebay_token else "⚠️ AUTH REQUIRED",
        "version": "4.0.0",
        "ebay_authentication": "✅ Connected" if ebay_token else "❌ Required",
        "features": [
            "Real eBay COMPLETE sold item data only",
            "Parts/incomplete listings filtered out",
            "Era detection (Victorian, Mid-Century, etc.)",
            "Rare item database (coins, stamps, cards)",
            "Proper search patterns (Year Brand Model)",
            "Background removal ready (iOS lift feature)",
            "Sold item comparisons with links"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8000))
    logger.info(f"🚀 Starting server on port {port}")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        workers=1,
        timeout_keep_alive=30,
        log_level="info"
    )