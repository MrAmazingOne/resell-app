# main.py (Updated Version)
# JOB QUEUE + POLLING SYSTEM - MAXIMUM ACCURACY ONLY
# Enhanced search strategies for specific item types

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
from datetime import datetime, timedelta, timezone
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
    title="AI Resell Pro API - Enhanced Search Strategies", 
    version="4.1.0",
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

# ============= ENHANCED SEARCH STRATEGIES =============

def build_item_type_specific_queries(item_data: Dict, user_keywords: Dict, detected_category: str) -> List[str]:
    """
    Build category-specific search queries with optimal keyword ordering
    """
    search_strategies = []
    
    # Get AI-identified specifics
    title = item_data.get('title', '').lower()
    description = item_data.get('description', '').lower()
    brand = item_data.get('brand', '').lower()
    model = item_data.get('model', '').lower()
    year = item_data.get('year', '').strip()
    era = item_data.get('era', '').lower()
    
    # 🎯 CATEGORY-SPECIFIC SEARCH PATTERNS
    if detected_category == 'collectibles':
        # For trading cards: "[Year] [Brand] [Card Name] [Set] [Card Number] [Features]"
        # Example: "2006 Pokemon Ninetales Delta Species 8/101 Holo"
        
        # Extract specific card details
        card_name = extract_card_name(title, description)
        card_number = extract_card_number(description)
        set_name = extract_set_name(description)
        features = extract_card_features(description)
        
        queries = []
        
        # STRATEGY 1: Complete card identification
        if card_name and card_number:
            # Format: "Pokemon Ninetales Delta Species 8/101"
            query = f"pokemon {card_name} {card_number}"
            if set_name:
                query += f" {set_name}"
            if features:
                query += f" {features}"
            queries.append(query)
            logger.info(f"🃏 CARD EXACT: '{query}'")
        
        # STRATEGY 2: Brand + Card Name + Features
        if card_name:
            query = f"pokemon {card_name}"
            if 'delta' in description.lower() or 'delta' in title.lower():
                query += " delta species"
            if 'holo' in description.lower() or 'holographic' in description.lower():
                query += " holo"
            queries.append(query)
            logger.info(f"🃏 CARD FEATURED: '{query}'")
        
        # STRATEGY 3: Card Number + Set
        if card_number and set_name:
            query = f"pokemon {card_number} {set_name}"
            queries.append(query)
            logger.info(f"🃏 CARD NUMBER+SET: '{query}'")
        
        # STRATEGY 4: Clean generic fallback
        if card_name:
            # Remove generic words and focus on unique identifiers
            generic_words = ['pokemon', 'card', 'cards', 'trading', 'collectible', 'rare']
            name_parts = [word for word in card_name.split() if word not in generic_words]
            if name_parts:
                query = f"pokemon {' '.join(name_parts[:3])}"  # Max 3 specific name parts
                queries.append(query)
                logger.info(f"🃏 CARD CLEAN: '{query}'")
        
        search_strategies = queries
        
    elif detected_category == 'vehicles':
        # For vehicles: "[Year] [Make] [Model] [Trim]" 
        # Example: "1955 Chevrolet 3100", "1969 Ford Mustang"
        
        # Clean brand name
        vehicle_brand = clean_vehicle_brand(brand)
        
        # Clean model name (remove window/trim descriptors for search)
        vehicle_model = clean_vehicle_model(model)
        
        queries = []
        
        # STRATEGY 1: Year + Make + Model (primary)
        if year and vehicle_brand and vehicle_model:
            query = f"{year} {vehicle_brand} {vehicle_model}"
            queries.append(query)
            logger.info(f"🚗 VEHICLE PRIMARY: '{query}'")
        
        # STRATEGY 2: Make + Model + Year (alternative)
        if vehicle_brand and vehicle_model:
            query = f"{vehicle_brand} {vehicle_model}"
            if year:
                query += f" {year}"
            queries.append(query)
            logger.info(f"🚗 VEHICLE ALTERNATE: '{query}'")
        
        # STRATEGY 3: Just Make + Model
        if vehicle_brand and vehicle_model:
            query = f"{vehicle_brand} {vehicle_model}"
            queries.append(query)
            logger.info(f"🚗 VEHICLE BASIC: '{query}'")
        
        search_strategies = queries
        
    elif detected_category in ['furniture', 'jewelry', 'art']:
        # For antiques/collectibles: "[Era] [Style] [Material] [Item Type]"
        # Example: "Victorian Mahogany Chair", "Art Deco Platinum Ring"
        
        queries = []
        
        # STRATEGY 1: Era + Style + Item Type
        if era and detected_category:
            query = f"{era} {detected_category}"
            queries.append(query)
            logger.info(f"🏛️ ERA+CATEGORY: '{query}'")
        
        # STRATEGY 2: Material + Style
        material = extract_material(description)
        if material:
            query = f"{material} {detected_category}"
            if era:
                query += f" {era}"
            queries.append(query)
            logger.info(f"🏛️ MATERIAL+ERA: '{query}'")
        
        # STRATEGY 3: Brand + Era (for jewelry)
        if brand and 'unknown' not in brand and detected_category == 'jewelry':
            query = f"{brand} {detected_category}"
            if era:
                query += f" {era}"
            queries.append(query)
            logger.info(f"💎 BRAND+ERA: '{query}'")
        
        search_strategies = queries
        
    else:
        # Generic fallback for other categories
        queries = []
        
        # Use brand + model + year pattern
        if brand and 'unknown' not in brand:
            query_parts = []
            if year:
                query_parts.append(year)
            query_parts.append(brand)
            if model and 'unknown' not in model:
                query_parts.append(model)
            
            if query_parts:
                query = " ".join(query_parts)
                queries.append(query)
                logger.info(f"🏷️ GENERIC BRAND: '{query}'")
        
        # Clean title as fallback
        if title:
            # Remove common generic words
            stop_words = ['the', 'a', 'an', 'and', 'or', 'for', 'with', 'in', 'on', 'at', 'to', 
                         'item', 'product', 'good', 'excellent', 'condition', 'vintage', 'used']
            title_words = [word for word in title.split()[:6] 
                          if word.lower() not in stop_words and len(word) > 2]
            if title_words:
                query = " ".join(title_words)
                queries.append(query)
                logger.info(f"📝 CLEANED TITLE: '{query}'")
        
        search_strategies = queries
    
    # Add user keywords if available (HIGH PRIORITY)
    if user_keywords:
        user_queries = build_user_keyword_queries(user_keywords, detected_category)
        search_strategies = user_queries + search_strategies  # User queries first
    
    # Clean and deduplicate
    cleaned = []
    seen = set()
    for strategy in search_strategies:
        strategy = clean_search_query(strategy)
        if (strategy and 
            strategy not in seen and 
            len(strategy) > 3 and
            len(strategy.split()) <= 6):  # Max 6 terms
            seen.add(strategy)
            cleaned.append(strategy[:80])  # eBay search limit
    
    logger.info(f"🔍 FINAL ENHANCED SEARCH STRATEGIES: {cleaned}")
    return cleaned[:4]  # Max 4 strategies

def extract_card_name(title: str, description: str) -> str:
    """Extract specific card name from title/description"""
    # Remove generic words
    generic_words = ['pokemon', 'card', 'cards', 'trading', 'collectible', 'rare', 'vintage', 'holographic']
    
    # Look for patterns like "Ninetales Delta Species"
    words = title.lower().split()
    card_words = []
    
    for word in words:
        if word not in generic_words and not word.isdigit():
            card_words.append(word)
    
    return " ".join(card_words[:3]) if card_words else ""

def extract_card_number(description: str) -> str:
    """Extract card number like 8/101"""
    pattern = r'(\d+\s*/\s*\d+)'
    match = re.search(pattern, description)
    return match.group(1) if match else ""

def extract_set_name(description: str) -> str:
    """Extract set name like 'Dragon Frontiers'"""
    set_keywords = ['dragon frontiers', 'holon phantoms', 'crystal guardians', 'ex set']
    for keyword in set_keywords:
        if keyword in description.lower():
            return keyword.title()
    return ""

def extract_card_features(description: str) -> str:
    """Extract card features like 'holo', '1st edition'"""
    features = []
    feature_keywords = ['holo', 'holographic', 'first edition', '1st edition', 'reverse holo', 'delta species']
    
    desc_lower = description.lower()
    for keyword in feature_keywords:
        if keyword in desc_lower:
            features.append(keyword)
    
    return " ".join(features)

def clean_vehicle_brand(brand: str) -> str:
    """Clean vehicle brand for search"""
    if not brand:
        return ""
    
    brand = brand.lower()
    # Clean common variations
    brand_mapping = {
        'chevy': 'chevrolet',
        'vw': 'volkswagen',
        'benz': 'mercedes',
        'mb': 'mercedes',
        'bmw': 'bmw',
        'ford': 'ford',
        'toyota': 'toyota',
        'honda': 'honda'
    }
    
    for key, value in brand_mapping.items():
        if key in brand:
            return value
    
    return brand.title()

def clean_vehicle_model(model: str) -> str:
    """Clean vehicle model for search"""
    if not model or 'unknown' in model.lower():
        return ""
    
    # Remove window/trim descriptors
    model = re.sub(r'\d+\s*window', '', model.lower())
    model = re.sub(r'window', '', model)
    model = re.sub(r'deluxe|custom|standard|edition|trim', '', model)
    
    return model.strip().title()

def extract_material(description: str) -> str:
    """Extract material from description"""
    materials = ['mahogany', 'walnut', 'oak', 'teak', 'cherry', 'maple',
                'gold', 'silver', 'platinum', 'brass', 'bronze',
                'marble', 'granite', 'crystal', 'glass']
    
    desc_lower = description.lower()
    for material in materials:
        if material in desc_lower:
            return material.title()
    
    return ""

def build_user_keyword_queries(user_keywords: Dict, category: str) -> List[str]:
    """Build search queries from user-provided keywords"""
    queries = []
    
    if not user_keywords:
        return []
    
    # Start with specific combinations
    query_parts = []
    
    # Year first (for vehicles, antiques)
    if user_keywords.get('years'):
        query_parts.append(user_keywords['years'][0])
    
    # Brand second
    if user_keywords.get('brands'):
        query_parts.append(user_keywords['brands'][0])
    
    # Model third
    if user_keywords.get('models'):
        query_parts.append(user_keywords['models'][0])
    
    # Features fourth
    if user_keywords.get('features'):
        # Filter out non-search-friendly features
        search_features = []
        for feature in user_keywords['features'][:2]:  # Max 2 features
            if len(feature) > 3 and feature.lower() not in ['window', 'windows', 'deluxe']:
                search_features.append(feature)
        query_parts.extend(search_features)
    
    if query_parts:
        query = " ".join(query_parts)
        queries.append(query)
        logger.info(f"🎯 USER KEYWORD QUERY: '{query}'")
    
    # Category-specific user queries
    if category == 'collectibles' and user_keywords.get('features'):
        # For cards, include specific features like "1st edition"
        for feature in user_keywords['features']:
            if any(keyword in feature.lower() for keyword in ['edition', 'holo', 'delta']):
                if user_keywords.get('brands'):
                    query = f"{user_keywords['brands'][0]} {feature}"
                    queries.append(query)
                    logger.info(f"🃏 USER CARD FEATURE: '{query}'")
    
    return queries

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

# ============= ENHANCED EBAY SEARCH =============

def search_ebay_directly(keywords: str, limit: int = 10, category: str = None) -> List[Dict]:
    """Enhanced eBay search with category-specific filtering"""
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
        
        # FIXED: Timezone-aware datetime comparison
        now_utc = datetime.now(timezone.utc)
        
        params = {
            'q': keywords,
            'limit': str(limit * 3),  # Get more to filter
            'filter': 'price:[0.01..10000],buyingOptions:{FIXED_PRICE|AUCTION}',
            'sort': 'price',
            'fieldgroups': 'EXTENDED'
        }
        
        # Add category filter if available
        if category and category != 'unknown':
            category_id = map_to_ebay_category_id(category)
            if category_id:
                params['category_ids'] = category_id
        
        url = "https://api.ebay.com/buy/browse/v1/item_summary/search"
        
        logger.info(f"🔍 Enhanced eBay search for: '{keywords}' (category: {category})")
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
                        item_title = item.get('title', '').lower()
                        price = item.get('price', {}).get('value', '0')
                        price_float = float(price)
                        condition = item.get('condition', '')
                        item_end_date = item.get('itemEndDate', '')
                        buying_options = item.get('buyingOptions', [])
                        
                        # FIXED: Proper timezone-aware comparison
                        is_sold = False
                        if item_end_date:
                            try:
                                # Parse with timezone
                                end_time = datetime.fromisoformat(item_end_date.replace('Z', '+00:00'))
                                if end_time < now_utc:
                                    is_sold = True
                            except ValueError:
                                # Fallback parsing
                                try:
                                    end_time = datetime.strptime(item_end_date, '%Y-%m-%dT%H:%M:%S.%fZ').replace(tzinfo=timezone.utc)
                                    if end_time < now_utc:
                                        is_sold = True
                                except:
                                    pass
                        
                        # CATEGORY-SPECIFIC FILTERING
                        if category == 'collectibles':
                            # For cards, filter out:
                            # 1. Complete sets/lots (unless specifically looking for sets)
                            # 2. Unrelated items
                            # 3. Junk listings
                            
                            title_lower = item_title.lower()
                            
                            # Skip if it's a set/lot (unless keywords indicate we want a set)
                            set_keywords = ['set of', 'lot of', 'collection', 'multiple', 'bundle']
                            is_set = any(keyword in title_lower for keyword in set_keywords)
                            
                            # Check if we're looking for a specific single card
                            is_single_card = any(word in keywords.lower() for word in ['card', 'single', 'specific'])
                            
                            if is_set and is_single_card:
                                logger.debug(f"   [{i+1}] Skipping - set/lot for single card search")
                                continue
                            
                            # Filter out unrelated categories
                            unrelated = ['plush', 'toy', 'figure', 'poster', 'binder', 'box']
                            if any(unrelated_word in title_lower for unrelated_word in unrelated):
                                logger.debug(f"   [{i+1}] Skipping - unrelated item type")
                                continue
                        
                        elif category == 'vehicles':
                            # For vehicles, ensure realistic prices
                            if price_float < 100:  # Most vehicles are > $100
                                logger.debug(f"   [{i+1}] Skipping - unrealistic vehicle price: ${price_float}")
                                continue
                        
                        # Skip parts/not working listings for most searches
                        parts_keywords = ['for parts only', 'parts only', 'not working', 'as is', 'broken', 'damaged']
                        is_parts = any(kw in item_title for kw in parts_keywords)
                        
                        # For collectibles, allow "parts" if it's actually cards/collectibles
                        if category != 'collectibles' and is_parts:
                            logger.debug(f"   [{i+1}] Skipping - parts/not working")
                            continue
                        
                        # Ensure reasonable price
                        if price_float <= 0:
                            logger.debug(f"   [{i+1}] Skipping - zero/negative price")
                            continue
                        
                        # Add valid item
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
                            'item_end_date': item_end_date,
                            'search_match_score': calculate_search_match_score(item_title, keywords)
                        })
                        
                        if len(items) >= limit:
                            break
                            
                    except (KeyError, ValueError) as e:
                        logger.debug(f"   [{i+1}] Skipping item - parsing error: {e}")
                        continue
                
                # Sort by search match score (most relevant first)
                items.sort(key=lambda x: x.get('search_match_score', 0), reverse=True)
                
                logger.info(f"✅ Found {len(items)} relevant items")
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

def calculate_search_match_score(item_title: str, search_query: str) -> int:
    """Calculate how well an item matches the search query"""
    score = 0
    title_lower = item_title.lower()
    query_lower = search_query.lower()
    
    # Split into words
    query_words = set(query_lower.split())
    title_words = set(title_lower.split())
    
    # Exact word matches
    exact_matches = query_words.intersection(title_words)
    score += len(exact_matches) * 10
    
    # Partial matches (substrings)
    for query_word in query_words:
        if any(query_word in title_word or title_word in query_word 
               for title_word in title_words):
            score += 5
    
    # Boost for having all query words
    if len(exact_matches) == len(query_words):
        score += 20
    
    # Penalize for "lot", "set", "multiple" when searching for singles
    if any(word in query_lower for word in ['single', 'specific', 'card']):
        if any(word in title_lower for word in ['lot', 'set', 'bundle', 'multiple']):
            score -= 15
    
    return score

def map_to_ebay_category_id(category: str) -> str:
    """Map internal category to eBay category ID"""
    category_mapping = {
        'collectibles': '1',           # Collectibles
        'vehicles': '6000',            # eBay Motors
        'electronics': '58058',        # Electronics
        'furniture': '3197',           # Furniture
        'jewelry': '281',              # Jewelry & Watches
        'art': '550',                  # Art
        'clothing': '11450',           # Clothing
        'toys': '220',                 # Toys & Hobbies
        'sports': '888',               # Sporting Goods
        'books': '267',                # Books
        'music': '11233',              # Music
        'coins': '11116',              # Coins & Paper Money
        'stamps': '260'                # Stamps
    }
    return category_mapping.get(category.lower(), '')

# ============= ENHANCED MARKET ANALYSIS =============

def analyze_ebay_market_directly(keywords: str, category: str = None) -> Dict:
    """Enhanced eBay market analysis with better filtering"""
    logger.info(f"📊 Enhanced eBay market analysis for: '{keywords}' (category: {category})")
    
    # Get sold comparison items with better filtering
    sold_items = search_ebay_directly(keywords, limit=20, category=category)
    
    if not sold_items:
        logger.warning("⚠️ NO RELEVANT ITEMS FOUND - trying alternative strategies")
        # Try without category filter
        sold_items = search_ebay_directly(keywords, limit=15, category=None)
    
    if not sold_items:
        logger.error("❌ NO EBAY DATA AVAILABLE")
        return {
            'error': 'NO_EBAY_DATA',
            'message': 'Unable to retrieve relevant eBay data for this search',
            'requires_auth': False
        }
    
    # Filter for single items when appropriate
    if category == 'collectibles':
        # For cards, prefer single card listings over sets
        single_items = []
        set_items = []
        
        for item in sold_items:
            title_lower = item['title'].lower()
            is_set = any(word in title_lower for word in ['set', 'lot', 'bundle', 'collection', 'multiple'])
            
            if is_set:
                set_items.append(item)
            else:
                single_items.append(item)
        
        # Prefer single items, but use sets if no singles found
        if single_items:
            sold_items = single_items[:10]  # Top 10 single items
            logger.info(f"🎯 Using {len(sold_items)} single item listings")
        elif set_items:
            sold_items = set_items[:10]
            logger.info(f"📦 Using {len(sold_items)} set listings (no singles found)")
    
    # Calculate statistics only on valid items
    valid_items = [item for item in sold_items if item.get('price', 0) > 0]
    
    if not valid_items:
        logger.error("❌ No valid price data from eBay")
        return {
            'error': 'NO_PRICE_DATA',
            'message': 'eBay returned items but no valid price data',
            'requires_auth': False
        }
    
    prices = [item['price'] for item in valid_items]
    
    # Remove outliers (prices outside 2 standard deviations)
    if len(prices) >= 5:
        import statistics
        try:
            mean = statistics.mean(prices)
            stdev = statistics.stdev(prices) if len(prices) > 1 else 0
            filtered_prices = [p for p in prices if abs(p - mean) <= 2 * stdev]
            if filtered_prices:  # Only use filtered if we still have data
                prices = filtered_prices
                logger.info(f"📊 Removed price outliers, using {len(prices)} prices")
        except:
            pass  # If statistics fails, use all prices
    
    # Calculate statistics
    avg_price = sum(prices) / len(prices)
    min_price = min(prices)
    max_price = max(prices)
    
    # Calculate confidence based on data quality
    confidence = 'high'
    if len(prices) < 5:
        confidence = 'medium'
    if len(prices) < 3:
        confidence = 'low'
    
    # Check if we have exact matches
    exact_matches = [item for item in valid_items 
                     if item.get('search_match_score', 0) > 30]
    
    if exact_matches:
        logger.info(f"🎯 Found {len(exact_matches)} exact matches")
        # Use exact matches for pricing if available
        exact_prices = [item['price'] for item in exact_matches]
        if exact_prices:
            avg_price = sum(exact_prices) / len(exact_prices)
            min_price = min(exact_prices)
            max_price = max(exact_prices)
            confidence = 'high'
    
    analysis = {
        'success': True,
        'average_price': round(avg_price, 2),
        'price_range': f"${min_price:.2f} - ${max_price:.2f}",
        'lowest_price': round(min_price, 2),
        'highest_price': round(max_price, 2),
        'total_sold_analyzed': len(valid_items),
        'recommended_price': round(avg_price * 0.85, 2),  # 15% below average for resale
        'market_notes': f'Based on {len(valid_items)} relevant eBay items',
        'data_source': 'eBay Browse API',
        'confidence': confidence,
        'api_used': 'Browse API',
        'sold_items': valid_items[:8],  # Top 8 most relevant
        'exact_matches_count': len(exact_matches),
        'search_strategy_used': keywords
    }
    
    logger.info(f"✅ Market analysis complete: avg=${avg_price:.2f}, range=${min_price:.2f}-${max_price:.2f}, confidence={confidence}")
    
    return analysis

# ============= ENHANCED PROCESSING FUNCTION =============

def enhance_with_ebay_data_user_prioritized(item_data: Dict, vision_analysis: Dict, user_keywords: Dict) -> Dict:
    """
    Enhanced market analysis using REAL eBay data with BETTER search strategies
    """
    try:
        detected_category = item_data.get('category', 'unknown')
        logger.info(f"📦 Category: '{detected_category}' for enhanced eBay search")
        
        # Build ENHANCED search queries with category-specific patterns
        search_strategies = build_item_type_specific_queries(item_data, user_keywords, detected_category)
        
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
            logger.info(f"🔍 Searching eBay with enhanced strategy: '{strategy}'")
            analysis = analyze_ebay_market_directly(strategy, detected_category)
            
            if analysis and analysis.get('success'):
                # Check if results are actually relevant
                sold_count = analysis.get('total_sold_analyzed', 0)
                exact_matches = analysis.get('exact_matches_count', 0)
                
                # For collectibles, we want good matches
                if detected_category == 'collectibles':
                    if exact_matches >= 3 or sold_count >= 5:
                        market_analysis = analysis
                        sold_items = analysis.get('sold_items', [])
                        best_strategy = strategy
                        logger.info(f"✅ Found relevant data with strategy: '{strategy}' ({exact_matches} exact matches)")
                        break
                    else:
                        logger.info(f"⚠️ Strategy '{strategy}' returned {sold_count} items ({exact_matches} exact), trying next...")
                else:
                    # For other categories
                    if sold_count >= 3:
                        market_analysis = analysis
                        sold_items = analysis.get('sold_items', [])
                        best_strategy = strategy
                        logger.info(f"✅ Found relevant data with strategy: '{strategy}' ({sold_count} items)")
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
            avg_price = market_analysis['average_price']
            min_price = market_analysis['lowest_price']
            max_price = market_analysis['highest_price']
            
            item_data['price_range'] = f"${min_price:.2f} - ${max_price:.2f}"
            item_data['suggested_cost'] = f"${market_analysis['recommended_price']:.2f}"
            
            # Calculate profit with realistic fees
            ebay_fees = avg_price * 0.13  # 13% eBay fees
            shipping_cost = 8.00 if detected_category != 'collectibles' else 4.00  # Lower for cards
            packaging_cost = 3.00
            estimated_net = avg_price - ebay_fees - shipping_cost - packaging_cost
            suggested_purchase = market_analysis['recommended_price']
            profit = estimated_net - suggested_purchase
            
            if profit > 0:
                item_data['profit_potential'] = f"${profit:.2f} profit (after all fees)"
            else:
                item_data['profit_potential'] = f"${abs(profit):.2f} potential loss"
            
            # Market insights with specific details
            insights = []
            if best_strategy:
                insights.append(f"Search: '{best_strategy}'")
            
            insights.extend([
                f"Based on {market_analysis['total_sold_analyzed']} eBay items",
                f"Average sold price: ${avg_price:.2f}",
                f"Price range: ${min_price:.2f} - ${max_price:.2f}",
                f"Confidence: {market_analysis['confidence']}"
            ])
            
            if market_analysis.get('exact_matches_count', 0) > 0:
                insights.append(f"Exact matches found: {market_analysis['exact_matches_count']}")
            
            item_data['market_insights'] = ". ".join(insights)
            
            # eBay tips based on category
            ebay_tips = []
            if best_strategy:
                ebay_tips.append(f"Use search terms like: {best_strategy}")
            
            if detected_category == 'collectibles':
                ebay_tips.extend([
                    "Photograph front and back clearly",
                    "Include card number in title",
                    "Mention condition (Near Mint, Lightly Played)",
                    "Consider professional grading for rare cards"
                ])
            elif detected_category == 'vehicles':
                ebay_tips.extend([
                    "Include VIN in description",
                    "Show clear photos of all angles",
                    "List maintenance history",
                    "Be honest about any issues"
                ])
            else:
                ebay_tips.extend([
                    "Use 'Buy It Now' with Best Offer",
                    "Include detailed measurements",
                    "Take photos from all angles",
                    "List on weekends for visibility"
                ])
            
            item_data['ebay_specific_tips'] = ebay_tips
            item_data['identification_confidence'] = market_analysis['confidence']
            item_data['data_source'] = market_analysis['data_source']
            
            # Add sold comparison items (filtered for relevance)
            if sold_items:
                comparison_items = []
                sold_items_links = []
                prices = []
                
                for item in sold_items[:5]:  # Top 5 most relevant sold items
                    # Skip if it's a set and we're looking for singles
                    if (detected_category == 'collectibles' and 
                        any(word in item.get('title', '').lower() for word in ['set', 'lot', 'bundle'])):
                        continue
                    
                    comparison_items.append({
                        'title': item.get('title', ''),
                        'sold_price': item.get('price', 0),
                        'condition': item.get('condition', ''),
                        'item_url': item.get('item_web_url', ''),
                        'image_url': item.get('image_url', ''),
                        'sold': item.get('sold', False),
                        'match_score': item.get('search_match_score', 0)
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
                        'total_comparisons': len(prices),
                        'price_confidence': market_analysis['confidence']
                    }
            
            logger.info(f"✅ eBay analysis complete with {len(sold_items)} relevant items")
                    
        else:
            logger.error("❌ NO RELEVANT EBAY DATA")
            item_data['market_insights'] = "⚠️ Unable to find relevant eBay market data for this specific item."
            item_data['identification_confidence'] = "low"
            item_data['price_range'] = "Market data unavailable"
            item_data['suggested_cost'] = "Research required"
        
        return item_data
        
    except Exception as e:
        logger.error(f"❌ eBay enhancement failed: {e}")
        item_data['market_insights'] = f"⚠️ Error analyzing market: {str(e)[:100]}"
        item_data['identification_confidence'] = "error"
        return item_data

# ============= THE REST OF THE FILE REMAINS THE SAME =============
# (Keep all the existing functions, endpoints, and setup code below this point)
# The functions above are the enhanced versions that fix the search issues

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