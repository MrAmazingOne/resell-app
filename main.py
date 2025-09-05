from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
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
from ebay_integration import ebay_api  # Added eBay integration
from ebay_auth import get_authorization_url, exchange_code_for_token  # NEW: Import auth functions
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Resell Pro API", version="2.5.0")

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

# ENHANCED MARKET ANALYSIS PROMPT WITH EBAY CONTEXT
market_analysis_prompt = """
You are an expert reseller and market analyst with deep knowledge of current eBay market trends. 
Use recent eBay sold data to provide accurate pricing and market insights.

CRITICAL REQUIREMENTS:
1. Focus on items with REAL resale potential on eBay
2. Base pricing on ACTUAL recent eBay sold prices, not retail prices
3. Consider current eBay market demand and trends
4. Account for item condition visible in the image
5. Provide honest resellability scores based on eBay sell-through rates

For each valuable item you identify, provide:
- eBay-optimized title with key search terms
- Pricing based on recent eBay sold listings
- Best eBay category for listing
- eBay-specific selling tips

RESPONSE FORMAT: Valid JSON array with these exact fields:
- "title": eBay-optimized title (include brand, model, condition)
- "description": Compelling description with eBay keywords
- "price_range": Current eBay market range (e.g. "$25-45")
- "resellability_rating": Honest score 1-10 based on eBay demand
- "suggested_cost": Maximum to pay for profit (e.g. "$8-15")
- "market_insights": eBay-specific market analysis
- "authenticity_checks": What eBay buyers look for
- "profit_potential": Estimated profit after eBay fees
- "category": Best eBay category
- "ebay_specific_tips": Tips for eBay listing optimization

JSON Response:
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
    
    # Return category with highest score
    return max(scores.items(), key=lambda x: x[1])[0]

def enhance_with_ebay_data(item_data: Dict) -> Dict:
    """Enhance AI analysis with real eBay market data"""
    try:
        # Search for similar completed items on eBay
        keywords = f"{item_data.get('brand', '')} {item_data.get('title', '')}".strip()
        completed_items = ebay_api.search_completed_items(keywords, max_results=10)
        
        if completed_items:
            prices = [item['price'] for item in completed_items]
            avg_price = sum(prices) / len(prices)
            
            # Update pricing based on real eBay data
            if 'price_range' in item_data:
                # Keep the AI's range but adjust based on real data
                item_data['market_insights'] += f" Based on {len(completed_items)} recent eBay sales, similar items sold for ${min(prices):.2f}-${max(prices):.2f}."
                item_data['price_range'] = f"${min(prices):.2f}-${max(prices):.2f}"
            
            # Add eBay-specific tips
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

# EBAY AUTH ENDPOINTS
@app.get("/auth/ebay")
async def auth_ebay():
    """Redirect to eBay authorization"""
    auth_url = get_authorization_url()
    return RedirectResponse(url=auth_url)

@app.get("/auth/callback")
async def auth_callback(code: str):
    """Handle eBay OAuth callback"""
    try:
        token_data = await exchange_code_for_token(code)
        # Store token (in production, use database)
        os.environ['EBAY_AUTH_TOKEN'] = token_data['access_token']
        return RedirectResponse(url="/auth/success")
    except Exception as e:
        logger.error(f"Auth callback error: {e}")
        return RedirectResponse(url="/auth/failed")

@app.get("/auth/success")
async def auth_success():
    """eBay auth success page"""
    return HTMLResponse("""
    <html>
        <body style="font-family: Arial, sans-serif; text-align: center; padding: 50px;">
            <h1 style="color: green;">✅ Authentication Successful!</h1>
            <p>You have successfully authenticated with eBay.</p>
            <p><a href="/" style="color: #007bff; text-decoration: none;">Return to ReReSell App</a></p>
        </body>
    </html>
    """)

@app.get("/auth/failed")
async def auth_failed():
    """eBay auth failure page"""
    return HTMLResponse("""
    <html>
        <body style="font-family: Arial, sans-serif; text-align: center; padding: 50px;">
            <h1 style="color: red;">❌ Authentication Failed</h1>
            <p>There was an issue authenticating with eBay.</p>
            <p><a href="/auth/ebay" style="color: #007bff; text-decoration: none;">Try again</a></p>
        </body>
    </html>
    """)

@app.get("/privacy")
async def privacy_policy():
    """Privacy policy page for eBay requirements"""
    return HTMLResponse("""
    <html>
        <body style="font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px;">
            <h1>Privacy Policy</h1>
            <p><strong>ReReSell AI eBay Assistant</strong> respects your privacy and is committed to protecting your personal information.</p>
            
            <h2>Information We Collect</h2>
            <p>We only collect information necessary for eBay integration:</p>
            <ul>
                <li>eBay OAuth tokens for API access</li>
                <li>Item images and descriptions for analysis</li>
                <li>Market data for pricing recommendations</li>
            </ul>
            
            <h2>How We Use Your Information</h2>
            <p>Your information is used solely for:</p>
            <ul>
                <li>eBay API authentication and listing management</li>
                <li>AI analysis of items for resale potential</li>
                <li>Market trend analysis and pricing recommendations</li>
            </ul>
            
            <h2>Data Security</h2>
            <p>We implement industry-standard security measures to protect your data.</p>
            
            <h2>Contact</h2>
            <p>For privacy concerns, please contact us through the eBay Developer Program.</p>
        </body>
    </html>
    """)

@app.post("/upload_item/")
async def create_upload_file(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    description: Optional[str] = Form(None)
):
    try:
        logger.info(f"Processing upload - Title: {title}, File: {file.filename}")
        
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and prepare image
        image_bytes = await file.read()
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        
        image_part = {
            "mime_type": file.content_type,
            "data": image_base64
        }
        
        # Build enhanced prompt with user context
        prompt_parts = [market_analysis_prompt]
        
        if title:
            prompt_parts.append(f"User-provided context - Title: \"{title}\"")
        if description:
            prompt_parts.append(f"User-provided context - Description: \"{description}\"")
        
        prompt_parts.append("\nFocus on items that would be profitable to resell. Ignore low-value common items unless they have specific collectible value.")
        
        try:
            # AI Analysis
            prompt_text = " ".join(prompt_parts)
            response = model.generate_content([prompt_text, image_part])
            
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                logger.error(f"Content blocked: {response.prompt_feedback.block_reason}")
                raise HTTPException(status_code=400, detail="Image content was blocked by safety filters")
            
            logger.info("AI analysis completed successfully")
            
        except Exception as ai_error:
            logger.error(f"AI analysis failed: {ai_error}")
            raise HTTPException(status_code=503, detail="AI service temporarily unavailable")
        
        # Parse AI response
        try:
            generated_content = response.text.strip()
            
            # Extract JSON
            if "```json" in generated_content:
                json_start = generated_content.find("```json") + 7
                json_end = generated_content.find("```", json_start)
                json_text = generated_content[json_start:json_end].strip()
            elif "```" in generated_content:
                json_start = generated_content.find("```") + 3
                json_end = generated_content.rfind("```")
                json_text = generated_content[json_start:json_end].strip()
            else:
                json_text = generated_content
            
            ai_response = json.loads(json_text)
            
            if not isinstance(ai_response, list):
                raise ValueError("AI response is not a JSON array")
            
            logger.info(f"Successfully parsed {len(ai_response)} items from AI")
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {e}")
            # Fallback response
            ai_response = [{
                "title": "Analysis Failed",
                "description": "Unable to parse AI response. Please try again with a clearer image.",
                "price_range": "$0-0",
                "resellability_rating": 1,
                "suggested_cost": "$0",
                "market_insights": "AI analysis failed",
                "authenticity_checks": "Unable to analyze",
                "profit_potential": "$0",
                "category": "unknown",
                "ebay_specific_tips": []
            }]
        
        # ENHANCE WITH CATEGORY DETECTION AND EBAY DATA
        enhanced_items = []
        for item_data in ai_response:
            try:
                # Add category detection
                detected_category = detect_category(item_data.get("title", ""), item_data.get("description", ""))
                item_data["category"] = detected_category.value
                
                # Enhance with eBay market data
                item_data = enhance_with_ebay_data(item_data)
                
                # Add category-specific prompt enhancements
                if detected_category in category_prompts:
                    item_data["market_insights"] += f" {category_prompts[detected_category]}"
                
                enhanced_items.append(EnhancedAppItem(item_data))
                    
            except Exception as enhance_error:
                logger.error(f"Enhancement failed for item: {enhance_error}")
                # Use original data if enhancement fails
                enhanced_items.append(EnhancedAppItem(item_data))
        
        # Convert to response format
        response_items = [item.to_dict() for item in enhanced_items]
        
        logger.info(f"Successfully processed {len(response_items)} items")
        
        return {
            "message": "Items analyzed successfully!",
            "filename": file.filename,
            "items": response_items,
            "analysis_timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

@app.post("/batch_analyze/")
async def batch_analyze_items(files: List[UploadFile] = File(...)):
    """Process multiple images in a single request"""
    try:
        results = []
        for file in files:
            # Process each file individually
            result = await create_upload_file(file)
            results.append(result)
        
        return {
            "message": f"Processed {len(results)} images",
            "results": results,
            "batch_timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Batch analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")

@app.post("/list_to_ebay/")
async def list_to_ebay(item_data: Dict):
    """List an analyzed item directly to eBay"""
    try:
        result = ebay_api.list_item(item_data)
        return result
    except Exception as e:
        logger.error(f"eBay listing failed: {e}")
        raise HTTPException(status_code=500, detail=f"eBay listing failed: {str(e)}")

@app.get("/ebay_search/{keywords}")
async def ebay_search(keywords: str, category: Optional[str] = None):
    """Search eBay for completed listings"""
    try:
        results = ebay_api.search_completed_items(keywords, category)
        return {"results": results}
    except Exception as e:
        logger.error(f"eBay search failed: {e}")
        raise HTTPException(status_code=500, detail=f"eBay search failed: {str(e)}")

@app.get("/")
async def root():
    """API health check"""
    return {
        "message": "AI Resell Pro API v2.5 - Ready to maximize profits! 🚀",
        "status": "healthy",
        "features": [
            "AI item identification",
            "Category detection",
            "Multi-object extraction support",
            "Profit optimization",
            "eBay market integration",
            "Direct eBay listing"
        ],
        "endpoints": {
            "auth": "/auth/ebay",
            "privacy": "/privacy",
            "upload": "/upload_item",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "ai_configured": bool(model),
        "ebay_configured": bool(ebay_api),
        "model": "gemini-2.5-flash-preview-05-20",
        "version": "2.5.0",
        "features": {
            "ai_analysis": True,
            "category_detection": True,
            "multi_object": True,
            "profit_calculation": True,
            "ebay_integration": True
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)