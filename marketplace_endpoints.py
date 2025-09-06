# Add these endpoints to your main.py file

from typing import Dict, List
import json
import base64
import tempfile
import os
from datetime import datetime

# MARKETPLACE LISTING ENDPOINTS

@app.post("/list_to_ebay/")
async def list_to_ebay(listing_data: Dict):
    """
    List item to eBay using Trading API
    """
    try:
        # Validate required fields
        required_fields = ['title', 'description', 'price', 'images', 'category', 'condition']
        for field in required_fields:
            if field not in listing_data:
                raise HTTPException(status_code=400, detail=f"Missing required field: {field}")
        
        # Process images - save base64 images temporarily
        image_urls = []
        for i, image_base64 in enumerate(listing_data['images'][:12]):  # eBay allows max 12 images
            try:
                # Decode base64 image
                image_data = base64.b64decode(image_base64)
                
                # Save to temporary file (in production, upload to image hosting service)
                temp_dir = tempfile.mkdtemp()
                temp_file = os.path.join(temp_dir, f"image_{i}.jpg")
                with open(temp_file, 'wb') as f:
                    f.write(image_data)
                
                # In production, upload to image hosting service and get URL
                image_url = f"https://your-image-host.com/temp/image_{i}.jpg"
                image_urls.append(image_url)
            except Exception as e:
                logger.warning(f"Failed to process image {i}: {e}")
                continue
        
        # Prepare eBay listing data
        ebay_item_data = {
            'title': listing_data['title'][:80],  # eBay title limit
            'description': create_ebay_description(listing_data),
            'price': listing_data['price'],
            'category_id': get_ebay_category_id(listing_data.get('category', 'Other')),
            'condition': get_ebay_condition_id(listing_data.get('condition', 'Used')),
            'image_urls': image_urls,
            'brand': listing_data.get('brand', ''),
            'model': listing_data.get('model', ''),
            'shipping_weight': listing_data.get('shippingWeight', 1.0)
        }
        
        # List to eBay using the existing eBay API integration
        result = ebay_api.list_item(ebay_item_data)
        
        if result.get('success'):
            return {
                "success": True,
                "marketplace": "ebay",
                "listingId": result.get('item_id'),
                "listingUrl": result.get('listing_url'),
                "fees": calculate_ebay_fees(listing_data['price']),
                "message": "Successfully listed to eBay"
            }
        else:
            return {
                "success": False,
                "marketplace": "ebay",
                "error": result.get('error', 'Unknown eBay error'),
                "listingId": None,
                "listingUrl": None,
                "fees": None
            }
            
    except Exception as e:
        logger.error(f"eBay listing error: {e}")
        return {
            "success": False,
            "marketplace": "ebay",
            "error": str(e),
            "listingId": None,
            "listingUrl": None,
            "fees": None
        }

@app.post("/list_to_craigslist/")
async def list_to_craigslist(listing_data: Dict):
    """
    Generate Craigslist listing format (Craigslist doesn't have direct API)
    """
    try:
        # Craigslist doesn't have an official API, so we create a formatted listing
        # that users can copy/paste to Craigslist
        
        craigslist_title = f"{listing_data['title']} - ${listing_data['price']:.0f}"
        craigslist_description = create_craigslist_description(listing_data)
        
        # Generate a pre-formatted listing
        formatted_listing = f"""
TITLE: {craigslist_title}

PRICE: ${listing_data['price']:.0f}

DESCRIPTION:
{craigslist_description}

CATEGORY: {listing_data.get('category', 'for sale - by owner')}
CONDITION: {listing_data.get('condition', 'good')}

CONTACT: [Your contact information]

Images: {len(listing_data.get('images', []))} images ready for upload
        """.strip()
        
        # Save formatted listing to temporary file for user download
        temp_dir = tempfile.mkdtemp()
        listing_file = os.path.join(temp_dir, f"craigslist_listing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        
        with open(listing_file, 'w', encoding='utf-8') as f:
            f.write(formatted_listing)
        
        return {
            "success": True,
            "marketplace": "craigslist",
            "listingId": f"CL_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "listingUrl": "https://craigslist.org/post",
            "fees": 0.0,  # Craigslist is typically free
            "message": "Craigslist listing formatted - manual posting required",
            "formatted_listing": formatted_listing,
            "instructions": "Copy the formatted listing and paste it when creating a new Craigslist post"
        }
        
    except Exception as e:
        logger.error(f"Craigslist listing error: {e}")
        return {
            "success": False,
            "marketplace": "craigslist",
            "error": str(e),
            "listingId": None,
            "listingUrl": None,
            "fees": None
        }

@app.post("/list_to_facebook/")
async def list_to_facebook(listing_data: Dict):
    """
    Generate Facebook Marketplace listing format
    Note: Facebook Marketplace API requires business verification
    """
    try:
        # Facebook Marketplace API is limited and requires business verification
        # For now, we'll create a formatted listing similar to Craigslist
        
        facebook_description = create_facebook_description(listing_data)
        
        # Format for Facebook Marketplace
        formatted_listing = {
            "title": listing_data['title'][:100],  # Facebook title limit
            "description": facebook_description,
            "price": int(listing_data['price']),
            "category": get_facebook_category(listing_data.get('category', 'Other')),
            "condition": listing_data.get('condition', 'Used'),
            "location": listing_data.get('location', {
                "city": "Your City",
                "state": "Your State",
                "zipCode": "12345"
            }),
            "images_count": len(listing_data.get('images', [])),
            "brand": listing_data.get('brand', ''),
            "model": listing_data.get('model', '')
        }
        
        # Instructions for manual posting
        instructions = """
        1. Go to Facebook Marketplace
        2. Click "Create new listing"
        3. Select "Item for sale"
        4. Upload your images
        5. Use the provided title and description
        6. Set the price and location
        7. Publish your listing
        """
        
        return {
            "success": True,
            "marketplace": "facebook",
            "listingId": f"FB_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "listingUrl": "https://www.facebook.com/marketplace/create/item",
            "fees": listing_data['price'] * 0.05,  # 5% Facebook fee
            "message": "Facebook Marketplace listing formatted - manual posting required",
            "formatted_listing": formatted_listing,
            "instructions": instructions
        }
        
    except Exception as e:
        logger.error(f"Facebook listing error: {e}")
        return {
            "success": False,
            "marketplace": "facebook",
            "error": str(e),
            "listingId": None,
            "listingUrl": None,
            "fees": None
        }

# HELPER FUNCTIONS

def create_ebay_description(listing_data: Dict) -> str:
    """Create eBay-optimized description with HTML formatting"""
    description = f"""
    <div style="font-family: Arial, sans-serif; max-width: 800px;">
        <h2>{listing_data['title']}</h2>
        
        <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin: 10px 0;">
            <h3>Item Details</h3>
            <ul>
                <li><strong>Condition:</strong> {listing_data.get('condition', 'Used')}</li>
                <li><strong>Brand:</strong> {listing_data.get('brand', 'See description')}</li>
                <li><strong>Model:</strong> {listing_data.get('model', 'See description')}</li>
                <li><strong>Category:</strong> {listing_data.get('category', 'Other')}</li>
            </ul>
        </div>
        
        <div style="margin: 15px 0;">
            <h3>Description</h3>
            <p>{listing_data['description'].replace('\n', '<br>')}</p>
        </div>
        
        <div style="background-color: #e8f4fd; padding: 15px; border-radius: 5px; margin: 15px 0;">
            <h3>Shipping & Returns</h3>
            <p>• Fast and secure shipping</p>
            <p>• Carefully packaged</p>
            <p>• 30-day return policy</p>
            <p>• Questions? Message us!</p>
        </div>
        
        <div style="text-align: center; margin: 20px 0; font-size: 12px; color: #666;">
            <p>Thank you for shopping with us!</p>
        </div>
    </div>
    """
    return description

def create_craigslist_description(listing_data: Dict) -> str:
    """Create Craigslist-optimized plain text description"""
    description_parts = []
    
    # Basic info
    if listing_data.get('brand'):
        description_parts.append(f"Brand: {listing_data['brand']}")
    if listing_data.get('model'):
        description_parts.append(f"Model: {listing_data['model']}")
    if listing_data.get('condition'):
        description_parts.append(f"Condition: {listing_data['condition']}")
    
    # Main description
    description_parts.append("")
    description_parts.append(listing_data['description'])
    description_parts.append("")
    
    # Additional details
    description_parts.append("Details:")
    description_parts.append("• Serious buyers only")
    description_parts.append("• Cash only, local pickup preferred")
    description_parts.append("• No trades please")
    description_parts.append("• Feel free to ask questions")
    
    return "\n".join(description_parts)

def create_facebook_description(listing_data: Dict) -> str:
    """Create Facebook Marketplace description"""
    description_parts = []
    
    # Add key details upfront
    if listing_data.get('brand') or listing_data.get('model'):
        brand_model = f"{listing_data.get('brand', '')} {listing_data.get('model', '')}".strip()
        description_parts.append(f"🏷️ {brand_model}")
        description_parts.append("")
    
    # Main description
    description_parts.append(listing_data['description'])
    description_parts.append("")
    
    # Key details
    description_parts.append("📋 Details:")
    if listing_data.get('condition'):
        description_parts.append(f"• Condition: {listing_data['condition']}")
    
    # Contact info
    description_parts.append("")
    description_parts.append("💬 Feel free to message with questions!")
    description_parts.append("📍 Local pickup available")
    
    return "\n".join(description_parts)

def get_ebay_category_id(category: str) -> str:
    """Map general category to eBay category ID"""
    category_mapping = {
        'electronics': '58058',  # Portable Audio & Headphones
        'clothing': '11450',     # Clothing, Shoes & Accessories
        'furniture': '3197',     # Home & Garden > Furniture
        'collectibles': '1',     # Collectibles
        'books': '267',          # Books
        'toys': '220',           # Toys & Hobbies
        'jewelry': '281',        # Jewelry & Watches
        'sports': '888',         # Sporting Goods
        'art': '550',            # Art
        'music': '11233',        # Music
        'video_games': '139973', # Video Games & Consoles
        'automotive': '6000',    # eBay Motors
        'health': '26395',       # Health & Beauty
        'other': '267'           # Default to Collectibles
    }
    return category_mapping.get(category.lower(), '267')

def get_ebay_condition_id(condition: str) -> str:
    """Map condition to eBay condition ID"""
    condition_mapping = {
        'new': '1000',              # New
        'new with tags': '1500',    # New with tags
        'new without tags': '1750', # New without tags
        'new with defects': '1750', # New with defects
        'manufacturer refurbished': '2000', # Manufacturer refurbished
        'seller refurbished': '2500',       # Seller refurbished
        'used': '3000',             # Used
        'very good': '4000',        # Very Good
        'good': '5000',             # Good
        'acceptable': '6000',       # Acceptable
        'for parts': '7000'         # For parts or not working
    }
    return condition_mapping.get(condition.lower(), '3000')  # Default to Used

def get_facebook_category(category: str) -> str:
    """Map general category to Facebook Marketplace category"""
    facebook_categories = {
        'electronics': 'Electronics',
        'clothing': 'Apparel',
        'furniture': 'Home & Garden',
        'collectibles': 'Antiques',
        'books': 'Entertainment',
        'toys': 'Family',
        'jewelry': 'Apparel',
        'sports': 'Sporting Goods',
        'art': 'Home & Garden',
        'music': 'Entertainment',
        'video_games': 'Entertainment',
        'automotive': 'Vehicles',
        'health': 'Health & Beauty',
        'other': 'Other'
    }
    return facebook_categories.get(category.lower(), 'Other')

def calculate_ebay_fees(price: float) -> float:
    """Calculate eBay fees for listing"""
    # Standard eBay final value fee is ~13% for most categories
    final_value_fee = price * 0.13
    
    # PayPal/Payment processing fee ~2.9%
    payment_fee = price * 0.029
    
    # Insertion fee (usually free for first 250 listings per month)
    insertion_fee = 0.0
    
    total_fees = final_value_fee + payment_fee + insertion_fee
    return round(total_fees, 2)

# BATCH LISTING ENDPOINT
@app.post("/list_to_multiple/")
async def list_to_multiple_marketplaces(listing_data: Dict):
    """
    List item to multiple marketplaces simultaneously
    """
    try:
        marketplaces = listing_data.get('marketplaces', [])
        if not marketplaces:
            raise HTTPException(status_code=400, detail="No marketplaces specified")
        
        results = []
        
        for marketplace in marketplaces:
            try:
                if marketplace == 'ebay':
                    result = await list_to_ebay(listing_data)
                elif marketplace == 'craigslist':
                    result = await list_to_craigslist(listing_data)
                elif marketplace == 'facebook':
                    result = await list_to_facebook(listing_data)
                else:
                    result = {
                        "success": False,
                        "marketplace": marketplace,
                        "error": f"Unsupported marketplace: {marketplace}",
                        "listingId": None,
                        "listingUrl": None,
                        "fees": None
                    }
                
                results.append(result)
                
                # Small delay between listings to avoid rate limits
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error listing to {marketplace}: {e}")
                results.append({
                    "success": False,
                    "marketplace": marketplace,
                    "error": str(e),
                    "listingId": None,
                    "listingUrl": None,
                    "fees": None
                })
        
        # Calculate summary
        successful_listings = [r for r in results if r['success']]
        total_fees = sum([r.get('fees', 0) for r in results if r.get('fees')])
        
        return {
            "message": f"Listed to {len(successful_listings)}/{len(marketplaces)} marketplaces",
            "results": results,
            "summary": {
                "total_marketplaces": len(marketplaces),
                "successful_listings": len(successful_listings),
                "total_fees": total_fees,
                "estimated_reach": calculate_estimated_reach(successful_listings)
            }
        }
        
    except Exception as e:
        logger.error(f"Multi-marketplace listing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def calculate_estimated_reach(successful_listings: List[Dict]) -> Dict:
    """Calculate estimated audience reach across platforms"""
    reach_estimates = {
        'ebay': 182_000_000,      # eBay monthly active users
        'craigslist': 50_000_000,  # Craigslist monthly visitors
        'facebook': 1_000_000_000  # Facebook Marketplace users
    }
    
    total_reach = 0
    platforms_used = []
    
    for listing in successful_listings:
        marketplace = listing.get('marketplace', '').lower()
        if marketplace in reach_estimates:
            total_reach += reach_estimates[marketplace]
            platforms_used.append(marketplace)
    
    return {
        "total_potential_reach": total_reach,
        "platforms_used": platforms_used,
        "reach_multiplier": len(platforms_used)
    }

# LISTING MANAGEMENT ENDPOINTS
@app.get("/listings/status/{listing_id}")
async def get_listing_status(listing_id: str):
    """Get status of a specific listing across all platforms"""
    # This would typically check the status from each marketplace
    # For now, return a mock response
    return {
        "listing_id": listing_id,
        "status": "active",
        "platforms": {
            "ebay": {"status": "active", "views": 45, "watchers": 3},
            "craigslist": {"status": "posted", "views": 12, "inquiries": 1},
            "facebook": {"status": "active", "views": 28, "saves": 2}
        },
        "total_stats": {
            "total_views": 85,
            "total_inquiries": 6,
            "days_active": 3
        }
    }

@app.delete("/listings/{listing_id}")
async def delete_listing(listing_id: str):
    """Remove listing from all platforms"""
    try:
        # This would typically call each marketplace's API to end the listing
        return {
            "message": f"Listing {listing_id} removal initiated",
            "ebay": {"status": "ended", "success": True},
            "craigslist": {"status": "manual_removal_required", "success": False},
            "facebook": {"status": "removed", "success": True}
        }
    except Exception as e:
        logger.error(f"Error removing listing {listing_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# PRICING OPTIMIZATION ENDPOINT
@app.post("/optimize_pricing/")
async def optimize_pricing(item_data: Dict):
    """
    Optimize pricing strategy across different marketplaces
    """
    try:
        base_price = item_data.get('price', 0)
        category = item_data.get('category', 'other')
        condition = item_data.get('condition', 'used')
        
        # Get market data from eBay
        keywords = f"{item_data.get('brand', '')} {item_data.get('model', '')} {item_data.get('title', '')}".strip()
        market_analysis = ebay_api.analyze_market_trends(keywords)
        
        # Calculate optimized pricing for each platform
        pricing_strategy = {
            'ebay': {
                'suggested_price': market_analysis.get('recommended_price', base_price),
                'pricing_strategy': 'competitive',
                'rationale': 'Based on completed eBay sales data',
                'fees': calculate_ebay_fees(base_price),
                'net_profit': base_price - calculate_ebay_fees(base_price) - 10  # Subtract shipping
            },
            'craigslist': {
                'suggested_price': base_price * 1.1,  # Slightly higher for local pickup
                'pricing_strategy': 'premium_local',
                'rationale': 'No fees, local pickup convenience',
                'fees': 0,
                'net_profit': base_price * 1.1
            },
            'facebook': {
                'suggested_price': base_price * 1.05,  # Competitive with social proof
                'pricing_strategy': 'social_competitive',
                'rationale': 'Lower fees than eBay, social marketplace dynamics',
                'fees': base_price * 0.05,
                'net_profit': (base_price * 1.05) - (base_price * 0.05)
            }
        }
        
        return {
            "item_analysis": {
                "base_price": base_price,
                "market_confidence": market_analysis.get('confidence', 'medium'),
                "category": category,
                "condition": condition
            },
            "pricing_recommendations": pricing_strategy,
            "summary": {
                "best_profit_platform": max(pricing_strategy.items(), key=lambda x: x[1]['net_profit'])[0],
                "best_reach_platform": "ebay",
                "fastest_sale_platform": "craigslist",
                "recommended_strategy": "List on all platforms with platform-optimized pricing"
            }
        }
        
    except Exception as e:
        logger.error(f"Pricing optimization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))