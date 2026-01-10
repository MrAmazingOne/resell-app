import os
import requests
import json
import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import time

logger = logging.getLogger(__name__)

class eBayOAuthAPI:
    """
    CORRECT eBay API implementation with proper endpoints
    """
    
    def __init__(self, sandbox=False):
        self.sandbox = sandbox
        self.auth_token = os.getenv('EBAY_AUTH_TOKEN')
        
        if not self.auth_token:
            logger.error("❌ EBAY_AUTH_TOKEN not set")
            raise ValueError("EBAY_AUTH_TOKEN required")
        
        # CORRECT eBay API endpoints
        if sandbox:
            self.browse_base_url = "https://api.sandbox.ebay.com/buy/browse/v1"
            self.taxonomy_base_url = "https://api.sandbox.ebay.com/commerce/taxonomy/v1"  # Taxonomy API
            self.marketing_base_url = "https://api.sandbox.ebay.com/sell/marketing/v1"
        else:
            self.browse_base_url = "https://api.ebay.com/buy/browse/v1"
            self.taxonomy_base_url = "https://api.ebay.com/commerce/taxonomy/v1"  # PRODUCTION Taxonomy API
            self.marketing_base_url = "https://api.ebay.com/sell/marketing/v1"
        
        logger.info("✅ eBay API initialized with correct endpoints")
        logger.info(f"   Browse API: {self.browse_base_url}")
        logger.info(f"   Taxonomy API: {self.taxonomy_base_url}")
        logger.info(f"   Marketing API: {self.marketing_base_url}")
    
    def _make_request(self, base_url: str, endpoint: str, method: str = 'GET', 
                     params: Dict = None, data: Dict = None, 
                     marketplace_id: str = 'EBAY_US') -> Optional[Dict]:
        """Make authenticated request with proper headers"""
        try:
            url = f"{base_url}{endpoint}"
            headers = {
                'Authorization': f'Bearer {self.auth_token}',
                'Content-Type': 'application/json'
            }
            
            # Add marketplace header for Browse API and Taxonomy API
            if 'buy/browse' in base_url:
                headers['X-EBAY-C-MARKETPLACE-ID'] = marketplace_id
            
            # Taxonomy API uses 'Accept-Language' header instead of marketplace ID
            if 'commerce/taxonomy' in base_url:
                headers['Accept-Language'] = 'en-US'
            
            logger.debug(f"🌐 {method} {endpoint}")
            
            if method.upper() == 'GET':
                response = requests.get(url, headers=headers, params=params, timeout=15)
            elif method.upper() == 'POST':
                response = requests.post(url, headers=headers, json=data, params=params, timeout=15)
            else:
                return None
            
            logger.debug(f"📡 Response: {response.status_code}")
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 401:
                logger.error(f"❌ Token invalid (401)")
                return None
            elif response.status_code == 403:
                logger.error(f"❌ Missing scope for this API")
                logger.error(f"   URL: {url}")
                logger.error(f"   Response: {response.text[:200]}")
                return None
            elif response.status_code == 404:
                logger.error(f"❌ Endpoint not found: {endpoint}")
                logger.error(f"   Check if endpoint URL is correct")
                return None
            else:
                logger.error(f"❌ API error {response.status_code}: {response.text[:200]}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Request error: {e}")
            return None
    
    def verify_token(self) -> bool:
        """Verify token works with Browse API"""
        try:
            result = self._make_request(
                self.browse_base_url,
                '/item_summary/search',
                params={'q': 'test', 'limit': '1'}
            )
            if result:
                logger.info("✅ Token verified with Browse API")
                return True
            return False
        except Exception as e:
            logger.error(f"❌ Token verification failed: {e}")
            return False
    
    def get_default_category_tree_id(self) -> Optional[str]:
        """Get default category tree ID for US marketplace"""
        try:
            result = self._make_request(
                self.taxonomy_base_url,
                '/get_default_category_tree_id',
                params={'marketplace_id': 'EBAY_US'}
            )
            if result and 'categoryTreeId' in result:
                logger.info(f"✅ Default category tree ID: {result['categoryTreeId']}")
                return result['categoryTreeId']
            return None
        except Exception as e:
            logger.error(f"❌ Failed to get category tree ID: {e}")
            return None
    
    def get_category_suggestions(self, keywords: str) -> Optional[Dict]:
        """
        Get category suggestions using eBay Taxonomy API
        Uses get_category_suggestions endpoint with proper category tree ID
        """
        try:
            logger.info(f"🔍 Getting category for: {keywords[:50]}...")
            
            # First get the default category tree ID
            category_tree_id = self.get_default_category_tree_id()
            if not category_tree_id:
                logger.warning("⚠️ Could not get category tree ID, using default")
                category_tree_id = "0"  # Default for US marketplace
            
            # Use the get_category_suggestions endpoint
            result = self._make_request(
                self.taxonomy_base_url,
                f'/category_tree/{category_tree_id}/get_category_suggestions',
                params={'q': keywords[:100]}
            )
            
            if result and 'categorySuggestions' in result:
                logger.info(f"✅ Taxonomy API returned {len(result['categorySuggestions'])} category suggestions")
                return result
            
            logger.warning("⚠️ No category suggestions returned, using Browse API inference")
            return self._get_category_from_browse_api(keywords)
            
        except Exception as e:
            logger.error(f"❌ Taxonomy API error: {e}")
            return self._get_category_from_browse_api(keywords)
    
    def _get_category_from_browse_api(self, keywords: str) -> Dict:
        """Get category by analyzing search results from Browse API"""
        try:
            search_result = self._make_request(
                self.browse_base_url,
                '/item_summary/search',
                params={'q': keywords, 'limit': '10'}
            )
            
            if search_result and 'itemSummaries' in search_result:
                items = search_result['itemSummaries']
                if items:
                    # Count categories in results
                    category_count = {}
                    for item in items:
                        category = item.get('primaryCategory', {})
                        cat_id = category.get('categoryId')
                        cat_name = category.get('categoryName', f'Category {cat_id}')
                        
                        if cat_id:
                            if cat_id not in category_count:
                                category_count[cat_id] = {
                                    'id': cat_id,
                                    'name': cat_name,
                                    'count': 0
                                }
                            category_count[cat_id]['count'] += 1
                    
                    if category_count:
                        # Get most common category
                        best = max(category_count.values(), key=lambda x: x['count'])
                        
                        return {
                            'categorySuggestions': [{
                                'category': {
                                    'categoryId': best['id'],
                                    'categoryName': best['name']
                                },
                                'categoryTreeNodeLevel': 2,
                                'relevance': 'HIGH'
                            }],
                            'source': 'browse_api_inference'
                        }
            
            # Default fallback
            return {
                'categorySuggestions': [{
                    'category': {
                        'categoryId': '267',
                        'categoryName': 'Collectibles'
                    },
                    'categoryTreeNodeLevel': 1,
                    'relevance': 'MEDIUM'
                }],
                'source': 'default_fallback'
            }
            
        except Exception as e:
            logger.error(f"❌ Browse API category inference failed: {e}")
            return {
                'categorySuggestions': [{
                    'category': {
                        'categoryId': '267',
                        'categoryName': 'Collectibles'
                    },
                    'categoryTreeNodeLevel': 1,
                    'relevance': 'LOW'
                }],
                'source': 'error_fallback'
            }
    
    def get_keyword_suggestions(self, seed_keywords: str, category_id: str = None) -> Optional[Dict]:
        """Get keyword suggestions using Marketing API"""
        try:
            logger.info(f"🔍 Getting keyword suggestions for: {seed_keywords}")
            
            # First, try to use Marketing API for keyword suggestions
            # This API requires sell.marketing scope
            headers = {
                'Authorization': f'Bearer {self.auth_token}',
                'Content-Type': 'application/json',
                'X-EBAY-C-MARKETPLACE-ID': 'EBAY_US'
            }
            
            # Try to get ad campaigns to find campaign ID
            try:
                campaigns_url = f"{self.marketing_base_url}/ad_campaign"
                campaigns_response = requests.get(
                    campaigns_url,
                    headers=headers,
                    params={'limit': '1'},
                    timeout=10
                )
                
                if campaigns_response.status_code == 200:
                    campaigns_data = campaigns_response.json()
                    if campaigns_data.get('campaigns'):
                        campaign_id = campaigns_data['campaigns'][0]['campaignId']
                        
                        # Get ad groups for the campaign
                        ad_groups_url = f"{self.marketing_base_url}/ad_campaign/{campaign_id}/ad_group"
                        ad_groups_response = requests.get(
                            ad_groups_url,
                            headers=headers,
                            params={'limit': '1'},
                            timeout=10
                        )
                        
                        if ad_groups_response.status_code == 200:
                            ad_groups_data = ad_groups_response.json()
                            if ad_groups_data.get('adGroups'):
                                ad_group_id = ad_groups_data['adGroups'][0]['adGroupId']
                                
                                # Get keyword suggestions
                                suggestions_url = f"{self.marketing_base_url}/ad_campaign/{campaign_id}/ad_group/{ad_group_id}/suggest_keywords"
                                
                                request_data = {
                                    "adGroupId": ad_group_id,
                                    "keywords": [seed_keywords],
                                    "maxNumOfKeywords": 10
                                }
                                
                                if category_id:
                                    request_data["categoryIds"] = [category_id]
                                
                                suggestions_response = requests.post(
                                    suggestions_url,
                                    headers=headers,
                                    json=request_data,
                                    timeout=10
                                )
                                
                                if suggestions_response.status_code == 200:
                                    suggestions_data = suggestions_response.json()
                                    logger.info("✅ Marketing API returned keyword suggestions")
                                    return suggestions_data
            except Exception as e:
                logger.debug(f"Marketing API attempt failed, using fallback: {e}")
            
            # Fallback to smart keyword generation
            logger.warning("⚠️ Marketing API not accessible, using smart keyword generation")
            return self._generate_smart_keywords(seed_keywords, category_id)
            
        except Exception as e:
            logger.error(f"❌ Keyword suggestions error: {e}")
            return self._generate_smart_keywords(seed_keywords, category_id)
    
    def _generate_smart_keywords(self, seed_keywords: str, category_id: str = None) -> Dict:
        """Generate smart keywords when Marketing API fails"""
        keywords = seed_keywords.lower().strip()
        
        # Basic variations
        variations = [
            keywords,
            f"{keywords} used",
            f"{keywords} new",
            f"{keywords} excellent condition",
            f"{keywords} like new"
        ]
        
        # Add category-specific variations
        if category_id:
            if category_id == '6001':  # Vehicles
                variations.append(f"{keywords} car")
                variations.append(f"{keywords} truck")
                variations.append(f"{keywords} vehicle")
            elif category_id == '9355':  # Phones
                variations.append(f"{keywords} smartphone")
                variations.append(f"{keywords} cell phone")
                variations.append(f"{keywords} mobile phone")
            elif category_id == '183454':  # Trading cards
                variations.append(f"{keywords} card")
                variations.append(f"{keywords} trading card")
                variations.append(f"{keywords} collectible")
        
        # Create response format similar to Marketing API
        suggested_keywords = []
        for i, keyword in enumerate(variations[:10]):
            suggested_keywords.append({
                'keywordText': keyword,
                'matchType': 'BROAD',
                'bidPercentage': '100'
            })
        
        return {
            'suggestedKeywords': suggested_keywords,
            'source': 'smart_generation'
        }
    
    def search_sold_items(self, keywords: str, category_id: str = None, 
                         limit: int = 50) -> List[Dict]:
        """Search for sold items with proper filtering"""
        try:
            logger.info(f"🔍 Searching sold items: '{keywords[:50]}...'")
            
            # Build filters
            filter_parts = ['soldItems:true']
            if category_id:
                filter_parts.append(f'category_ids:{category_id}')
            
            params = {
                'q': keywords,
                'limit': str(min(limit, 100)),  # eBay max is 200
                'filter': ','.join(filter_parts),
                'sort': '-endTime'  # Most recent first
            }
            
            data = self._make_request(
                self.browse_base_url,
                '/item_summary/search',
                params=params
            )
            
            if not data or 'itemSummaries' not in data:
                logger.warning("⚠️ No sold items found")
                return []
            
            items = []
            for item in data['itemSummaries']:
                try:
                    price = item.get('price', {}).get('value', '0')
                    items.append({
                        'title': item.get('title', ''),
                        'price': float(price),
                        'item_id': item.get('itemId', ''),
                        'condition': item.get('condition', ''),
                        'category_id': item.get('primaryCategory', {}).get('categoryId', ''),
                        'category_name': item.get('primaryCategory', {}).get('categoryName', ''),
                        'end_time': item.get('endDate', ''),
                        'image_url': item.get('image', {}).get('imageUrl', ''),
                        'item_location': item.get('itemLocation', {}).get('country', '')
                    })
                except Exception as e:
                    logger.debug(f"Skipping item: {e}")
                    continue
            
            logger.info(f"✅ Found {len(items)} sold items")
            return items
            
        except Exception as e:
            logger.error(f"❌ Search error: {e}")
            return []
    
    def analyze_market_trends(self, keywords: str, category_id: str = None, 
                             item_data: Dict = None) -> Dict:
        """Analyze market trends with robust error handling"""
        logger.info(f"📊 Analyzing market for: '{keywords}'")
        
        # Search for sold items
        sold_items = self.search_sold_items(keywords, category_id, limit=50)
        
        if not sold_items:
            logger.warning("⚠️ No sold items found for analysis")
            return self._generate_empty_analysis(keywords)
        
        # Extract prices
        prices = [item['price'] for item in sold_items if item['price'] > 0]
        
        if not prices:
            logger.warning("⚠️ No valid prices found")
            return self._generate_empty_analysis(keywords)
        
        # Calculate statistics
        prices.sort()
        median_price = prices[len(prices) // 2]
        avg_price = sum(prices) / len(prices)
        min_price = prices[0]
        max_price = prices[-1]
        
        # Calculate quartiles for better recommendations
        q1 = prices[len(prices) // 4]  # 25th percentile
        q3 = prices[3 * len(prices) // 4]  # 75th percentile
        
        # Confidence based on sample size
        sample_size = len(prices)
        if sample_size >= 50:
            confidence = 'high'
            confidence_reason = f'Excellent sample size ({sample_size} sold items)'
        elif sample_size >= 20:
            confidence = 'good'
            confidence_reason = f'Good sample size ({sample_size} sold items)'
        elif sample_size >= 10:
            confidence = 'medium'
            confidence_reason = f'Moderate sample size ({sample_size} sold items)'
        elif sample_size >= 5:
            confidence = 'low'
            confidence_reason = f'Limited sample size ({sample_size} sold items)'
        else:
            confidence = 'very low'
            confidence_reason = f'Very limited sample size ({sample_size} sold items)'
        
        # Calculate price stability
        price_range = max_price - min_price
        if median_price > 0:
            stability_ratio = price_range / median_price
            if stability_ratio < 0.3:
                price_stability = 'high'
            elif stability_ratio < 0.6:
                price_stability = 'medium'
            else:
                price_stability = 'low'
        else:
            price_stability = 'unknown'
        
        # Recommended buy price (25th percentile - good deal price)
        recommended_buy = q1
        
        # Calculate recency
        recent_items = sold_items[:5]  # Check 5 most recent
        days_list = []
        for item in recent_items:
            if 'end_time' in item and item['end_time']:
                try:
                    # Parse eBay date format
                    date_str = item['end_time'].replace('Z', '+00:00')
                    end_date = datetime.fromisoformat(date_str)
                    days_ago = (datetime.now() - end_date).days
                    days_list.append(days_ago)
                except:
                    continue
        
        days_since_last = min(days_list) if days_list else 999
        
        analysis = {
            'average_price': round(avg_price, 2),
            'median_price': round(median_price, 2),
            'price_range': f"${min_price:.2f} - ${max_price:.2f}",
            'lowest_price': round(min_price, 2),
            'highest_price': round(max_price, 2),
            'total_sold_analyzed': sample_size,
            'recommended_buy_below': round(recommended_buy, 2),
            'market_notes': self._generate_market_notes(sample_size, days_since_last),
            'confidence': confidence,
            'confidence_reason': confidence_reason,
            'sample_size': sample_size,
            'days_since_last_sale': days_since_last,
            'price_stability': price_stability,
            'data_source': 'eBay Sold Listings',
            'quartile_25': round(q1, 2),
            'quartile_75': round(q3, 2),
            'price_volatility': round(price_range / median_price, 2) if median_price > 0 else 0
        }
        
        logger.info(f"✅ Market analysis complete: {sample_size} comps, confidence: {confidence}")
        return analysis
    
    def _generate_market_notes(self, sample_size: int, days_since_last: int) -> str:
        """Generate helpful market notes"""
        notes = []
        
        if sample_size >= 20:
            notes.append(f"Based on {sample_size} recent sold listings - highly reliable data")
        elif sample_size >= 10:
            notes.append(f"Based on {sample_size} sold listings - good reliability")
        elif sample_size >= 5:
            notes.append(f"Based on {sample_size} sold listings - use with caution")
        else:
            notes.append(f"Very limited data ({sample_size} listings) - consider broader search")
        
        if days_since_last <= 7:
            notes.append("Very recent sales (last 7 days)")
        elif days_since_last <= 30:
            notes.append(f"Recent sales (last {days_since_last} days)")
        elif days_since_last <= 90:
            notes.append(f"Sales within last {days_since_last} days")
        else:
            notes.append("Sales may be outdated - market may have changed")
        
        return ". ".join(notes)
    
    def _generate_empty_analysis(self, keywords: str) -> Dict:
        """Generate analysis when no data found"""
        return {
            'average_price': 0.0,
            'median_price': 0.0,
            'price_range': "$0.00 - $0.00",
            'lowest_price': 0.0,
            'highest_price': 0.0,
            'total_sold_analyzed': 0,
            'recommended_buy_below': 0.0,
            'market_notes': f'No sold items found for "{keywords}". Try different keywords or check spelling.',
            'confidence': 'very low',
            'confidence_reason': 'No sold items found',
            'sample_size': 0,
            'days_since_last_sale': 999,
            'price_stability': 'unknown',
            'data_source': 'eBay Sold Listings',
            'quartile_25': 0.0,
            'quartile_75': 0.0,
            'price_volatility': 0.0
        }

# Create global instance
ebay_api = eBayOAuthAPI(sandbox=False)