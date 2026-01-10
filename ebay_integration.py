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
    Enhanced eBay API client using OAuth 2.0
    Uses Taxonomy API, Marketing API, and Browse API
    """
    
    def __init__(self, sandbox=False):
        self.sandbox = sandbox
        self.app_id = os.getenv('EBAY_APP_ID')
        self.auth_token = os.getenv('EBAY_AUTH_TOKEN')
        
        if not self.auth_token:
            logger.error("❌ EBAY_AUTH_TOKEN not set. Get OAuth token from /ebay/oauth/start")
            raise ValueError("EBAY_AUTH_TOKEN environment variable required")
        
        # Verify token format
        if not self.auth_token.startswith('v^'):
            logger.warning("⚠️ Token doesn't match typical eBay OAuth format")
        
        # Base URLs
        if sandbox:
            self.browse_base_url = "https://api.sandbox.ebay.com/buy/browse/v1"
            self.taxonomy_base_url = "https://api.sandbox.ebay.com/commerce/taxonomy/v1_beta"
            self.marketing_base_url = "https://api.sandbox.ebay.com/sell/marketing/v1"
        else:
            self.browse_base_url = "https://api.ebay.com/buy/browse/v1"
            self.taxonomy_base_url = "https://api.ebay.com/commerce/taxonomy/v1_beta"
            self.marketing_base_url = "https://api.ebay.com/sell/marketing/v1"
        
        logger.info(f"🔧 eBay OAuth API initialized (sandbox: {sandbox})")
    
    def _make_request(self, base_url: str, endpoint: str, method: str = 'GET', 
                     params: Dict = None, data: Dict = None) -> Optional[Dict]:
        """Make authenticated OAuth request"""
        try:
            url = f"{base_url}{endpoint}"
            headers = {
                'Authorization': f'Bearer {self.auth_token}',
                'Content-Type': 'application/json'
            }
            
            # Add marketplace header for Browse API
            if 'buy/browse' in base_url:
                headers['X-EBAY-C-MARKETPLACE-ID'] = 'EBAY_US'
            
            logger.debug(f"🌐 {method} {endpoint}")
            
            if method.upper() == 'GET':
                response = requests.get(url, headers=headers, params=params, timeout=15)
            elif method.upper() == 'POST':
                response = requests.post(url, headers=headers, json=data, params=params, timeout=15)
            else:
                return None
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 401:
                logger.error(f"❌ OAuth token invalid for {endpoint}")
                return None
            elif response.status_code == 403:
                logger.error(f"❌ Missing scope for {endpoint}")
                return None
            else:
                logger.error(f"❌ API error {response.status_code}: {response.text[:200]}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Request error for {endpoint}: {e}")
            return None
    
    def verify_token(self) -> bool:
        """Verify OAuth token is valid"""
        try:
            # Test with Browse API
            result = self._make_request(
                self.browse_base_url,
                '/item_summary/search',
                params={'q': 'test', 'limit': '1'}
            )
            return result is not None
        except:
            return False
    
    def get_category_suggestions(self, keywords: str) -> Optional[Dict]:
        """Get category suggestions using Taxonomy API"""
        try:
            # Get default category tree
            tree_result = self._make_request(
                self.taxonomy_base_url,
                '/get_default_category_tree_id'
            )
            
            if not tree_result:
                logger.error("❌ Failed to get category tree")
                return None
            
            tree_id = tree_result.get('categoryTreeId')
            if not tree_id:
                return None
            
            # Get category suggestions
            endpoint = f"/category_tree/{tree_id}/get_category_suggestions"
            params = {'q': keywords[:200]}  # Limit query length
            
            suggestions = self._make_request(
                self.taxonomy_base_url,
                endpoint,
                params=params
            )
            
            if suggestions:
                logger.info(f"✅ Taxonomy API found categories for: {keywords[:50]}...")
            
            return suggestions
            
        except Exception as e:
            logger.error(f"❌ Category suggestions error: {e}")
            return None
    
    def get_keyword_suggestions(self, seed_keywords: str, category_id: str = None) -> Optional[Dict]:
        """Get keyword suggestions using Marketing API"""
        try:
            # First, check if we have campaigns
            campaigns = self._make_request(
                self.marketing_base_url,
                '/ad_campaign'
            )
            
            if not campaigns:
                logger.warning("⚠️ No campaigns found for Marketing API")
                return None
            
            campaign_list = campaigns.get('campaigns', [])
            if not campaign_list:
                logger.warning("⚠️ Empty campaign list")
                return None
            
            # Use first campaign
            campaign_id = campaign_list[0]['campaignId']
            
            # Get ad groups
            ad_groups_endpoint = f"/ad_campaign/{campaign_id}/ad_group"
            ad_groups = self._make_request(
                self.marketing_base_url,
                ad_groups_endpoint
            )
            
            if not ad_groups:
                logger.warning("⚠️ No ad groups found")
                return None
            
            ad_group_list = ad_groups.get('adGroups', [])
            if not ad_group_list:
                logger.warning("⚠️ Empty ad group list")
                return None
            
            ad_group_id = ad_group_list[0]['adGroupId']
            
            # Get keyword suggestions
            keywords_endpoint = f"/ad_campaign/{campaign_id}/ad_group/{ad_group_id}/suggest_keywords"
            
            request_data = {
                "keywords": [seed_keywords],
                "maxNumOfKeywords": 20
            }
            
            if category_id:
                request_data["categoryIds"] = [category_id]
            
            suggestions = self._make_request(
                self.marketing_base_url,
                keywords_endpoint,
                method='POST',
                data=request_data
            )
            
            if suggestions:
                logger.info(f"✅ Marketing API provided keyword suggestions")
            
            return suggestions
            
        except Exception as e:
            logger.error(f"❌ Keyword suggestions error: {e}")
            return None
    
    def search_sold_items(self, keywords: str, category_id: str = None, 
                         aspect_filters: Dict = None, limit: int = 50) -> List[Dict]:
        """Search for sold items with optimized filters"""
        try:
            # Build filter string
            filter_parts = ['soldItems:true']
            
            if category_id:
                filter_parts.append(f'categoryIds:{category_id}')
            
            # Build aspect filter if provided
            aspect_filter_str = None
            if aspect_filters:
                aspect_parts = []
                for key, value in aspect_filters.items():
                    if isinstance(value, list):
                        value_str = '|'.join(str(v) for v in value)
                        aspect_parts.append(f'{key}:{{{value_str}}}')
                    else:
                        aspect_parts.append(f'{key}:{{{value}}}')
                
                if aspect_parts:
                    aspect_filter_str = ','.join(aspect_parts)
            
            params = {
                'q': keywords,
                'limit': str(min(limit, 100)),  # eBay max is 200
                'filter': ','.join(filter_parts),
                'sort': '-endTime'
            }
            
            if aspect_filter_str:
                params['aspect_filter'] = aspect_filter_str
            
            logger.info(f"🔍 Searching: '{keywords[:50]}...'")
            
            data = self._make_request(
                self.browse_base_url,
                '/item_summary/search',
                params=params
            )
            
            if not data or 'itemSummaries' not in data:
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
                        'end_time': item.get('endDate', ''),
                        'image_url': item.get('image', {}).get('imageUrl', '')
                    })
                except:
                    continue
            
            logger.info(f"✅ Found {len(items)} sold items")
            return items
            
        except Exception as e:
            logger.error(f"❌ Search error: {e}")
            return []
    
    def analyze_market_trends(self, keywords: str, category_id: str = None, 
                             item_data: Dict = None) -> Dict:
        """Analyze market trends with optimized search"""
        logger.info(f"📊 Analyzing market for: '{keywords}'")
        
        # Extract core aspects for filtering
        aspect_filters = self._extract_core_aspects(keywords, item_data)
        
        # Search for sold items
        sold_items = self.search_sold_items(
            keywords=keywords,
            category_id=category_id,
            aspect_filters=aspect_filters,
            limit=50
        )
        
        if not sold_items:
            return self._generate_empty_analysis(keywords)
        
        # Calculate statistics
        prices = [item['price'] for item in sold_items if item['price'] > 0]
        
        if not prices:
            return self._generate_empty_analysis(keywords)
        
        # Calculate metrics
        prices.sort()
        median_price = prices[len(prices) // 2]
        avg_price = sum(prices) / len(prices)
        min_price = prices[0]
        max_price = prices[-1]
        
        # Calculate quartiles
        q1 = prices[len(prices) // 4]
        q3 = prices[3 * len(prices) // 4]
        
        # Recommended buy price (25th percentile)
        recommended_buy = q1
        
        # Calculate confidence
        if len(prices) >= 50:
            confidence = 'high'
            confidence_reason = f'Excellent sample size ({len(prices)} sold items)'
        elif len(prices) >= 20:
            confidence = 'good'
            confidence_reason = f'Good sample size ({len(prices)} sold items)'
        elif len(prices) >= 10:
            confidence = 'medium'
            confidence_reason = f'Moderate sample size ({len(prices)} sold items)'
        else:
            confidence = 'low'
            confidence_reason = f'Limited sample size ({len(prices)} sold items)'
        
        # Get date range for recency
        recent_dates = []
        for item in sold_items[:10]:
            if 'end_time' in item:
                try:
                    date_str = item['end_time'].replace('Z', '+00:00')
                    recent_dates.append(datetime.fromisoformat(date_str))
                except:
                    pass
        
        days_since_last = 999
        if recent_dates:
            latest_date = max(recent_dates)
            days_since_last = (datetime.now() - latest_date).days
        
        analysis = {
            'average_price': round(avg_price, 2),
            'median_price': round(median_price, 2),
            'price_range': f"${min_price:.2f} - ${max_price:.2f}",
            'lowest_price': round(min_price, 2),
            'highest_price': round(max_price, 2),
            'total_sold_analyzed': len(sold_items),
            'recommended_buy_below': round(recommended_buy, 2),
            'market_notes': self._generate_market_notes(len(sold_items), days_since_last),
            'confidence': confidence,
            'confidence_reason': confidence_reason,
            'sample_size': len(sold_items),
            'days_since_last_sale': days_since_last,
            'price_stability': 'high' if (max_price - min_price) / median_price < 0.5 else 'medium',
            'data_source': 'eBay Sold Listings',
            'aspects_used': list(aspect_filters.keys()) if aspect_filters else [],
            'sold_items_sample': sold_items[:3]
        }
        
        logger.info(f"✅ Market analysis complete: {len(sold_items)} comps, confidence: {confidence}")
        return analysis
    
    def _extract_core_aspects(self, keywords: str, item_data: Dict = None) -> Dict:
        """Extract core structural aspects for filtering"""
        aspects = {}
        
        # Extract year
        year_match = re.search(r'\b(19[0-9]{2}|20[0-9]{2})\b', keywords)
        if year_match:
            aspects['Year'] = year_match.group(1)
        
        # Extract vehicle makes
        vehicle_makes = ['Toyota', 'Honda', 'Ford', 'Chevrolet', 'Chevy', 'BMW', 'Mercedes']
        for make in vehicle_makes:
            if make.lower() in keywords.lower():
                aspects['Make'] = make
                break
        
        return aspects
    
    def _generate_market_notes(self, sample_size: int, days_since_last: int) -> str:
        """Generate market notes"""
        notes = []
        
        if sample_size >= 20:
            notes.append(f"Based on {sample_size} recent sold listings")
        elif sample_size >= 10:
            notes.append(f"Based on {sample_size} sold listings")
        else:
            notes.append(f"Limited data ({sample_size} sold listings)")
        
        if days_since_last <= 7:
            notes.append("Very recent sales (last 7 days)")
        elif days_since_last <= 30:
            notes.append(f"Recent sales (last {days_since_last} days)")
        
        return ". ".join(notes)
    
    def _generate_empty_analysis(self, keywords: str) -> Dict:
        """Generate empty analysis when no data found"""
        return {
            'average_price': 0.0,
            'median_price': 0.0,
            'price_range': "$0.00 - $0.00",
            'lowest_price': 0.0,
            'highest_price': 0.0,
            'total_sold_analyzed': 0,
            'recommended_buy_below': 0.0,
            'market_notes': f'No sold items found for "{keywords}"',
            'confidence': 'very low',
            'confidence_reason': 'No sold items found',
            'sample_size': 0,
            'days_since_last_sale': 999,
            'price_stability': 'unknown',
            'data_source': 'eBay Sold Listings',
            'aspects_used': [],
            'sold_items_sample': []
        }

# Create global instance (PRODUCTION)
ebay_api = eBayOAuthAPI(sandbox=False)