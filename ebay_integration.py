import os
import requests
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import time
import urllib.parse

logger = logging.getLogger(__name__)

class eBayAPI:
    def __init__(self, sandbox=False):
        self.sandbox = sandbox
        self.app_id = os.getenv('EBAY_APP_ID')
        self.auth_token = os.getenv('EBAY_AUTH_TOKEN')
        
        logger.info(f"🔧 eBay API Configuration:")
        logger.info(f"   Environment: {'SANDBOX' if sandbox else 'PRODUCTION'}")
        logger.info(f"   App ID: {self.app_id[:10]}..." if self.app_id else "   App ID: NOT SET")
        
        if sandbox:
            self.browse_base_url = "https://api.sandbox.ebay.com/buy/browse/v1"
            self.taxonomy_base_url = "https://api.sandbox.ebay.com/commerce/taxonomy/v1_beta"
        else:
            self.browse_base_url = "https://api.ebay.com/buy/browse/v1"
            self.taxonomy_base_url = "https://api.ebay.com/commerce/taxonomy/v1_beta"

    def _make_browse_api_request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Make a request to eBay Browse API"""
        try:
            if not self.auth_token:
                logger.error("❌ No OAuth token available for Browse API")
                return None
                
            headers = {
                'Authorization': f'Bearer {self.auth_token}',
                'Content-Type': 'application/json',
                'X-EBAY-C-MARKETPLACE-ID': 'EBAY_US'
            }
            
            url = f"{self.browse_base_url}{endpoint}"
            
            logger.info(f"🔍 Making Browse API request to {endpoint}")
            
            response = requests.get(url, headers=headers, params=params, timeout=15)
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 401:
                logger.error("❌ Authentication failed - invalid/missing OAuth token")
                return None
            elif response.status_code == 403:
                logger.error("❌ Forbidden - missing scope or insufficient permissions")
                return None
            else:
                logger.error(f"❌ Browse API error {response.status_code}: {response.text[:200]}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Browse API request error: {e}")
            return None

    def get_category_suggestions(self, keywords: str) -> str:
        """
        Get category suggestions using eBay Taxonomy API
        Returns the best category ID for the keywords
        """
        try:
            if not self.auth_token:
                logger.warning("⚠️ No OAuth token - using default category")
                return "267"  # Default Collectibles category
            
            # Get default category tree ID for US marketplace
            headers = {
                'Authorization': f'Bearer {self.auth_token}',
                'Content-Type': 'application/json'
            }
            
            # Get category tree ID
            tree_url = f"{self.taxonomy_base_url}/get_default_category_tree_id"
            tree_response = requests.get(tree_url, headers=headers, timeout=10)
            
            if tree_response.status_code != 200:
                logger.error(f"❌ Failed to get category tree ID: {tree_response.status_code}")
                return "267"
            
            tree_id = tree_response.json().get('categoryTreeId')
            if not tree_id:
                return "267"
            
            # Get category suggestions
            suggest_url = f"{self.taxonomy_base_url}/category_tree/{tree_id}/get_category_suggestions"
            params = {'q': keywords}
            
            suggest_response = requests.get(suggest_url, headers=headers, params=params, timeout=10)
            
            if suggest_response.status_code == 200:
                suggestions = suggest_response.json().get('categorySuggestions', [])
                if suggestions:
                    # Return the first (most relevant) category ID
                    category_id = suggestions[0]['category']['categoryId']
                    logger.info(f"✅ Found category {category_id} for keywords: {keywords[:50]}...")
                    return category_id
            
            logger.warning(f"⚠️ No category suggestions found for: {keywords[:50]}...")
            return "267"
            
        except Exception as e:
            logger.error(f"❌ Category suggestion error: {e}")
            return "267"

    def search_sold_items(self, keywords: str, category_id: str = None, 
                         aspect_filters: Dict = None, limit: int = 50) -> List[Dict]:
        """
        Search for sold items with optimized filters for valuation
        Returns 20-50+ sold items when possible
        """
        if not self.auth_token:
            logger.warning("⚠️ No OAuth token - cannot search sold items")
            return []
        
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
                        # Multiple values (OR)
                        value_str = '|'.join(str(v) for v in value)
                        aspect_parts.append(f'{key}:{{{value_str}}}')
                    else:
                        # Single value
                        aspect_parts.append(f'{key}:{{{value}}}')
                
                if aspect_parts:
                    aspect_filter_str = ','.join(aspect_parts)
            
            params = {
                'q': keywords,
                'limit': str(limit),
                'filter': ','.join(filter_parts),
                'sort': '-endTime'
            }
            
            if aspect_filter_str:
                params['aspect_filter'] = aspect_filter_str
            
            logger.info(f"🔍 Searching sold items: '{keywords[:50]}...'")
            logger.info(f"   Category: {category_id}")
            logger.info(f"   Aspect filters: {aspect_filters}")
            
            data = self._make_browse_api_request('/item_summary/search', params)
            
            if not data or 'itemSummaries' not in data:
                logger.warning("⚠️ No sold items found in search")
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
                        'image_url': item.get('image', {}).get('imageUrl', ''),
                        'buying_options': item.get('buyingOptions', []),
                        'seller_feedback': item.get('seller', {}).get('feedbackPercentage', 0)
                    })
                except (KeyError, ValueError, TypeError) as e:
                    logger.debug(f"   Skipping item due to error: {e}")
                    continue
            
            logger.info(f"✅ Found {len(items)} sold items")
            return items
            
        except Exception as e:
            logger.error(f"❌ Sold items search error: {e}")
            return []

    def analyze_market_trends(self, keywords: str, category_id: str = None, 
                             item_data: Dict = None) -> Dict:
        """
        Analyze market trends with optimized search for valuation
        Target: 20-50+ sold items for reliable statistics
        """
        logger.info(f"📊 Analyzing market for: '{keywords}'")
        
        # Extract core aspects for filtering
        aspect_filters = self._extract_core_aspects(keywords, item_data)
        
        # Try search with core aspects first
        sold_items = self.search_sold_items(
            keywords=keywords,
            category_id=category_id,
            aspect_filters=aspect_filters,
            limit=50
        )
        
        # If not enough results, relax filters
        confidence = self._calculate_confidence(len(sold_items))
        
        if confidence['level'] == 'low' and aspect_filters:
            logger.info(f"⚠️ Only {len(sold_items)} items found with core aspects, relaxing filters...")
            # Remove condition filter first (most restrictive)
            if 'Condition' in aspect_filters:
                relaxed_filters = {k: v for k, v in aspect_filters.items() if k != 'Condition'}
                sold_items = self.search_sold_items(
                    keywords=keywords,
                    category_id=category_id,
                    aspect_filters=relaxed_filters,
                    limit=50
                )
                confidence = self._calculate_confidence(len(sold_items))
        
        if not sold_items:
            logger.warning("⚠️ No sold items found after filter relaxation")
            return self._generate_empty_analysis(keywords)
        
        # Calculate statistics
        prices = [item['price'] for item in sold_items if item['price'] > 0]
        
        if not prices:
            return self._generate_empty_analysis(keywords)
        
        # Sort and calculate percentiles for better statistics
        prices.sort()
        median_price = prices[len(prices) // 2]
        avg_price = sum(prices) / len(prices)
        min_price = prices[0]
        max_price = prices[-1]
        
        # Calculate interquartile range to identify outliers
        q1 = prices[len(prices) // 4]
        q3 = prices[3 * len(prices) // 4]
        iqr = q3 - q1
        
        # Filter out extreme outliers (more than 1.5 * IQR from Q1/Q3)
        filtered_prices = [p for p in prices if q1 - 1.5 * iqr <= p <= q3 + 1.5 * iqr]
        
        if filtered_prices:
            median_price = filtered_prices[len(filtered_prices) // 2]
            avg_price = sum(filtered_prices) / len(filtered_prices)
        
        # Get date range for recency
        recent_dates = []
        for item in sold_items[:10]:  # Check most recent 10
            if 'end_time' in item:
                try:
                    recent_dates.append(datetime.fromisoformat(item['end_time'].replace('Z', '+00:00')))
                except:
                    pass
        
        days_since_last = 999
        if recent_dates:
            latest_date = max(recent_dates)
            days_since_last = (datetime.now() - latest_date).days
        
        # Calculate recommended buy price (25th percentile for good deal)
        recommended_buy = prices[len(prices) // 4]
        
        analysis = {
            'average_price': round(avg_price, 2),
            'median_price': round(median_price, 2),
            'price_range': f"${min_price:.2f} - ${max_price:.2f}",
            'lowest_price': round(min_price, 2),
            'highest_price': round(max_price, 2),
            'total_sold_analyzed': len(sold_items),
            'recommended_buy_below': round(recommended_buy, 2),
            'market_notes': self._generate_market_notes(len(sold_items), days_since_last),
            'confidence': confidence['level'],
            'confidence_reason': confidence['reason'],
            'sample_size': len(sold_items),
            'days_since_last_sale': days_since_last,
            'price_stability': 'high' if (max_price - min_price) / median_price < 0.5 else 'medium',
            'data_source': 'eBay Sold Listings',
            'aspects_used': list(aspect_filters.keys()) if aspect_filters else [],
            'sold_items_sample': sold_items[:5]  # Include sample for verification
        }
        
        logger.info(f"✅ Market analysis complete: {len(sold_items)} comps, confidence: {confidence['level']}")
        return analysis
    
    def _extract_core_aspects(self, keywords: str, item_data: Dict = None) -> Dict:
        """Extract only core structural aspects (Year, Make, Model) for filtering"""
        aspects = {}
        
        # Try to extract year from keywords
        year_match = re.search(r'\b(19[0-9]{2}|20[0-9]{2})\b', keywords)
        if year_match:
            aspects['Year'] = year_match.group(1)
        
        # Common vehicle makes to look for
        vehicle_makes = ['Toyota', 'Honda', 'Ford', 'Chevrolet', 'Chevy', 'BMW', 'Mercedes', 
                        'Audi', 'Nissan', 'Dodge', 'Jeep', 'Subaru', 'Volkswagen', 'VW']
        
        for make in vehicle_makes:
            if make.lower() in keywords.lower():
                aspects['Make'] = make
                break
        
        # If item_data has additional info, use it
        if item_data:
            title = item_data.get('title', '')
            description = item_data.get('description', '')
            
            # Look for model numbers or specific identifiers
            model_patterns = [
                r'\b([A-Z][0-9]{3,4}[A-Z]?)\b',  # Like A1234B
                r'\b([0-9]{3,4}[A-Z]?)\b',       # Like 1234A
                r'\b(S[0-9]+|Series[ -]?[0-9]+)\b',  # Like S5, Series 7
            ]
            
            all_text = f"{title} {description}".upper()
            for pattern in model_patterns:
                matches = re.findall(pattern, all_text)
                if matches:
                    aspects['Model'] = matches[0]
                    break
        
        return aspects
    
    def _calculate_confidence(self, sample_size: int) -> Dict:
        """Calculate confidence level based on sample size"""
        if sample_size >= 50:
            return {'level': 'high', 'reason': f'Excellent sample size ({sample_size}+ sold items)'}
        elif sample_size >= 20:
            return {'level': 'good', 'reason': f'Good sample size ({sample_size} sold items)'}
        elif sample_size >= 10:
            return {'level': 'medium', 'reason': f'Moderate sample size ({sample_size} sold items)'}
        elif sample_size >= 5:
            return {'level': 'low', 'reason': f'Limited sample size ({sample_size} sold items)'}
        else:
            return {'level': 'very low', 'reason': f'Insufficient sample size ({sample_size} sold items)'}
    
    def _generate_market_notes(self, sample_size: int, days_since_last: int) -> str:
        """Generate helpful market notes"""
        notes = []
        
        if sample_size >= 20:
            notes.append(f"Based on {sample_size} recent sold listings")
        elif sample_size >= 10:
            notes.append(f"Based on {sample_size} sold listings (consider broadening search)")
        else:
            notes.append(f"Limited data ({sample_size} sold listings) - use caution")
        
        if days_since_last <= 7:
            notes.append("Very recent sales (last 7 days)")
        elif days_since_last <= 30:
            notes.append(f"Recent sales (last {days_since_last} days)")
        elif days_since_last <= 90:
            notes.append(f"Sales within last {days_since_last} days")
        else:
            notes.append("Sales may be outdated")
        
        if sample_size < 20:
            notes.append("Consider checking similar models for better comps")
        
        return ". ".join(notes)
    
    def _generate_empty_analysis(self, keywords: str) -> Dict:
        """Generate analysis when no data is found"""
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
            'aspects_used': [],
            'sold_items_sample': []
        }
    
    def test_api_connection(self) -> Dict:
        """Test API connections"""
        tests = []
        
        # Test 1: Check if App ID is set
        if not self.app_id:
            tests.append("❌ App ID not set")
        else:
            tests.append(f"✅ App ID: {self.app_id[:10]}...")
        
        # Test 2: Check if OAuth token is set
        if not self.auth_token:
            tests.append("❌ OAuth token not set")
        else:
            tests.append(f"✅ OAuth token: {self.auth_token[:10]}...")
        
        # Test 3: Try Browse API
        browse_working = False
        if self.auth_token:
            try:
                params = {'q': 'test', 'limit': '1', 'filter': 'soldItems:true'}
                data = self._make_browse_api_request('/item_summary/search', params)
                if data:
                    tests.append("✅ Browse API working (sold items search)")
                    browse_working = True
                else:
                    tests.append("❌ Browse API failed")
            except Exception as e:
                tests.append(f"❌ Browse API test failed: {str(e)[:50]}")
        
        status = 'success' if browse_working else 'warning'
        
        return {
            'status': status,
            'message': 'eBay API tests completed',
            'tests': tests,
            'browse_api_working': browse_working
        }

# Global instance
ebay_api = eBayAPI(sandbox=False)