import os
import requests
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import time

logger = logging.getLogger(__name__)

class eBayAPI:
    def __init__(self, sandbox=False):
        self.sandbox = sandbox
        self.app_id = os.getenv('EBAY_APP_ID')
        self.auth_token = os.getenv('EBAY_AUTH_TOKEN')  # For Browse API
        
        logger.info(f"🔧 eBay API Configuration:")
        logger.info(f"   Environment: {'SANDBOX' if sandbox else 'PRODUCTION'}")
        logger.info(f"   App ID: {self.app_id[:10]}..." if self.app_id else "   App ID: NOT SET")
        
        if sandbox:
            self.browse_base_url = "https://api.sandbox.ebay.com/buy/browse/v1"
            self.finding_base_url = "https://svcs.sandbox.ebay.com/services/search/FindingService/v1"
        else:
            self.browse_base_url = "https://api.ebay.com/buy/browse/v1"
            self.finding_base_url = "https://svcs.ebay.com/services/search/FindingService/v1"

    def _make_browse_api_request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Make a request to eBay Browse API (modern OAuth API)"""
        try:
            headers = {
                'Authorization': f'Bearer {self.auth_token}',
                'Content-Type': 'application/json',
                'X-EBAY-C-MARKETPLACE-ID': 'EBAY_US'
            }
            
            url = f"{self.browse_base_url}{endpoint}"
            
            logger.info(f"🔍 Making Browse API request")
            logger.info(f"   URL: {url}")
            logger.info(f"   Params: {params}")
            
            response = requests.get(url, headers=headers, params=params, timeout=10)
            
            logger.info(f"   Status: {response.status_code}")
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 401:
                logger.error("❌ Authentication failed - invalid/missing OAuth token")
                logger.error("   Run /ebay/auth to get a new token")
                return None
            elif response.status_code == 403:
                logger.error("❌ Forbidden - missing scope or insufficient permissions")
                logger.error("   Required scope: https://api.ebay.com/oauth/api_scope")
                return None
            else:
                logger.error(f"❌ Browse API error: {response.status_code}")
                logger.error(f"   Response: {response.text[:500]}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Browse API request error: {e}")
            return None

    def _make_finding_api_request(self, params: Dict) -> Optional[Dict]:
        """Make a request to eBay Finding API (legacy API)"""
        try:
            # Add required parameters
            params['SECURITY-APPNAME'] = self.app_id
            params['RESPONSE-DATA-FORMAT'] = 'JSON'
            params['SERVICE-VERSION'] = '1.0.0'
            
            logger.info(f"🔍 Making Finding API request")
            logger.info(f"   URL: {self.finding_base_url}")
            logger.info(f"   Operation: {params.get('OPERATION-NAME', 'unknown')}")
            
            response = requests.get(self.finding_base_url, params=params, timeout=10)
            
            logger.info(f"   Status: {response.status_code}")
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 500:
                # Try to parse the error
                try:
                    error_data = response.json()
                    error_msg = error_data.get('errorMessage', [{}])[0].get('error', [{}])[0].get('message', ['Unknown'])[0]
                    logger.error(f"❌ Finding API 500: {error_msg}")
                except:
                    logger.error(f"❌ Finding API 500: {response.text[:200]}")
                return None
            else:
                logger.error(f"❌ Finding API error {response.status_code}: {response.text[:200]}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Finding API request error: {e}")
            return None

    def search_sold_items(self, keywords: str, limit: int = 5) -> List[Dict]:
        """
        Search for sold items using Browse API's item_summary/search
        This is the modern way to search eBay
        """
        if not self.auth_token:
            logger.warning("⚠️ No OAuth token - using fallback search")
            return self._fallback_search(keywords, limit)
        
        try:
            params = {
                'q': keywords,
                'limit': str(limit),
                'filter': 'soldItems'  # This is key - only sold items
            }
            
            data = self._make_browse_api_request('/item_summary/search', params)
            
            if not data or 'itemSummaries' not in data:
                logger.warning("⚠️ Browse API returned no items")
                return self._fallback_search(keywords, limit)
            
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
            
            logger.info(f"✅ Browse API found {len(items)} sold items")
            return items
            
        except Exception as e:
            logger.error(f"❌ Browse API search error: {e}")
            return self._fallback_search(keywords, limit)

    def _fallback_search(self, keywords: str, limit: int) -> List[Dict]:
        """
        Fallback to Finding API if Browse API fails
        """
        try:
            params = {
                'OPERATION-NAME': 'findCompletedItems',
                'keywords': keywords,
                'paginationInput.entriesPerPage': str(limit),
                'sortOrder': 'EndTimeSoonest'
            }
            
            data = self._make_finding_api_request(params)
            
            if not data:
                return []
            
            items = []
            if 'findCompletedItemsResponse' in data:
                search_result = data['findCompletedItemsResponse'][0]
                if 'searchResult' in search_result and search_result['searchResult'][0]['@count'] != '0':
                    for item in search_result['searchResult'][0]['item'][:limit]:
                        try:
                            price_info = item.get('sellingStatus', [{}])[0]
                            current_price = float(price_info.get('currentPrice', [{}])[0].get('@value', '0'))
                            
                            items.append({
                                'title': item['title'][0],
                                'price': current_price,
                                'item_id': item['itemId'][0],
                                'condition': item.get('condition', [{}])[0].get('conditionDisplayName', [''])[0],
                                'end_time': item.get('listingInfo', [{}])[0].get('endTime', [''])[0]
                            })
                        except:
                            continue
            
            logger.info(f"✅ Finding API found {len(items)} items")
            return items
            
        except Exception as e:
            logger.error(f"❌ Fallback search error: {e}")
            return []

    def search_current_listings(self, keywords: str, limit: int = 3) -> List[Dict]:
        """Search current listings using Browse API"""
        if not self.auth_token:
            logger.warning("⚠️ No OAuth token - skipping current listings")
            return []
        
        try:
            params = {
                'q': keywords,
                'limit': str(limit)
                # No filter - get current listings
            }
            
            data = self._make_browse_api_request('/item_summary/search', params)
            
            if not data or 'itemSummaries' not in data:
                return []
            
            items = []
            for item in data['itemSummaries'][:limit]:
                try:
                    price = item.get('price', {}).get('value', '0')
                    items.append({
                        'title': item.get('title', ''),
                        'price': float(price),
                        'item_id': item.get('itemId', ''),
                        'condition': item.get('condition', '')
                    })
                except:
                    continue
            
            return items
            
        except Exception as e:
            logger.error(f"❌ Current listings error: {e}")
            return []

    def analyze_market_trends(self, keywords: str) -> Dict:
        """
        Analyze market trends using Browse API for sold items
        """
        logger.info(f"📊 Analyzing market for: '{keywords}'")
        
        # Get sold items
        sold_items = self.search_sold_items(keywords, limit=10)
        
        if not sold_items:
            logger.warning("⚠️ No sold items found - using estimated data")
            return self._generate_estimated_analysis(keywords)
        
        # Calculate statistics
        prices = [item['price'] for item in sold_items if item['price'] > 0]
        
        if not prices:
            return self._generate_estimated_analysis(keywords)
        
        avg_price = sum(prices) / len(prices)
        min_price = min(prices)
        max_price = max(prices)
        
        # Get current listings for comparison
        current_items = self.search_current_listings(keywords, limit=5)
        current_avg = None
        if current_items:
            current_prices = [item['price'] for item in current_items if item['price'] > 0]
            if current_prices:
                current_avg = sum(current_prices) / len(current_prices)
        
        analysis = {
            'average_price': round(avg_price, 2),
            'price_range': f"${min_price:.2f} - ${max_price:.2f}",
            'total_sold_analyzed': len(sold_items),
            'recommended_price': round(avg_price * 0.85, 2),
            'market_notes': f'Based on {len(sold_items)} recent eBay sales',
            'data_source': 'eBay Browse API',
            'confidence': 'high' if len(sold_items) >= 5 else 'medium',
            'api_used': 'Browse API' if self.auth_token else 'Finding API'
        }
        
        if current_avg:
            analysis['current_average'] = round(current_avg, 2)
            analysis['competition_count'] = len(current_items)
        
        return analysis
    
    def _generate_estimated_analysis(self, keywords: str) -> Dict:
        """Generate estimated analysis when API fails"""
        return {
            'average_price': 75.00,
            'price_range': "$50.00 - $150.00",
            'total_sold_analyzed': 0,
            'recommended_price': 63.75,
            'market_notes': 'Estimated pricing - eBay API limited',
            'data_source': 'estimated',
            'confidence': 'low',
            'api_used': 'none'
        }

    def test_api_connection(self) -> Dict:
        """Test API connections"""
        tests = []
        
        # Test 1: Check if App ID is set
        if not self.app_id:
            tests.append("❌ App ID not set")
        else:
            tests.append(f"✅ App ID: {self.app_id[:10]}...")
        
        # Test 2: Check if OAuth token is set (for Browse API)
        if not self.auth_token:
            tests.append("⚠️ OAuth token not set (Browse API will use Finding API fallback)")
        else:
            tests.append(f"✅ OAuth token: {self.auth_token[:10]}...")
        
        # Test 3: Try Browse API if token exists
        browse_working = False
        if self.auth_token:
            try:
                # Simple test request
                params = {'q': 'test', 'limit': '1'}
                data = self._make_browse_api_request('/item_summary/search', params)
                if data:
                    tests.append("✅ Browse API working")
                    browse_working = True
                else:
                    tests.append("❌ Browse API failed")
            except:
                tests.append("❌ Browse API test failed")
        
        # Test 4: Try Finding API
        finding_working = False
        try:
            params = {
                'OPERATION-NAME': 'findItemsByKeywords',
                'keywords': 'test',
                'paginationInput.entriesPerPage': '1'
            }
            data = self._make_finding_api_request(params)
            if data:
                tests.append("✅ Finding API working")
                finding_working = True
            else:
                tests.append("❌ Finding API failed")
        except:
            tests.append("❌ Finding API test failed")
        
        status = 'success' if browse_working or finding_working else 'warning'
        
        return {
            'status': status,
            'message': 'eBay API tests completed',
            'tests': tests,
            'recommendation': 'Get OAuth token at /ebay/auth for best results'
        }

# Global instance
ebay_api = eBayAPI(sandbox=False)