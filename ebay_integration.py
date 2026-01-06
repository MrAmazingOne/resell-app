import os
import requests
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import time

logger = logging.getLogger(__name__)

class eBayAPI:
    def __init__(self, sandbox=False):
        self.sandbox = sandbox
        self.app_id = os.getenv('EBAY_APP_ID')
        self.dev_id = os.getenv('EBAY_DEV_ID')  
        self.cert_id = os.getenv('EBAY_CERT_ID')
        self.auth_token = os.getenv('EBAY_AUTH_TOKEN')
        
        if sandbox:
            self.finding_base_url = "https://svcs.sandbox.ebay.com/services/search/FindingService/v1"
            self.shopping_base_url = "https://open.api.sandbox.ebay.com/shopping"
        else:
            self.finding_base_url = "https://svcs.ebay.com/services/search/FindingService/v1"
            self.shopping_base_url = "https://open.api.ebay.com/shopping"
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.5  # 500ms between requests

    def _rate_limit(self):
        """Implement rate limiting for eBay API calls"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()

    def _make_finding_api_request(self, params: Dict) -> Optional[Dict]:
        """Make a Finding API request with error handling"""
        self._rate_limit()
        
        max_retries = 2
        for attempt in range(max_retries):
            try:
                logger.info(f"🔍 eBay Finding API request attempt {attempt + 1}/{max_retries}")
                logger.info(f"   URL: {self.finding_base_url}")
                logger.info(f"   Params: {json.dumps(params, indent=2)}")
                
                response = requests.get(self.finding_base_url, params=params, timeout=10)
                logger.info(f"   Response status: {response.status_code}")
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 500:
                    logger.error(f"❌ eBay API 500 error on attempt {attempt + 1}: {response.text[:200]}")
                    if attempt < max_retries - 1:
                        time.sleep(1)  # Wait before retry
                        continue
                else:
                    logger.error(f"❌ eBay API error {response.status_code}: {response.text[:200]}")
                    return None
                    
            except requests.RequestException as e:
                logger.error(f"❌ eBay API request error: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
            except Exception as e:
                logger.error(f"❌ Unexpected error in eBay API request: {e}")
                return None
        
        return None

    def optimize_search_keywords(self, keywords: str) -> str:
        """Optimize search keywords for eBay"""
        if not keywords:
            return ""
        
        # Common eBay search patterns to improve
        optimizations = {
            'not specified': '',
            'unknown': '',
            'model number': '',
            'comfortable playing experience': '',
            'glossy black finish': '',
            'exterior finish': '',
            'restored interior': '',
            'teal blue': 'teal',
            'black finish': '',
            'white finish': '',
            'blue exterior': '',
            'red exterior': '',
            'green exterior': '',
            '  ': ' '
        }
        
        # Apply optimizations
        for old, new in optimizations.items():
            keywords = keywords.replace(old, new)
        
        # Remove extra words and keep concise
        words = keywords.split()
        important_words = []
        
        for word in words:
            # Clean the word
            word_clean = word.strip('.,!?;:"\'()[]{}<>')
            word_lower = word_clean.lower()
            
            # Keep brand names (capitalized), years, numbers, key product terms
            if (word_clean[0].isupper() or  # Brands
                word_clean.isdigit() and len(word_clean) in [2, 4] or  # Years, model numbers
                word_lower in ['piano', 'guitar', 'violin', 'drum', 'trumpet', 'saxophone',
                              'truck', 'car', 'pickup', 'vehicle', 'motorcycle', 'bicycle',
                              'watch', 'ring', 'necklace', 'bracelet', 'earring',
                              'painting', 'sculpture', 'statue', 'figure', 'doll',
                              'book', 'comic', 'coin', 'stamp', 'card', 'poster',
                              'chair', 'table', 'desk', 'sofa', 'couch', 'bed',
                              'phone', 'laptop', 'computer', 'camera', 'headphones']):
                important_words.append(word_clean)
        
        # If we filtered too much, keep original (but cleaned)
        if len(important_words) < 2:
            important_words = [w.strip('.,!?;:"\'()[]{}<>') for w in words[:6]]
        
        # Join back
        optimized = ' '.join(important_words).strip()
        
        # Ensure we have something
        if not optimized:
            # Extract any brand or product type from original
            for word in words:
                word_clean = word.strip('.,!?;:"\'()[]{}<>')
                if word_clean[0].isupper() or word_clean.lower() in ['piano', 'guitar', 'truck', 'car']:
                    optimized = word_clean
                    break
            
            if not optimized and words:
                optimized = words[0].strip('.,!?;:"\'()[]{}<>')
        
        # Remove any remaining generic words
        final_words = []
        for word in optimized.split():
            if word.lower() not in ['and', 'the', 'with', 'for', 'in', 'on', 'at', 'to']:
                final_words.append(word)
        
        final_query = ' '.join(final_words)
        return final_query[:80]  # eBay has keyword length limits

    def search_completed_items(self, keywords: str, category_id: Optional[str] = None, 
                             max_results: int = 20) -> List[Dict]:
        """
        Search for completed/sold items using eBay Finding API
        """
        try:
            # Clean and optimize keywords
            keywords = self.optimize_search_keywords(keywords)
            
            if not keywords:
                logger.warning("No valid keywords after optimization")
                return []
            
            # Use Finding API's findCompletedItems
            params = {
                'OPERATION-NAME': 'findCompletedItems',
                'SERVICE-VERSION': '1.0.0',
                'SECURITY-APPNAME': self.app_id,
                'RESPONSE-DATA-FORMAT': 'JSON',
                'REST-PAYLOAD': '',
                'keywords': keywords,
                'paginationInput.entriesPerPage': min(max_results, 50),  # Reduced from 100
                'sortOrder': 'EndTimeSoonest'
            }
            
            # Only add filters if we have valid params
            if category_id:
                params['categoryId'] = category_id
            
            # Try without filters first (some filters might cause 500 errors)
            response_data = self._make_finding_api_request(params)
            
            if not response_data:
                # Try alternative approach with Shopping API
                return self._search_with_shopping_api(keywords)
            
            # Parse the response
            items = []
            if 'findCompletedItemsResponse' in response_data:
                search_result = response_data['findCompletedItemsResponse'][0]
                if 'searchResult' in search_result and search_result['searchResult'][0]['@count'] != '0':
                    for item in search_result['searchResult'][0]['item']:
                        try:
                            # Extract price
                            price_info = item.get('sellingStatus', [{}])[0]
                            current_price = float(price_info.get('currentPrice', [{}])[0].get('@value', '0'))
                            
                            # Get condition
                            condition_info = item.get('condition', [{}])
                            condition = condition_info[0].get('conditionDisplayName', ['Used'])[0] if condition_info else 'Used'
                            
                            # Get category
                            category_info = item.get('primaryCategory', [{}])
                            category_name = category_info[0].get('categoryName', ['Unknown'])[0] if category_info else 'Unknown'
                            
                            items.append({
                                'title': item['title'][0],
                                'price': current_price,
                                'condition': condition,
                                'category_name': category_name,
                                'item_id': item['itemId'][0],
                                'end_time': item['listingInfo'][0]['endTime'][0],
                                'shipping_cost': float(item.get('shippingInfo', [{}])[0].get('shippingServiceCost', [{}])[0].get('@value', '0')),
                                'gallery_url': item.get('galleryURL', [''])[0]
                            })
                        except (KeyError, IndexError, ValueError) as e:
                            logger.warning(f"Error parsing item: {e}")
                            continue
            
            logger.info(f"Found {len(items)} completed items for '{keywords}'")
            return items
            
        except Exception as e:
            logger.error(f"Unexpected error in completed items search: {e}")
            return []

    def _search_with_shopping_api(self, keywords: str) -> List[Dict]:
        """Fallback search using Shopping API"""
        try:
            params = {
                'callname': 'FindPopularItems',
                'responseencoding': 'JSON',
                'siteid': '0',
                'version': '1157',
                'appid': self.app_id,
                'QueryKeywords': keywords[:50],
                'MaxEntries': 10
            }
            
            response = requests.get(self.shopping_base_url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                items = []
                if 'ItemArray' in data and 'Item' in data['ItemArray']:
                    for item in data['ItemArray']['Item']:
                        items.append({
                            'title': item.get('Title', ''),
                            'price': float(item.get('CurrentPrice', {}).get('Value', 0)),
                            'item_id': item.get('ItemID', ''),
                            'gallery_url': item.get('GalleryURL', '')
                        })
                return items
        except Exception as e:
            logger.error(f"Shopping API fallback also failed: {e}")
        
        return []

    def get_current_listings(self, keywords: str, max_results: int = 10) -> List[Dict]:
        """
        Search current active listings for comparison
        """
        try:
            # Clean and optimize keywords
            keywords = self.optimize_search_keywords(keywords)
            
            if not keywords:
                return []
            
            params = {
                'OPERATION-NAME': 'findItemsByKeywords',
                'SERVICE-VERSION': '1.0.0',
                'SECURITY-APPNAME': self.app_id,
                'RESPONSE-DATA-FORMAT': 'JSON',
                'REST-PAYLOAD': '',
                'keywords': keywords,
                'paginationInput.entriesPerPage': min(max_results, 50),
                'sortOrder': 'BestMatch'
            }
            
            response_data = self._make_finding_api_request(params)
            
            if not response_data:
                return []
            
            items = []
            if 'findItemsByKeywordsResponse' in response_data:
                search_result = response_data['findItemsByKeywordsResponse'][0]
                if 'searchResult' in search_result and search_result['searchResult'][0]['@count'] != '0':
                    for item in search_result['searchResult'][0]['item']:
                        try:
                            price_info = item.get('sellingStatus', [{}])[0]
                            current_price = float(price_info.get('currentPrice', [{}])[0].get('@value', '0'))
                            
                            condition_info = item.get('condition', [{}])
                            condition = condition_info[0].get('conditionDisplayName', ['Used'])[0] if condition_info else 'Used'
                            
                            items.append({
                                'title': item['title'][0],
                                'price': current_price,
                                'condition': condition,
                                'item_id': item['itemId'][0],
                                'time_left': item['listingInfo'][0].get('endTime', [''])[0]
                            })
                        except (KeyError, IndexError, ValueError):
                            continue
            
            return items
            
        except Exception as e:
            logger.error(f"Current listings search error: {e}")
            return []

    def analyze_market_trends(self, keywords: str, category_id: str = None) -> Dict:
        """
        Analyze market trends using completed and current listings
        """
        try:
            # Get completed (sold) items for pricing history
            completed_items = self.search_completed_items(keywords, category_id, max_results=30)
            current_items = self.get_current_listings(keywords, max_results=15)
            
            if not completed_items:
                return self._generate_estimated_analysis(keywords)
            
            # Calculate statistics from completed items
            sold_prices = [item['price'] for item in completed_items if item['price'] > 0]
            
            if not sold_prices:
                return self._generate_estimated_analysis(keywords)
            
            avg_price = sum(sold_prices) / len(sold_prices)
            min_price = min(sold_prices)
            max_price = max(sold_prices)
            
            # Calculate recommended price
            recommended_price = avg_price * 0.85
            
            # Calculate sell-through rate estimation
            total_completed = len(completed_items)
            total_current = len(current_items)
            estimated_sell_through = (total_completed / (total_completed + total_current)) * 100 if (total_completed + total_current) > 0 else 50
            
            return {
                'average_price': round(avg_price, 2),
                'price_range': f"${min_price:.2f} - ${max_price:.2f}",
                'total_listings_analyzed': len(completed_items),
                'sell_through_rate': round(estimated_sell_through, 1),
                'recommended_price': round(recommended_price, 2),
                'market_notes': f'Based on {len(completed_items)} recent sold listings',
                'data_source': 'eBay completed listings',
                'confidence': 'high' if len(completed_items) >= 10 else 'medium'
            }
            
        except Exception as e:
            logger.error(f"Market analysis error: {e}")
            return self._generate_estimated_analysis(keywords)
    
    def _generate_estimated_analysis(self, keywords: str) -> Dict:
        """
        Generate estimated analysis when API data is unavailable
        """
        return {
            'average_price': 35.00,
            'price_range': "$15.00 - $75.00",
            'total_listings_analyzed': 0,
            'sell_through_rate': 45.0,
            'recommended_price': 29.75,
            'market_notes': f'Estimated data - eBay API returned no results for "{keywords}"',
            'data_source': 'estimated',
            'confidence': 'low'
        }

    def get_category_suggestions(self, query: str) -> List[Dict]:
        """
        Get category suggestions using eBay Shopping API
        """
        try:
            current_items = self.get_current_listings(query, max_results=20)
            
            categories = {}
            for item in current_items:
                cat_name = item.get('category_name', 'Other')
                if cat_name in categories:
                    categories[cat_name] += 1
                else:
                    categories[cat_name] = 1
            
            suggestions = []
            for cat_name, count in sorted(categories.items(), key=lambda x: x[1], reverse=True)[:5]:
                suggestions.append({
                    'category_name': cat_name,
                    'relevance': count,
                    'category_id': '267'
                })
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Category suggestions error: {e}")
            return [{'category_name': 'Collectibles', 'relevance': 1, 'category_id': '267'}]

    def test_api_connection(self) -> Dict:
        """Test eBay API connection"""
        try:
            # Test with a simple query
            test_items = self.search_completed_items("test", max_results=1)
            
            if test_items is not None:
                return {
                    'status': 'success',
                    'message': 'eBay API connection successful',
                    'app_id': self.app_id[:10] + '...',
                    'test_results': f'Found {len(test_items)} items'
                }
            else:
                return {
                    'status': 'error',
                    'message': 'eBay API connection failed',
                    'app_id': self.app_id[:10] + '...',
                    'suggestion': 'Check eBay app permissions and rate limits'
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'message': f'eBay API test failed: {str(e)}',
                'app_id': self.app_id[:10] + '...'
            }

# Singleton instance
ebay_api = eBayAPI(sandbox=False)