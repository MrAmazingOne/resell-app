import os
import requests
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class eBayAPI:
    def __init__(self, sandbox=True):
        self.sandbox = sandbox
        self.app_id = os.getenv('EBAY_APP_ID')
        self.dev_id = os.getenv('EBAY_DEV_ID')  
        self.cert_id = os.getenv('EBAY_CERT_ID')
        self.auth_token = os.getenv('EBAY_AUTH_TOKEN')
        
        # Use Finding API for completed listings (no auth required for basic searches)
        if sandbox:
            self.finding_base_url = "https://svcs.sandbox.ebay.com/services/search/FindingService/v1"
            self.shopping_base_url = "https://open.api.sandbox.ebay.com/shopping"
        else:
            self.finding_base_url = "https://svcs.ebay.com/services/search/FindingService/v1"
            self.shopping_base_url = "https://open.api.ebay.com/shopping"

    def search_completed_items(self, keywords: str, category_id: Optional[str] = None, 
                             max_results: int = 20) -> List[Dict]:
        """
        Search for completed/sold items using eBay Finding API
        This provides real market data for pricing analysis
        """
        try:
            # Use Finding API's findCompletedItems - this works without authentication
            params = {
                'OPERATION-NAME': 'findCompletedItems',
                'SERVICE-VERSION': '1.0.0',
                'SECURITY-APPNAME': self.app_id,
                'RESPONSE-DATA-FORMAT': 'JSON',
                'REST-PAYLOAD': '',
                'keywords': keywords,
                'paginationInput.entriesPerPage': min(max_results, 100),
                'itemFilter(0).name': 'SoldItemsOnly',
                'itemFilter(0).value': 'true',
                'itemFilter(1).name': 'ListingType',
                'itemFilter(1).value': 'FixedPrice',
                'sortOrder': 'EndTimeSoonest'
            }
            
            if category_id:
                params['categoryId'] = category_id
            
            response = requests.get(self.finding_base_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Parse the response
            items = []
            if 'findCompletedItemsResponse' in data:
                search_result = data['findCompletedItemsResponse'][0]
                if 'searchResult' in search_result and search_result['searchResult'][0]['@count'] != '0':
                    for item in search_result['searchResult'][0]['item']:
                        try:
                            # Extract price - handle both sold and current price
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
            
        except requests.RequestException as e:
            logger.error(f"eBay Finding API Error: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error in completed items search: {e}")
            return []

    def get_current_listings(self, keywords: str, max_results: int = 10) -> List[Dict]:
        """
        Search current active listings for comparison
        """
        try:
            params = {
                'OPERATION-NAME': 'findItemsByKeywords',
                'SERVICE-VERSION': '1.0.0',
                'SECURITY-APPNAME': self.app_id,
                'RESPONSE-DATA-FORMAT': 'JSON',
                'REST-PAYLOAD': '',
                'keywords': keywords,
                'paginationInput.entriesPerPage': min(max_results, 100),
                'itemFilter(0).name': 'ListingType',
                'itemFilter(0).value': 'FixedPrice',
                'sortOrder': 'BestMatch'
            }
            
            response = requests.get(self.finding_base_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            items = []
            if 'findItemsByKeywordsResponse' in data:
                search_result = data['findItemsByKeywordsResponse'][0]
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
                                'time_left': item['listingInfo'][0]['endTime'][0]
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
            completed_items = self.search_completed_items(keywords, category_id, max_results=50)
            current_items = self.get_current_listings(keywords, max_results=20)
            
            if not completed_items:
                return self._generate_estimated_analysis(keywords)
            
            # Calculate statistics from completed items
            sold_prices = [item['price'] for item in completed_items if item['price'] > 0]
            
            if not sold_prices:
                return self._generate_estimated_analysis(keywords)
            
            avg_price = sum(sold_prices) / len(sold_prices)
            min_price = min(sold_prices)
            max_price = max(sold_prices)
            
            # Calculate recommended price (slightly below average for quick sale)
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
        Get category suggestions using eBay Shopping API (no auth required)
        """
        try:
            # Use a simpler approach - search for items and extract their categories
            current_items = self.get_current_listings(query, max_results=20)
            
            categories = {}
            for item in current_items:
                cat_name = item.get('category_name', 'Other')
                if cat_name in categories:
                    categories[cat_name] += 1
                else:
                    categories[cat_name] = 1
            
            # Return top categories sorted by frequency
            suggestions = []
            for cat_name, count in sorted(categories.items(), key=lambda x: x[1], reverse=True)[:5]:
                suggestions.append({
                    'category_name': cat_name,
                    'relevance': count,
                    'category_id': '267'  # Default to collectibles
                })
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Category suggestions error: {e}")
            return [{'category_name': 'Collectibles', 'relevance': 1, 'category_id': '267'}]

# Singleton instance
ebay_api = eBayAPI(sandbox=True)