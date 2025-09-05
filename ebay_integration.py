import os
from ebaysdk.exception import ConnectionError
from ebaysdk.finding import Connection as FindingConnection
from ebaysdk.trading import Connection as TradingConnection
from ebaysdk.shopping import Connection as ShoppingConnection
from datetime import datetime, timedelta
import json
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class eBayAPI:
    def __init__(self, sandbox=True):
        self.sandbox = sandbox
        self.config = {
            'appid': os.getenv('EBAY_APP_ID', 'JustinHa-ReReSell-SBX-e11823bc1-b28e2f71'),
            'devid': os.getenv('EBAY_DEV_ID', 'a3376eb0-d27c-46a6-9a04-e6fdfed5e5fc'),
            'certid': os.getenv('EBAY_CERT_ID', 'SBX-l1823bcla4a9-aa79-4e52-8a77-735e'),
            'token': os.getenv('EBAY_AUTH_TOKEN', 'v^1.1#i^1#r^0#f^0#p^3#I^3#t^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'),  # You'll need to get this
            'siteid': '0',  # US site
            # CRITICAL FIX: Explicitly set config_file to None
            'config_file': None
        }
        
        if not sandbox:
            # Production credentials (you'll need to request these)
            self.config['appid'] = os.getenv('EBAY_PROD_APP_ID', 'JustinHa-ReReSell-PRD-b1l8c3532-f23588a1')
            # Add production certid and devid when available

    def search_completed_items(self, keywords: str, category_id: Optional[str] = None, 
                             max_results: int = 10) -> List[Dict]:
        """
        Search for completed items to analyze market prices
        """
        try:
            # CRITICAL FIX: Pass config_file=None explicitly
            api = FindingConnection(
                config_file=None,  # ← This fixes the YAML error
                appid=self.config['appid'],
                siteid='EBAY-US'
            )
            
            request_data = {
                'keywords': keywords,
                'itemFilter': [
                    {'name': 'SoldItemsOnly', 'value': True},
                    {'name': 'EndTimeFrom', 'value': (datetime.now() - timedelta(days=30)).isoformat()},
                    {'name': 'EndTimeTo', 'value': datetime.now().isoformat()}
                ],
                'paginationInput': {'entriesPerPage': max_results},
                'sortOrder': 'EndTimeSoonest'
            }
            
            if category_id:
                request_data['categoryId'] = category_id
                
            response = api.execute('findCompletedItems', request_data)
            
            items = []
            if response.reply.searchResult.item:
                for item in response.reply.searchResult.item:
                    items.append({
                        'title': item.title,
                        'price': float(item.sellingStatus.currentPrice.value),
                        'end_time': item.listingInfo.endTime,
                        'condition': item.condition.conditionDisplayName if hasattr(item, 'condition') else 'Unknown',
                        'category_id': item.primaryCategory.categoryId,
                        'category_name': item.primaryCategory.categoryName,
                        'view_count': item.listingInfo.viewCount if hasattr(item.listingInfo, 'viewCount') else 0,
                        'watch_count': item.listingInfo.watchCount if hasattr(item.listingInfo, 'watchCount') else 0
                    })
            
            return items
            
        except ConnectionError as e:
            logger.error(f"eBay API Error: {e}")
            return []

    def get_item_details(self, item_id: str) -> Optional[Dict]:
        """
        Get detailed information about a specific item
        """
        try:
            api = ShoppingConnection(
                config_file=None,  # ← Fix here too
                appid=self.config['appid']
            )
            response = api.execute('GetSingleItem', {
                'ItemID': item_id,
                'IncludeSelector': 'Details,Description,ShippingCosts'
            })
            
            item = response.reply.Item
            return {
                'title': item.Title,
                'description': item.Description if hasattr(item, 'Description') else '',
                'price': float(item.ConvertedCurrentPrice.value),
                'condition': item.ConditionDisplayName,
                'category': item.PrimaryCategoryName,
                'seller_feedback': item.Seller.FeedbackScore,
                'listing_type': item.ListingType,
                'shipping_cost': float(item.ShippingCostSummary.ShippingServiceCost.value) if hasattr(item, 'ShippingCostSummary') else 0
            }
            
        except ConnectionError as e:
            logger.error(f"eBay API Error: {e}")
            return None

    def get_category_suggestions(self, query: str) -> List[Dict]:
        """
        Get eBay category suggestions for an item
        """
        try:
            api = TradingConnection(
                config_file=None,  # ← Fix here too
                appid=self.config['appid'],
                certid=self.config['certid'], 
                devid=self.config['devid'],
                domain='api.sandbox.ebay.com',
                warnings=True
            )
            response = api.execute('GetCategorySuggestions', {
                'Query': query
            })
            
            suggestions = []
            if hasattr(response.reply, 'CategoryArray'):
                for category in response.reply.CategoryArray.Category:
                    suggestions.append({
                        'category_id': category.CategoryID,
                        'category_name': category.CategoryName,
                        'category_path': category.CategoryParentName if hasattr(category, 'CategoryParentName') else '',
                        'relevance': category.Relevance if hasattr(category, 'Relevance') else ''
                    })
            
            return suggestions
            
        except ConnectionError as e:
            logger.error(f"eBay API Error: {e}")
            return []

    def analyze_market_trends(self, category_id: str, keywords: str = "") -> Dict:
        """
        Analyze market trends for a specific category
        """
        completed_items = self.search_completed_items(keywords, category_id, max_results=50)
        
        if not completed_items:
            return {}
        
        # Calculate average price and other metrics
        prices = [item['price'] for item in completed_items]
        avg_price = sum(prices) / len(prices)
        max_price = max(prices)
        min_price = min(prices)
        
        # Calculate sell-through rate (approximate)
        items_with_watches = [item for item in completed_items if item['watch_count'] > 0]
        sell_through_rate = len(completed_items) / (len(items_with_watches) + 1)  # +1 to avoid division by zero
        
        return {
            'average_price': round(avg_price, 2),
            'price_range': f"${min_price} - ${max_price}",
            'total_listings_analyzed': len(completed_items),
            'sell_through_rate': round(sell_through_rate * 100, 2),
            'recommended_price': round(avg_price * 0.9, 2),  # 10% below average for quick sale
            'recent_examples': completed_items[:5]  # First 5 examples
        }

    def list_item(self, item_data: Dict) -> Dict:
        """
        List an item on eBay
        """
        try:
            api = TradingConnection(
                config_file=None,  # ← Fix here too
                appid=self.config['appid'],
                certid=self.config['certid'], 
                devid=self.config['devid'],
                token=self.config['token'],
                domain='api.sandbox.ebay.com',
                warnings=True
            )
            
            # Build eBay listing request
            request = {
                'Item': {
                    'Title': item_data['title'],
                    'Description': item_data['description'],
                    'PrimaryCategory': {
                        'CategoryID': item_data.get('category_id', '267')  # Default: Collectibles
                    },
                    'StartPrice': item_data['price'],
                    'ConditionID': '3000',  # Used
                    'Country': 'US',
                    'Currency': 'USD',
                    'DispatchTimeMax': '3',
                    'ListingDuration': 'Days_7',
                    'ListingType': 'FixedPriceItem',
                    'PaymentMethods': 'PayPal',
                    'PayPalEmailAddress': 'your-paypal@email.com',
                    'PictureDetails': {
                        'PictureURL': item_data.get('image_url', '')
                    },
                    'PostalCode': '90210',
                    'Quantity': '1',
                    'ReturnPolicy': {
                        'ReturnsAcceptedOption': 'ReturnsAccepted',
                        'RefundOption': 'MoneyBack',
                        'ReturnsWithinOption': 'Days_30',
                        'ShippingCostPaidByOption': 'Buyer'
                    },
                    'ShippingDetails': {
                        'ShippingType': 'Flat',
                        'ShippingServiceOptions': {
                            'ShippingServicePriority': '1',
                            'ShippingService': 'USPSPriority',
                            'ShippingServiceCost': '10.0'
                        }
                    },
                    'Site': 'US'
                }
            }
            
            response = api.execute('AddItem', request)
            
            return {
                'success': True,
                'item_id': response.reply.ItemID,
                'listing_url': f"https://www.ebay.com/itm/{response.reply.ItemID}",
                'fees': response.reply.FeeSummary
            }
            
        except ConnectionError as e:
            logger.error(f"eBay listing error: {e}")
            return {'success': False, 'error': str(e)}

# Singleton instance
ebay_api = eBayAPI(sandbox=True)