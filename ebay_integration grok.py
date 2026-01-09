import os
import requests
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import time
import statistics

logger = logging.getLogger(__name__)

class eBayAPI:
    def __init__(self, sandbox=False):
        self.sandbox = sandbox
        self.app_id = os.getenv('EBAY_APP_ID')
        self.auth_token = os.getenv('EBAY_AUTH_TOKEN')
        
        logger.info(f"🔧 eBay API Configuration: {'SANDBOX' if sandbox else 'PRODUCTION'}")
        
        if sandbox:
            self.browse_base_url = "https://api.sandbox.ebay.com/buy/browse/v1"
            self.taxonomy_base_url = "https://api.sandbox.ebay.com/commerce/taxonomy/v1_beta"
            self.marketing_base_url = "https://api.sandbox.ebay.com/sell/marketing/v1"
        else:
            self.browse_base_url = "https://api.ebay.com/buy/browse/v1"
            self.taxonomy_base_url = "https://api.ebay.com/commerce/taxonomy/v1_beta"
            self.marketing_base_url = "https://api.ebay.com/sell/marketing/v1"

    def _make_api_request(self, base_url: str, endpoint: str, method: str = 'GET', 
                         params: Dict = None, body: Dict = None, retries: int = 3) -> Optional[Dict]:
        """Generic API request with retries"""
        url = f"{base_url}{endpoint}"
        for attempt in range(retries):
            try:
                headers = {
                    'Authorization': f'Bearer {self.auth_token}',
                    'Content-Type': 'application/json',
                    'X-EBAY-C-MARKETPLACE-ID': 'EBAY_US'
                }
                
                if method == 'GET':
                    response = requests.get(url, headers=headers, params=params, timeout=12)
                elif method == 'POST':
                    response = requests.post(url, headers=headers, json=body, timeout=12)
                else:
                    raise ValueError("Unsupported method")
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code in [401, 403]:
                    logger.error(f"Auth error {response.status_code} - Check token/scopes")
                    return None
                else:
                    logger.warning(f"API error {response.status_code} (attempt {attempt+1})")
                    time.sleep(2 ** attempt)
            except Exception as e:
                logger.error(f"Request error: {e}")
                time.sleep(2 ** attempt)
        return None

    def get_category_suggestions(self, keywords: str) -> Optional[str]:
        """Get best category suggestion using Taxonomy API"""
        params = {'q': keywords}
        data = self._make_api_request(self.taxonomy_base_url, 
                                    '/category_tree/EBAY_US/get_category_suggestions', 
                                    params=params)
        if data and 'categorySuggestions' in data and data['categorySuggestions']:
            top = data['categorySuggestions'][0]
            return top['category']['categoryId']
        return None

    def search_sold_items(self, 
                         keywords: str, 
                         category_id: Optional[str] = None,
                         core_aspects: Dict[str, str] = None,  # e.g. {'Year':'1955', 'Make':'Chevrolet', 'Model':'3100'}
                         max_items: int = 100,
                         min_confidence_items: int = 20) -> Tuple[List[Dict], Dict]:
        """
        Search sold items with progressive relaxation strategy
        Returns: (sold_items, metadata)
        """
        if not self.auth_token:
            return [], {"error": "No OAuth token"}

        items = []
        metadata = {
            "query": keywords,
            "category_id": category_id,
            "filter_level": "strict",
            "sample_size": 0,
            "confidence": "low",
            "notes": []
        }

        # Build base filter
        base_filter = "soldItemsOnly:true"
        if category_id:
            base_filter += f",category_ids:{category_id}"

        # Level 1: Strict (core aspects only)
        params = {
            'q': keywords,
            'limit': '100',
            'filter': base_filter,
            'sort': '-endTime'
        }

        if core_aspects:
            aspect_str = ','.join([f"{k}:{{{v}}}" for k,v in core_aspects.items()])
            params['aspect_filter'] = f"categoryId:{category_id},{aspect_str}" if category_id else aspect_str

        data = self._make_api_request(self.browse_base_url, '/item_summary/search', params=params)
        if data and 'itemSummaries' in data:
            items = self._process_items(data['itemSummaries'])

        # If too few results → relax condition if present
        if len(items) < min_confidence_items and 'Condition' in (core_aspects or {}):
            metadata["notes"].append("Relaxed condition filter due to low sample size")
            params['aspect_filter'] = params['aspect_filter'].replace("Condition:{...}", "")  # crude but effective
            data = self._make_api_request(self.browse_base_url, '/item_summary/search', params=params)
            if data and 'itemSummaries' in data:
                items = self._process_items(data['itemSummaries'])

        # Still too few? → remove aspects completely, keep category + keywords
        if len(items) < min_confidence_items:
            metadata["notes"].append("Removed aspect filters to increase sample size")
            params.pop('aspect_filter', None)
            data = self._make_api_request(self.browse_base_url, '/item_summary/search', params=params)
            if data and 'itemSummaries' in data:
                items = self._process_items(data['itemSummaries'])

        # Final metadata
        metadata["sample_size"] = len(items)
        if len(items) >= 50:
            metadata["confidence"] = "high"
        elif len(items) >= 20:
            metadata["confidence"] = "good"
        elif len(items) >= 10:
            metadata["confidence"] = "medium"
        else:
            metadata["confidence"] = "low"
            metadata["notes"].append("Very limited comparable sales — use with caution")

        # Keep only most recent 100
        return items[:max_items], metadata

    def _process_items(self, raw_items: List[Dict]) -> List[Dict]:
        """Clean and standardize sold item data"""
        processed = []
        for item in raw_items:
            try:
                price_val = float(item.get('price', {}).get('value', 0))
                if price_val <= 0:
                    continue
                processed.append({
                    'title': item.get('title', '').strip(),
                    'price': price_val,
                    'item_id': item.get('itemId', ''),
                    'condition': item.get('condition', 'Unknown'),
                    'end_date': item.get('itemEndDate', ''),
                    'web_url': item.get('itemWebUrl', '')
                })
            except:
                continue
        return processed

    def analyze_market_trends(self, 
                            keywords: str,
                            category_id: Optional[str] = None,
                            core_aspects: Dict[str, str] = None,
                            campaign_id: str = None,
                            ad_group_id: str = None) -> Dict:
        """
        Full valuation analysis with intelligent filter relaxation
        """
        logger.info(f"Valuation analysis for: {keywords}")

        sold_items, meta = self.search_sold_items(
            keywords, 
            category_id=category_id,
            core_aspects=core_aspects
        )

        if not sold_items:
            return {
                "error": "No usable sold data found",
                "confidence": "none",
                **meta
            }

        prices = [item['price'] for item in sold_items]

        result = {
            "median_price": round(statistics.median(prices), 2),
            "average_price": round(statistics.mean(prices), 2),
            "price_range": f"${min(prices):.2f} – ${max(prices):.2f}",
            "sample_size": len(sold_items),
            "confidence": meta["confidence"],
            "notes": meta["notes"],
            "query_used": keywords,
            "category_id": meta["category_id"],
            "recommended_buy_below": round(statistics.median(prices) * 0.75, 2),
            "sample_links": [item['web_url'] for item in sold_items[:5] if item.get('web_url')],
            **meta
        }

        # Add simple trend indicator (last 30 vs previous)
        if len(sold_items) >= 15:
            now = datetime.utcnow()
            recent = [p for item in sold_items 
                     if (now - datetime.fromisoformat(item['end_date'].replace('Z', '+00:00'))).days <= 30]
            older = [p for item in sold_items if p not in recent]
            if recent and older:
                trend = "↑" if statistics.mean(recent) > statistics.mean(older) else "↓" if statistics.mean(recent) < statistics.mean(older) else "→"
                result["recent_trend"] = trend

        return result

# Global instance
ebay_api = eBayAPI(sandbox=False)