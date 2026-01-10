import os
import requests
import json
import logging
import re
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import time

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
            self.marketing_base_url = "https://api.sandbox.ebay.com/sell/marketing/v1"
        else:
            self.browse_base_url = "https://api.ebay.com/buy/browse/v1"
            self.taxonomy_base_url = "https://api.ebay.com/commerce/taxonomy/v1_beta"
            self.marketing_base_url = "https://api.ebay.com/sell/marketing/v1"

    def _make_api_request(self, url: str, method: str = 'GET', params: Dict = None, 
                         data: Dict = None, headers: Dict = None) -> Optional[Dict]:
        """Make generic API request"""
        try:
            if not self.auth_token:
                logger.error("❌ No OAuth token available")
                return None
                
            default_headers = {
                'Authorization': f'Bearer {self.auth_token}',
                'Content-Type': 'application/json'
            }
            
            if headers:
                default_headers.update(headers)
            
            if method.upper() == 'GET':
                response = requests.get(url, headers=default_headers, params=params, timeout=15)
            elif method.upper() == 'POST':
                response = requests.post(url, headers=default_headers, json=data, params=params, timeout=15)
            else:
                logger.error(f"❌ Unsupported method: {method}")
                return None
            
            logger.debug(f"API {method} {url}: {response.status_code}")
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:
                logger.error(f"❌ API endpoint not found: {url}")
                return None
            elif response.status_code == 403:
                logger.error(f"❌ Forbidden - missing scope or insufficient permissions")
                logger.error(f"   Response: {response.text[:200]}")
                return None
            else:
                logger.error(f"❌ API error {response.status_code}: {response.text[:200]}")
                return None
                
        except Exception as e:
            logger.error(f"❌ API request error: {e}")
            return None

    def get_category_suggestions(self, keywords: str) -> str:
        """
        Get category suggestions using eBay Taxonomy API
        Returns the best category ID for the keywords
        """
        if not keywords or keywords.strip() == "":
            return "267"  # Default Collectibles category
        
        try:
            # Try Taxonomy API first
            category_id = self._get_category_from_taxonomy_api(keywords)
            if category_id:
                return category_id
            
            # Fallback to our mapping if API fails
            return self._get_category_from_mapping(keywords)
            
        except Exception as e:
            logger.error(f"❌ Category suggestion error: {e}")
            return "267"

    def _get_category_from_taxonomy_api(self, keywords: str) -> Optional[str]:
        """Get category using official Taxonomy API"""
        try:
            # Get default category tree ID
            tree_url = f"{self.taxonomy_base_url}/get_default_category_tree_id"
            tree_data = self._make_api_request(tree_url)
            
            if not tree_data:
                logger.warning("⚠️ Could not get category tree ID")
                return None
            
            tree_id = tree_data.get('categoryTreeId')
            if not tree_id:
                return None
            
            # Get category suggestions
            suggest_url = f"{self.taxonomy_base_url}/category_tree/{tree_id}/get_category_suggestions"
            params = {'q': keywords}
            
            suggestions = self._make_api_request(suggest_url, params=params)
            
            if suggestions and 'categorySuggestions' in suggestions:
                category_suggestions = suggestions['categorySuggestions']
                if category_suggestions:
                    # Get the most relevant category
                    best_category = category_suggestions[0]
                    category_id = best_category['category']['categoryId']
                    category_name = best_category['category']['categoryName']
                    
                    logger.info(f"✅ Taxonomy API suggested: {category_name} (ID: {category_id})")
                    return category_id
            
            return None
            
        except Exception as e:
            logger.warning(f"⚠️ Taxonomy API failed, using fallback: {e}")
            return None

    def _get_category_from_mapping(self, keywords: str) -> str:
        """Fallback category mapping"""
        keywords_lower = keywords.lower()
        
        # Expanded category mapping
        category_mapping = [
            # Vehicles
            (['car', 'truck', 'vehicle', 'automobile', 'auto'], '6001'),
            (['motorcycle', 'scooter', 'atv'], '6024'),
            (['boat', 'yacht', 'marine'], '26429'),
            
            # Electronics
            (['phone', 'iphone', 'smartphone', 'cellphone'], '9355'),
            (['computer', 'laptop', 'desktop', 'pc', 'macbook'], '58058'),
            (['camera', 'dslr', 'mirrorless'], '31388'),
            (['tv', 'television', 'monitor', 'display'], '15052'),
            (['tablet', 'ipad', 'kindle'], '171485'),
            
            # Collectibles
            (['pokemon', 'pokémon', 'trading card', 'tcg'], '183454'),
            (['magic', 'mtg', 'magic the gathering'], '183454'),
            (['sports card', 'baseball card', 'basketball card'], '213'),
            (['coin', 'currency', 'banknote'], '11116'),
            (['stamp', 'philately'], '260'),
            
            # Clothing
            (['shirt', 'pants', 'jeans', 'jacket', 'hoodie'], '11450'),
            (['shoe', 'sneaker', 'boot', 'footwear'], '11450'),
            (['watch', 'wristwatch', 'timepiece'], '31387'),
            (['jewelry', 'ring', 'necklace', 'bracelet'], '281'),
            
            # Home & Garden
            (['furniture', 'chair', 'table', 'sofa', 'couch'], '3197'),
            (['tool', 'drill', 'saw', 'wrench'], '11700'),
            (['appliance', 'refrigerator', 'washer', 'dryer'], '20710'),
            
            # Entertainment
            (['game', 'video game', 'console', 'playstation', 'xbox'], '139973'),
            (['toy', 'lego', 'action figure', 'doll'], '220'),
            (['book', 'novel', 'textbook', 'comic'], '267'),
            (['music', 'cd', 'vinyl', 'record', 'album'], '11233'),
            (['movie', 'dvd', 'blu-ray', 'film'], '617'),
            
            # Sports
            (['sport', 'fitness', 'exercise', 'gym'], '888'),
            (['bicycle', 'bike', 'cycling'], '7294'),
            (['golf', 'club', 'golf club'], '1513'),
            
            # Default
            (['item', 'thing', 'object', 'stuff'], '267')
        ]
        
        for keyword_list, category_id in category_mapping:
            if any(keyword in keywords_lower for keyword in keyword_list):
                logger.info(f"✅ Mapped '{keywords_lower}' to category {category_id}")
                return category_id
        
        logger.warning(f"⚠️ No category match found for: {keywords[:50]}... - using default")
        return "267"

    def get_keyword_suggestions(self, keywords: str, category_id: str = None) -> List[str]:
        """
        Get keyword suggestions using eBay Marketing API
        Returns optimized keywords for search
        """
        try:
            # First, try to get suggestions from Marketing API
            marketing_keywords = self._get_keywords_from_marketing_api(keywords, category_id)
            if marketing_keywords:
                return marketing_keywords
            
            # Fallback to smart keyword generation
            return self._generate_smart_keywords(keywords, category_id)
            
        except Exception as e:
            logger.error(f"❌ Keyword suggestion error: {e}")
            return [keywords]  # Return original as fallback

    def _get_keywords_from_marketing_api(self, keywords: str, category_id: str = None) -> Optional[List[str]]:
        """Get keywords using official Marketing API"""
        try:
            # Marketing API requires a campaign - we'll check if one exists
            campaigns_url = f"{self.marketing_base_url}/ad_campaign"
            campaigns_data = self._make_api_request(campaigns_url)
            
            if not campaigns_data or 'campaigns' not in campaigns_data or not campaigns_data['campaigns']:
                logger.warning("⚠️ No campaigns found for Marketing API")
                return None
            
            # Use the first campaign
            campaign_id = campaigns_data['campaigns'][0]['campaignId']
            
            # Get ad groups for this campaign
            ad_groups_url = f"{self.marketing_base_url}/ad_campaign/{campaign_id}/ad_group"
            ad_groups_data = self._make_api_request(ad_groups_url)
            
            if not ad_groups_data or 'adGroups' not in ad_groups_data or not ad_groups_data['adGroups']:
                logger.warning("⚠️ No ad groups found for Marketing API")
                return None
            
            ad_group_id = ad_groups_data['adGroups'][0]['adGroupId']
            
            # Get keyword suggestions
            keywords_url = f"{self.marketing_base_url}/ad_campaign/{campaign_id}/ad_group/{ad_group_id}/suggest_keywords"
            
            request_data = {
                "keywords": [keywords],
                "maxNumOfKeywords": 10
            }
            
            if category_id:
                request_data["categoryIds"] = [category_id]
            
            suggestions_data = self._make_api_request(keywords_url, method='POST', data=request_data)
            
            if suggestions_data and 'suggestedKeywords' in suggestions_data:
                suggested_keywords = [kw['keywordText'] for kw in suggestions_data['suggestedKeywords']]
                logger.info(f"✅ Marketing API suggested {len(suggested_keywords)} keywords")
                return suggested_keywords
            
            return None
            
        except Exception as e:
            logger.warning(f"⚠️ Marketing API failed, using fallback: {e}")
            return None

    def _generate_smart_keywords(self, keywords: str, category_id: str = None) -> List[str]:
        """Generate smart keywords when API is unavailable"""
        # Basic keyword optimization
        keywords = keywords.lower().strip()
        
        # Remove common stop words
        stop_words = ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']
        words = [word for word in keywords.split() if word not in stop_words]
        
        # Create variations
        variations = []
        
        # Original keywords
        variations.append(' '.join(words))
        
        # Add brand/model variations if detected
        if any(brand in keywords for brand in ['iphone', 'samsung', 'sony', 'nike', 'adidas']):
            variations.append(f"{keywords} new")
            variations.append(f"{keywords} used")
            variations.append(f"{keywords} original")
        
        # Add condition variations
        condition_words = ['excellent', 'good', 'fair', 'mint', 'like new', 'preowned']
        for condition in condition_words[:2]:  # Add top 2 conditions
            variations.append(f"{keywords} {condition}")
        
        # Add size/color if detected
        if any(size in keywords for size in ['large', 'medium', 'small', 'xl', 'xxl']):
            variations.append(f"{keywords}")
        else:
            variations.append(f"{keywords} size")
        
        # Limit to 5 variations
        return variations[:5]

    # [Keep the rest of your existing methods: search_sold_items, analyze_market_trends, etc.]

# Global instance
ebay_api = eBayAPI(sandbox=False)