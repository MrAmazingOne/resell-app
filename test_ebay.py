import requests
import json

# Use YOUR actual App ID
app_id = "JustinHa-ReReSell-SBX-e11823bc1-b28e2f71"  

response = requests.get(
    "https://svcs.sandbox.ebay.com/services/search/FindingService/v1",
    params={
        'OPERATION-NAME': 'findCompletedItems',
        'SERVICE-VERSION': '1.0.0', 
        'SECURITY-APPNAME': app_id,
        'RESPONSE-DATA-FORMAT': 'JSON',
        'keywords': '*',
        'paginationInput.entriesPerPage': '1'
    },
    timeout=10
)

print(f"Status: {response.status_code}")
print(f"Response: {response.text}")