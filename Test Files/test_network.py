# test_network.py
import requests
import socket

def test_network():
    print("🔍 Testing network connectivity...")
    
    # Test basic internet connectivity
    try:
        response = requests.get('https://www.google.com', timeout=10)
        print("✅ Internet connectivity: OK")
    except Exception as e:
        print(f"❌ Internet connectivity failed: {e}")
        return False
    
    # Test DNS resolution for eBay
    try:
        socket.gethostbyname('api.sandbox.ebay.com')
        print("✅ eBay DNS resolution: OK")
    except Exception as e:
        print(f"❌ eBay DNS resolution failed: {e}")
        return False
    
    # Test connection to eBay API
    try:
        response = requests.get('https://api.sandbox.ebay.com/ws/api.dll', timeout=10)
        print("✅ eBay API reachable: OK")
    except Exception as e:
        print(f"❌ eBay API connection failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    test_network()