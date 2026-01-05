# keep_alive.py
import threading
import time
import requests
import os
import logging

logger = logging.getLogger(__name__)

class KeepAlive:
    def __init__(self, base_url=None):
        self.base_url = base_url or os.getenv('RENDER_EXTERNAL_URL', 'http://localhost:8000')
        self.is_running = False
        self.thread = None
        self.interval = 25  # Ping every 25 seconds (less than 30s timeout)
        
    def start(self):
        """Start the keep-alive thread"""
        if self.is_running:
            return
            
        self.is_running = True
        self.thread = threading.Thread(target=self._keep_alive_loop, daemon=True)
        self.thread.start()
        logger.info(f"Keep-alive started for {self.base_url}")
        
    def stop(self):
        """Stop the keep-alive thread"""
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=5)
            
    def _keep_alive_loop(self):
        """Main keep-alive loop"""
        while self.is_running:
            try:
                # Ping our own health endpoint
                response = requests.get(f"{self.base_url}/ping", timeout=10)
                if response.status_code == 200:
                    logger.debug(f"Keep-alive ping successful: {response.json()}")
                else:
                    logger.warning(f"Keep-alive ping failed: {response.status_code}")
            except Exception as e:
                logger.warning(f"Keep-alive ping error: {e}")
                
            # Wait before next ping
            for _ in range(self.interval):
                if not self.is_running:
                    break
                time.sleep(1)
                
    def ping(self):
        """Manual ping"""
        try:
            response = requests.get(f"{self.base_url}/ping", timeout=5)
            return response.status_code == 200
        except:
            return False

# Global instance
keep_alive = KeepAlive()