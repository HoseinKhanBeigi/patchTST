"""
Alert handler for trading signals - Telegram notifications
"""

from datetime import datetime
from typing import Dict, List
import os
import requests
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_IDS, TELEGRAM_ENABLED, ALERT_ON_STRONG_SIGNALS_ONLY, ALERT_ON_ALL_SIGNALS, DEBUG_RUNTIME


class AlertHandler:
    """Handle signal alerts via Telegram"""
    
    def __init__(self):
        self.bot_token = TELEGRAM_BOT_TOKEN
        self.chat_ids = TELEGRAM_CHAT_IDS
        self.enabled = TELEGRAM_ENABLED
        self.last_signal = None  # Track last signal to avoid spam
        
    def send_telegram_message(self, message: str, chat_id: int = None) -> bool:
        """
        Send a message to Telegram.
        
        Args:
            message: Message text to send
            chat_id: Specific chat ID (if None, sends to all configured chats)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            if DEBUG_RUNTIME:
                print("Telegram notifications disabled")
            return False
        
        if not self.bot_token:
            if DEBUG_RUNTIME:
                print("Telegram bot token not configured")
            return False
        
        chat_ids_to_notify = [chat_id] if chat_id else self.chat_ids
        
        if not chat_ids_to_notify:
            if DEBUG_RUNTIME:
                print("No Telegram chat IDs configured")
            return False
        
        success = True
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        
        for cid in chat_ids_to_notify:
            try:
                payload = {
                    'chat_id': cid,
                    'text': message,
                    'parse_mode': 'HTML'
                }
                
                response = requests.post(url, json=payload, timeout=10)
                response.raise_for_status()
                
                result = response.json()
                if result.get('ok'):
                    if DEBUG_RUNTIME:
                        print(f"✓ Telegram message sent to chat {cid}")
                else:
                    success = False
                    error_desc = result.get('description', 'Unknown error')
                    print(f"✗ Telegram API error: {error_desc}")
                    
            except requests.exceptions.RequestException as e:
                success = False
                error_msg = str(e)
                print(f"✗ Failed to send Telegram message to {cid}: {error_msg}")
                if hasattr(e, 'response') and e.response is not None:
                    try:
                        error_data = e.response.json()
                        print(f"  API Error: {error_data}")
                    except:
                        print(f"  Response: {e.response.text}")
        
        return success
    
    def format_signal_message(self, signal_type: str, signal_strength: int, 
                            confidence: str, current_price: float, 
                            predicted_price: float, change_pct: float,
                            reasons: List[str], timestamp: datetime) -> str:
        """
        Format a trading signal message for Telegram.
        """
        # Emoji based on signal type
        emoji_map = {
            "STRONG BUY": "🟢🔥",
            "BUY": "🟢",
            "WEAK BUY": "🟡",
            "NEUTRAL": "⚪",
            "WEAK SELL": "🟡",
            "SELL": "🔴",
            "STRONG SELL": "🔴🔥"
        }
        
        emoji = emoji_map.get(signal_type, "⚪")
        
        # Format message
        message = f"""
{emoji} <b>{signal_type}</b> {emoji}

💰 <b>Bitcoin Price Alert</b>
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 <b>Current Price:</b> ${current_price:,.2f}
🎯 <b>Predicted (5min):</b> ${predicted_price:,.2f}
📈 <b>Change:</b> {change_pct:+.2f}%

💪 <b>Signal Strength:</b> {signal_strength}/10
🎯 <b>Confidence:</b> {confidence}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
<b>Key Reasons:</b>
"""
        
        # Add top 3 reasons
        for i, reason in enumerate(reasons[:3], 1):
            message += f"  {i}. {reason}\n"
        
        message += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⏰ {timestamp.strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        return message.strip()
    
    def should_alert(self, signal_type: str) -> bool:
        """
        Determine if we should send an alert for this signal.
        """
        if DEBUG_RUNTIME:
            print(f"Checking if should alert for: {signal_type}")
            print(f"  ALERT_ON_ALL_SIGNALS: {ALERT_ON_ALL_SIGNALS}")
            print(f"  ALERT_ON_STRONG_SIGNALS_ONLY: {ALERT_ON_STRONG_SIGNALS_ONLY}")
        
        if not ALERT_ON_ALL_SIGNALS and not ALERT_ON_STRONG_SIGNALS_ONLY:
            if DEBUG_RUNTIME:
                print("  → No alert settings enabled")
            return False
        
        if ALERT_ON_STRONG_SIGNALS_ONLY:
            result = "STRONG" in signal_type
            if DEBUG_RUNTIME:
                print(f"  → STRONG signals only: {result}")
            return result
        
        # Alert on all signals if enabled (including NEUTRAL, WEAK BUY, WEAK SELL)
        # This includes: STRONG BUY, BUY, WEAK BUY, NEUTRAL, WEAK SELL, SELL, STRONG SELL
        result = True  # Send alert for ANY signal type when ALERT_ON_ALL_SIGNALS is True
        if DEBUG_RUNTIME:
            print(f"  → All signals enabled (including NEUTRAL): {result}")
        return result
    
    def send_signal_alert(self, signal_type: str, signal_strength: int,
                         confidence: str, current_price: float,
                         predicted_price: float, change_pct: float,
                         reasons: List[str], timestamp: datetime) -> bool:
        """
        Send a trading signal alert via Telegram.
        
        Returns:
            True if alert was sent, False otherwise
        """
        if DEBUG_RUNTIME:
            print(f"\n[ALERT] Processing signal: {signal_type}")
        
        # Check if we should alert
        if not self.should_alert(signal_type):
            if DEBUG_RUNTIME:
                print(f"[ALERT] Skipping alert for {signal_type} (not configured)")
            return False
        
        # Avoid spam - don't send same signal type repeatedly
        if self.last_signal == signal_type:
            if DEBUG_RUNTIME:
                print(f"[ALERT] Skipping duplicate {signal_type} signal")
            return False
        
        if DEBUG_RUNTIME:
            print(f"[ALERT] Sending {signal_type} alert to Telegram...")
        
        # Format and send message
        message = self.format_signal_message(
            signal_type, signal_strength, confidence,
            current_price, predicted_price, change_pct,
            reasons, timestamp
        )
        
        success = self.send_telegram_message(message)
        
        if success:
            self.last_signal = signal_type
            if DEBUG_RUNTIME:
                print(f"[ALERT] ✓ Alert sent successfully!")
        else:
            if DEBUG_RUNTIME:
                print(f"[ALERT] ✗ Failed to send alert")
        
        return success
    
    def send_price_update(self, current_price: float, predicted_price: float,
                         change_pct: float, timestamp: datetime) -> bool:
        """
        Send a simple price update (for periodic updates).
        """
        message = f"""
📊 <b>Bitcoin Price Update</b>
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💰 Current: ${current_price:,.2f}
🎯 Predicted (5min): ${predicted_price:,.2f}
📈 Change: {change_pct:+.2f}%
⏰ {timestamp.strftime('%H:%M:%S')}
"""
        
        return self.send_telegram_message(message.strip())

