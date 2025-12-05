"""
Test Telegram bot connection
"""

from alert_handler import AlertHandler
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_IDS, TELEGRAM_ENABLED

def test_telegram():
    """Test if Telegram bot can send messages"""
    print("Testing Telegram bot connection...")
    print(f"Bot Token: {TELEGRAM_BOT_TOKEN[:10]}...")
    print(f"Chat IDs: {TELEGRAM_CHAT_IDS}")
    print(f"Enabled: {TELEGRAM_ENABLED}")
    print()
    
    alert_handler = AlertHandler()
    
    # Test simple message
    print("Sending test message...")
    test_message = "🤖 Test message from PatchTST bot!\n\nIf you receive this, Telegram integration is working! ✅"
    
    success = alert_handler.send_telegram_message(test_message)
    
    if success:
        print("✅ Message sent successfully!")
    else:
        print("❌ Failed to send message. Check:")
        print("  1. Bot token is correct")
        print("  2. Chat ID is correct")
        print("  3. You've started a conversation with the bot")
        print("  4. Internet connection is working")

if __name__ == "__main__":
    test_telegram()

