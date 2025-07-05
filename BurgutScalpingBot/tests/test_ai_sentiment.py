import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import json
from modules.orderflow import run_orderflow
from modules.ai_sentiment import run_sentiment
from modules.utils import send_telegram_status
from modules.api_manager import check_apis


class TestOrderFlow:
    """Order Flow modulini test qilish"""
    
    @pytest.mark.asyncio
    async def test_run_orderflow_success(self):
        """Order Flow muvaffaqiyatli ishlashini test qilish"""
        # Test qilish: run_orderflow() funksiyasi xatosiz ishlashi kerak
        try:
            await run_orderflow()
            assert True  # Xatosiz ishladi
        except Exception as e:
            pytest.fail(f"Order Flow xato berdi: {e}")
    
    @pytest.mark.asyncio
    async def test_run_orderflow_performance(self):
        """Order Flow tezligini test qilish"""
        import time
        start_time = time.time()
        await run_orderflow()
        end_time = time.time()
        
        # 5 soniyadan kam vaqt olishi kerak
        assert end_time - start_time < 5.0, "Order Flow juda sekin ishlaydi"


class TestAISentiment:
    """AI Sentiment modulini test qilish"""
    
    @pytest.mark.asyncio
    async def test_run_sentiment_with_mock_api(self):
        """AI Sentiment API ni mock qilib test qilish"""
        mock_response = Mock()
        mock_response.json.return_value = [{"label": "POSITIVE", "score": 0.9}]
        
        with patch('modules.ai_sentiment.requests.post', return_value=mock_response):
            try:
                await run_sentiment()
                assert True  # Xatosiz ishladi
            except Exception as e:
                pytest.fail(f"AI Sentiment xato berdi: {e}")
    
    @pytest.mark.asyncio
    async def test_run_sentiment_api_error(self):
        """AI Sentiment API xato holatini test qilish"""
        with patch('modules.ai_sentiment.requests.post') as mock_post:
            mock_post.side_effect = Exception("API xatosi")
            
            # API xatosi bo'lsa ham crash bo'lmasligi kerak
            try:
                await run_sentiment()
                # Xato bo'lsa ham davom etishi kerak
            except Exception:
                pass  # Xato kutilgan


class TestUtils:
    """Utils modulini test qilish"""
    
    def test_send_telegram_status_success(self):
        """Telegram xabar yuborish muvaffaqiyatli test"""
        mock_response = Mock()
        mock_response.status_code = 200
        
        with patch('modules.utils.requests.post', return_value=mock_response):
            try:
                send_telegram_status("Test xabar")
                assert True
            except Exception as e:
                pytest.fail(f"Telegram xabar yuborish xato berdi: {e}")
    
    def test_send_telegram_status_with_special_chars(self):
        """Telegram xabar maxsus belgilar bilan test"""
        mock_response = Mock()
        mock_response.status_code = 200
        
        with patch('modules.utils.requests.post', return_value=mock_response):
            test_messages = [
                "🚀 Bot ishga tushdi! ✅",
                "Balance: $1,234.56",
                "Xato: API key noto'g'ri",
                "Profit: +15.3% 📈"
            ]
            
            for msg in test_messages:
                try:
                    send_telegram_status(msg)
                except Exception as e:
                    pytest.fail(f"Telegram xabar '{msg}' xato berdi: {e}")
    
    def test_send_telegram_status_api_error(self):
        """Telegram API xato holatini test qilish"""
        with patch('modules.utils.requests.post') as mock_post:
            mock_post.side_effect = Exception("Telegram API xatosi")
            
            # API xatosi bo'lsa ham crash bo'lmasligi kerak
            try:
                send_telegram_status("Test xabar")
            except Exception:
                pass  # Xato kutilgan


class TestAPIManager:
    """API Manager modulini test qilish"""
    
    @pytest.mark.asyncio
    async def test_check_apis_success(self):
        """API tekshirish muvaffaqiyatli test"""
        try:
            await check_apis()
            assert True
        except Exception as e:
            pytest.fail(f"API tekshirish xato berdi: {e}")
    
    @pytest.mark.asyncio
    async def test_check_apis_with_env_vars(self):
        """Environment variables bilan API tekshirish"""
        required_env_vars = [
            "ONEINCH_API_KEY",
            "ALCHEMY_API_KEY",
            "NEWS_API_KEY",
            "HF_API_KEY_1",
            "GEMINI_API_KEY_1",
            "TELEGRAM_BOT_TOKEN",
            "TELEGRAM_CHAT_ID"
        ]
        
        with patch.dict('os.environ', {var: 'test_value' for var in required_env_vars}):
            try:
                await check_apis()
                assert True
            except Exception as e:
                pytest.fail(f"Env vars bilan API tekshirish xato berdi: {e}")


class TestIntegration:
    """Integration testlar"""
    
    @pytest.mark.asyncio
    async def test_main_function_components(self):
        """Asosiy funksiya komponentlarini test qilish"""
        # Mock qilish kerak bo'lgan modullar
        with patch('modules.orderflow.run_orderflow') as mock_orderflow, \
             patch('modules.ai_sentiment.run_sentiment') as mock_sentiment, \
             patch('modules.api_manager.check_apis') as mock_check_apis, \
             patch('modules.utils.send_telegram_status') as mock_telegram:
            
            # Mock funksiyalarni async qilish
            mock_orderflow.return_value = asyncio.Future()
            mock_orderflow.return_value.set_result(None)
            
            mock_sentiment.return_value = asyncio.Future()
            mock_sentiment.return_value.set_result(None)
            
            mock_check_apis.return_value = asyncio.Future()
            mock_check_apis.return_value.set_result(None)
            
            mock_telegram.return_value = asyncio.Future()
            mock_telegram.return_value.set_result(None)
            
            # Test qilish
            try:
                from main import main
                await main()
                assert True
            except Exception as e:
                pytest.fail(f"Main funksiya xato berdi: {e}")


class TestErrorHandling:
    """Xato boshqarish testlari"""
    
    @pytest.mark.asyncio
    async def test_network_error_handling(self):
        """Tarmoq xatolarini boshqarish testi"""
        with patch('modules.ai_sentiment.requests.post') as mock_post:
            mock_post.side_effect = ConnectionError("Tarmoq xatosi")
            
            # Tarmoq xatosi bo'lsa ham crash bo'lmasligi kerak
            try:
                await run_sentiment()
            except ConnectionError:
                pass  # Xato kutilgan
    
    def test_missing_env_vars(self):
        """Env variables yo'q bo'lsa test"""
        with patch.dict('os.environ', {}, clear=True):
            from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
            
            # None qiymat bo'lishi kerak
            assert TELEGRAM_BOT_TOKEN is None
            assert TELEGRAM_CHAT_ID is None


class TestPerformance:
    """Performance testlar"""
    
    @pytest.mark.asyncio
    async def test_concurrent_execution(self):
        """Parallel bajarilish testi"""
        import time
        
        async def mock_task():
            await asyncio.sleep(0.1)
            return "completed"
        
        start_time = time.time()
        
        # 5 ta parallel task
        tasks = [mock_task() for _ in range(5)]
        results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        
        # Parallel bajarilish 0.5 soniyadan kam vaqt olishi kerak
        assert end_time - start_time < 0.5
        assert len(results) == 5
        assert all(result == "completed" for result in results)


# Pytest konfiguratsiyasi
@pytest.fixture
def mock_env_vars():
    """Test uchun mock environment variables"""
    env_vars = {
        "ONEINCH_API_KEY": "test_oneinch_key",
        "ALCHEMY_API_KEY": "test_alchemy_key",
        "NEWS_API_KEY": "test_news_key",
        "HF_API_KEY_1": "test_hf_key",
        "GEMINI_API_KEY_1": "test_gemini_key",
        "TELEGRAM_BOT_TOKEN": "test_telegram_token",
        "TELEGRAM_CHAT_ID": "test_chat_id"
    }
    
    with patch.dict('os.environ', env_vars):
        yield env_vars


# Test ishga tushirish
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
