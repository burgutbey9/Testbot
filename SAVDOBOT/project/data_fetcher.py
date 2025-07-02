# project/data_fetcher.py - Turli API'lardan ma'lumotlarni olish funksiyalari

import requests
import json
import logging
import os
import asyncio
from web3 import Web3 # Alchemy API bilan ishlash uchun

# Log konfiguratsiyasi (api_rotation.log ga yoziladi)
LOGS_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOGS_DIR, exist_ok=True) # logs papkasini yaratish

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(os.path.join(LOGS_DIR, "api_rotation.log"), encoding='utf-8', mode='a'),
                        logging.StreamHandler()
                    ])

# Xatolar uchun alohida logger
error_logger = logging.getLogger('errors')
error_handler = logging.FileHandler(os.path.join(LOGS_DIR, "errors.log"), encoding='utf-8')
error_handler.setLevel(logging.ERROR)
error_logger.addHandler(error_handler)
error_logger.propagate = False # Asosiy loggerga ikki marta yozmaslik uchun

# API kalitlarini olish uchun api_manager modulini import qilish
from project.api_manager import get_one_inch_api_key, get_alchemy_api_key, get_news_api_key, get_reddit_api_key

# --- 1inch API konfiguratsiyasi ---
ONE_INCH_API_BASE_URL = "https://api.1inch.dev/swap/v5.2" # Ethereum mainnet uchun

async def fetch_1inch_quote(api_key: str, from_token_address: str, to_token_address: str, amount: int, chain_id: int) -> dict | None:
    """
    1inch API orqali token almashtirish uchun narx (quote) ma'lumotlarini oladi.
    Args:
        api_key (str): 1inch API kaliti.
        from_token_address (str): Sotiladigan token manzili.
        to_token_address (str): Sotib olinadigan token manzili.
        amount (int): Sotiladigan token miqdori (tokenning minimal birligida, masalan, wei).
        chain_id (int): Blokcheyn zanjiri ID (masalan, 1 - Ethereum, 56 - BSC).
    Returns:
        dict: Narx ma'lumotlari yoki None agar xato yuz bersa.
    """
    endpoint = f"{ONE_INCH_API_BASE_URL}/{chain_id}/quote"
    headers = {"Authorization": f"Bearer {api_key}"}
    params = {
        "fromTokenAddress": from_token_address,
        "toTokenAddress": to_token_address,
        "amount": str(amount),
    }
    try:
        response = requests.get(endpoint, headers=headers, params=params, timeout=10)
        response.raise_for_status() # HTTP xatolarini tekshirish
        data = response.json()
        logging.info(f"1inch quote olindi: {from_token_address} -> {to_token_address} (ChainID: {chain_id})")
        return data
    except requests.exceptions.RequestException as e:
        error_logger.error(f"1inch quote olishda xato: {e}", exc_info=True)
        return None
    except Exception as e:
        error_logger.error(f"1inch quote olishda kutilmagan xato: {e}", exc_info=True)
        return None

# --- Alchemy API konfiguratsiyasi ---
# Alchemy URL'lari main.py da berilgan, bu yerda ularni ishlatamiz.
ALCHEMY_URLS = {
    1: "https://eth-mainnet.g.alchemy.com/v2/", # Ethereum
    56: "https://bnb-mainnet.g.alchemy.com/v2/", # BNB Chain
    42161: "https://arbnova-mainnet.g.alchemy.com/v2/", # Arbitrum Nova
    137: "https://polygon-mainnet.g.alchemy.com/v2/" # Polygon
}

async def fetch_alchemy_gas_price(api_key: str, chain_id: int) -> float | None:
    """
    Alchemy API orqali joriy gaz narxini (Gwei) oladi.
    Args:
        api_key (str): Alchemy API kaliti.
        chain_id (int): Blokcheyn zanjiri ID.
    Returns:
        float: Joriy gaz narxi Gwei'da yoki None.
    """
    if chain_id not in ALCHEMY_URLS:
        error_logger.error(f"Alchemy uchun {chain_id} zanjir ID'si qo'llab-quvvatlanmaydi.")
        return None

    api_url = f"{ALCHEMY_URLS[chain_id]}{api_key}"
    w3 = Web3(Web3.HTTPProvider(api_url))

    try:
        if not w3.is_connected():
            error_logger.error(f"Alchemy ({chain_id}) ga ulanib bo'lmadi.")
            return None
        
        gas_price_wei = w3.eth.gas_price
        gas_price_gwei = w3.from_wei(gas_price_wei, 'gwei')
        logging.info(f"Alchemy orqali gaz narxi olindi ({chain_id}): {gas_price_gwei} Gwei")
        return float(gas_price_gwei)
    except Exception as e:
        error_logger.error(f"Alchemy orqali gaz narxini olishda xato: {e}", exc_info=True)
        return None

# --- NewsAPI.org konfiguratsiyasi ---
NEWS_API_BASE_URL = "https://newsapi.org/v2/everything"

async def fetch_news_articles(api_key: str, query: str, language: str = "en", page_size: int = 10) -> list[dict] | None:
    """
    NewsAPI.org orqali yangilik maqolalarini oladi.
    Args:
        api_key (str): NewsAPI.org API kaliti.
        query (str): Qidiruv so'rovi (masalan, "cryptocurrency").
        language (str): Til (masalan, "en" - inglizcha).
        page_size (int): Qaytariladigan maqolalar soni.
    Returns:
        list[dict]: Yangilik maqolalari ro'yxati yoki None.
    """
    params = {
        "q": query,
        "language": language,
        "pageSize": page_size,
        "apiKey": api_key
    }
    try:
        response = requests.get(NEWS_API_BASE_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data.get("status") == "ok" and data.get("articles"):
            logging.info(f"NewsAPI.org orqali {len(data['articles'])} ta yangilik olindi.")
            return data["articles"]
        logging.warning(f"NewsAPI.org orqali yangiliklarni olishda muammo: {data.get('message', 'Noma’lum xato')}")
        return None
    except requests.exceptions.RequestException as e:
        error_logger.error(f"NewsAPI.org dan yangiliklarni olishda xato: {e}", exc_info=True)
        return None
    except Exception as e:
        error_logger.error(f"NewsAPI.org dan yangiliklarni olishda kutilmagan xato: {e}", exc_info=True)
        return None

# --- Reddit API konfiguratsiyasi ---
REDDIT_AUTH_URL = "https://www.reddit.com/api/v1/access_token"
REDDIT_API_BASE_URL = "https://oauth.reddit.com"

async def fetch_reddit_posts(reddit_config: dict, subreddit: str, limit: int = 5) -> list[dict] | None:
    """
    Reddit API orqali subreddit'dan postlarni oladi.
    Args:
        reddit_config (dict): Reddit API konfiguratsiyasi (client_id, client_secret, username, password).
        subreddit (str): Subreddit nomi (masalan, "cryptocurrency").
        limit (int): Qaytariladigan postlar soni.
    Returns:
        list[dict]: Reddit postlari ro'yxati yoki None.
    """
    client_id = reddit_config.get("client_id")
    client_secret = reddit_config.get("client_secret")
    username = reddit_config.get("username")
    password = reddit_config.get("password")

    if not all([client_id, client_secret, username, password]):
        error_logger.error("Reddit API konfiguratsiyasi to'liq emas.")
        return None

    # Token olish
    auth = requests.auth.HTTPBasicAuth(client_id, client_secret)
    data = {
        'grant_type': 'password',
        'username': username,
        'password': password
    }
    headers = {'User-Agent': 'ScalpingBot/1.0'} # Unique User-Agent
    try:
        response = requests.post(REDDIT_AUTH_URL, auth=auth, data=data, headers=headers, timeout=10)
        response.raise_for_status()
        token = response.json().get('access_token')
        if not token:
            error_logger.error("Reddit access token olib bo'lmadi.")
            return None
        
        headers['Authorization'] = f'bearer {token}'
        
        # Postlarni olish
        post_endpoint = f"{REDDIT_API_BASE_URL}/r/{subreddit}/hot"
        params = {"limit": limit}
        post_response = requests.get(post_endpoint, headers=headers, params=params, timeout=10)
        post_response.raise_for_status()
        posts_data = post_response.json()
        
        posts = []
        if posts_data and posts_data.get('data') and posts_data['data'].get('children'):
            for child in posts_data['data'].get('children'):
                posts.append(child['data'])
            logging.info(f"Reddit orqali {len(posts)} ta post olindi.")
            return posts
        return None
    except requests.exceptions.RequestException as e:
        error_logger.error(f"Reddit API dan postlarni olishda xato: {e}", exc_info=True)
        return None
    except Exception as e:
        error_logger.error(f"Reddit API dan postlarni olishda kutilmagan xato: {e}", exc_info=True)
        return None

# --- The Graph (Conceptual) ---
# The Graph API kaliti talab qilmaydi, lekin subgraph URL kerak bo'ladi.
# Misol uchun, Uniswap V3 subgraph URL'i:
UNISWAP_V3_SUBGRAPH_URL = "https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3"

async def fetch_the_graph_data(query: str) -> dict | None:
    """
    The Graph orqali ma'lumotlarni oladi (GraphQL so'rovlari).
    Args:
        query (str): GraphQL so'rovi.
    Returns:
        dict: GraphQL javobi yoki None.
    """
    headers = {"Content-Type": "application/json"}
    payload = {"query": query}
    try:
        response = requests.post(UNISWAP_V3_SUBGRAPH_URL, headers=headers, data=json.dumps(payload), timeout=10)
        response.raise_for_status()
        data = response.json()
        if data.get("data"):
            logging.info("The Graph orqali ma'lumotlar olindi.")
            return data["data"]
        logging.warning(f"The Graph orqali ma'lumot olishda muammo: {data.get('errors', 'Noma’lum xato')}")
        return None
    except requests.exceptions.RequestException as e:
        error_logger.error(f"The Graph dan ma'lumot olishda xato: {e}", exc_info=True)
        return None
    except Exception as e:
        error_logger.error(f"The Graph dan ma'lumot olishda kutilmagan xato: {e}", exc_info=True)
        return None


if __name__ == "__main__":
    # Test uchun .env faylidan API kalitlarini yuklash
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

    async def test_fetches():
        # 1inch test
        one_inch_key = get_one_inch_api_key()
        if one_inch_key:
            print("\n--- 1inch API testi ---")
            quote = await fetch_1inch_quote(one_inch_key, "0xEeeeeEeeeEeEeeEeEeEeeEEEeeeeEeeeeeeeEEeE", "0xdAC17F958D2ee523a2206206994597C13D831ec7", 10**18, 1)
            print(f"1inch Quote: {quote}")

        # Alchemy test
        alchemy_key = get_alchemy_api_key()
        if alchemy_key:
            print("\n--- Alchemy API testi ---")
            gas_price = await fetch_alchemy_gas_price(alchemy_key, 1)
            print(f"Alchemy Gas Price: {gas_price}")

        # NewsAPI test
        news_key = get_news_api_key()
        if news_key:
            print("\n--- NewsAPI testi ---")
            news = await fetch_news_articles(news_key, "bitcoin")
            print(f"NewsAPI Articles (first 2): {news[:2] if news else 'None'}")

        # Reddit test
        reddit_config = get_reddit_api_key()
        if reddit_config:
            print("\n--- Reddit API testi ---")
            reddit_posts = await fetch_reddit_posts(reddit_config, "cryptocurrency")
            print(f"Reddit Posts (first 2): {reddit_posts[:2] if reddit_posts else 'None'}")
        
        # The Graph test (conceptual)
        print("\n--- The Graph testi (konseptual) ---")
        # Misol GraphQL so'rovi (real Uniswap V3 subgraph uchun)
        graphql_query = """
        {
          bundles(first: 1, where: {id: "1"}) {
            ethPriceUSD
          }
        }
        """
        graph_data = await fetch_the_graph_data(graphql_query)
        print(f"The Graph Data: {graph_data}")

    asyncio.run(test_fetches())
