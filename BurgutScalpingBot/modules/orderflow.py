import asyncio
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from web3 import Web3
from web3.exceptions import Web3Exception
import json
import aiohttp
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class SwapEvent:
    """Swap hodisasi ma'lumotlari"""
    timestamp: datetime
    tx_hash: str
    dex: str
    token_in: str
    token_out: str
    amount_in: float
    amount_out: float
    price: float
    gas_price: float
    block_number: int

@dataclass
class OrderFlowMetrics:
    """Order Flow metrikalari"""
    total_volume: float
    buy_volume: float
    sell_volume: float
    buy_sell_ratio: float
    avg_trade_size: float
    large_trades_count: int
    unique_wallets: int
    price_impact: float
    liquidity_score: float

class OrderFlowManager:
    """Order Flow menejerini"""
    
    def __init__(self, config):
        self.config = config
        self.web3_connections = {}
        self.event_filters = {}
        self.recent_swaps = []
        self.metrics_cache = {}
        self.last_block_processed = {}
        
        # DEX contract addresslari
        self.dex_contracts = {
            'uniswap_v2': '0x5C69bEe701ef814a2B6a3EDD4B1652CB9cc5aA6f',
            'uniswap_v3': '0x1F98431c8aD98523631AE4a59f267346ea31F984',
            'sushiswap': '0xC0AEe478e3658e2610c5F7A4A2E1777cE9e4f2Ac',
            'pancakeswap': '0xcA143Ce32Fe78f1f7019d7d551a6402fC5350c73'
        }
        
        # Token contractlari
        self.token_contracts = {
            'WETH': '0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2',
            'USDC': '0xA0b86a33E6417c32b32cE63B5c1A9e3a8A26f5A1',
            'USDT': '0xdAC17F958D2ee523a2206206994597C13D831ec7',
            'DAI': '0x6B175474E89094C44Da98b954EedeAC495271d0F'
        }
    
    async def initialize(self):
        """Komponentni boshlash"""
        try:
            logger.info("🔄 OrderFlow Manager boshlanyapti...")
            
            # Web3 bog'lanishlarini sozlash
            await self.setup_web3_connections()
            
            # Event filterlarini sozlash
            await self.setup_event_filters()
            
            # Oxirgi blocklarni olish
            await self.load_last_processed_blocks()
            
            logger.info("✅ OrderFlow Manager muvaffaqiyatli boshlandi")
            
        except Exception as e:
            logger.error(f"❌ OrderFlow Manager boshlashda xato: {str(e)}")
            raise
    
    async def setup_web3_connections(self):
        """Web3 bog'lanishlarini sozlash"""
        networks = {
            'ethereum': self.config.api.alchemy_eth,
            'bsc': self.config.api.alchemy_bnb,
            'arbitrum': self.config.api.alchemy_arb,
            'polygon': self.config.api.alchemy_polygon
        }
        
        for network, rpc_url in networks.items():
            try:
                web3 = Web3(Web3.HTTPProvider(rpc_url))
                if web3.is_connected():
                    self.web3_connections[network] = web3
                    logger.info(f"✅ {network} ga bog'landi")
                else:
                    logger.error(f"❌ {network} ga bog'lanib bo'lmadi")
            except Exception as e:
                logger.error(f"❌ {network} xatosi: {str(e)}")
    
    async def setup_event_filters(self):
        """Event filterlarini sozlash"""
        # Swap event signature
        swap_event_sig = Web3.keccak(text="Swap(address,uint256,uint256,uint256,uint256,address)")
        
        for network, web3 in self.web3_connections.items():
            try:
                # Uniswap V2 Swap events
                filter_params = {
                    'fromBlock': 'latest',
                    'topics': [swap_event_sig.hex()]
                }
                
                event_filter = web3.eth.filter(filter_params)
                self.event_filters[network] = event_filter
                logger.info(f"✅ {network} uchun event filter sozlandi")
                
            except Exception as e:
                logger.error(f"❌ {network} event filter xatosi: {str(e)}")
    
    async def load_last_processed_blocks(self):
        """Oxirgi qayta ishlangan bloklarni yuklash"""
        for network, web3 in self.web3_connections.items():
            try:
                latest_block = web3.eth.block_number
                self.last_block_processed[network] = latest_block - 10  # 10 blok orqada boshlash
                logger.info(f"📊 {network}: {latest_block} blokdan boshlanadi")
            except Exception as e:
                logger.error(f"❌ {network} blok raqamini olishda xato: {str(e)}")
    
    async def analyze(self) -> Dict[str, Any]:
        """Asosiy tahlil funksiyasi"""
        try:
            # Yangi eventlarni olish
            new_events = await self.fetch_new_events()
            
            # Eventlarni qayta ishlash
            processed_swaps = await self.process_events(new_events)
            
            # Metrikalari hisoblash
            metrics = await self.calculate_metrics(processed_swaps)
            
            # Signallarni generatsiya qilish
            signals = await self.generate_signals(metrics)
            
            return {
                'timestamp': datetime.now().isoformat(),
                'new_swaps_count': len(processed_swaps),
                'metrics': asdict(metrics),
                'signals': signals,
                'health': 'OK'
            }
            
        except Exception as e:
            logger.error(f"❌ OrderFlow tahlilida xato: {str(e)}")
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'health': 'ERROR'
            }
    
    async def fetch_new_events(self) -> List[Dict]:
        """Yangi eventlarni olish"""
        all_events = []
        
        for network, web3 in self.web3_connections.items():
            try:
                # Yangi bloklarni tekshirish
                current_block = web3.eth.block_number
                last_processed = self.last_block_processed.get(network, current_block - 10)
                
                if current_block > last_processed:
                    # Swap eventlarini olish
                    events = await self.get_swap_events(web3, last_processed + 1, current_block)
                    all_events.extend(events)
                    
                    # Oxirgi qayta ishlangan blokni yangilash
                    self.last_block_processed[network] = current_block
                    
                    logger.info(f"📊 {network}: {len(events)} yangi event topildi")
                
            except Exception as e:
                logger.error(f"❌ {network} eventlarini olishda xato: {str(e)}")
        
        return all_events
    
    async def get_swap_events(self, web3: Web3, from_block: int, to_block: int) -> List[Dict]:
        """Swap eventlarini olish"""
        try:
            # Swap event signature
            swap_signature = Web3.keccak(text="Swap(address,uint256,uint256,uint256,uint256,address)")
            
            # Event filter
            filter_params = {
                'fromBlock': from_block,
                'toBlock': to_block,
                'topics': [swap_signature.hex()]
            }
            
            # Eventlarni olish
            events = web3.eth.get_logs(filter_params)
            
            return [dict(event) for event in events]
            
        except Exception as e:
            logger.error(f"❌ Swap eventlarini olishda xato: {str(e)}")
            return []
    
    async def process_events(self, events: List[Dict]) -> List[SwapEvent]:
        """Eventlarni qayta ishlash"""
        processed_swaps = []
        
        for event in events:
            try:
                # Event ma'lumotlarini dekodlash
                swap_data = await self.decode_swap_event(event)
                if swap_data:
                    swap_event = SwapEvent(**swap_data)
                    processed_swaps.append(swap_event)
                    
            except Exception as e:
                logger.error(f"❌ Event qayta ishlashda xato: {str(e)}")
        
        # Recent swaps listini yangilash
        self.recent_swaps.extend(processed_swaps)
        
        # Eski ma'lumotlarni tozalash (oxirgi 1 soat)
        cutoff_time = datetime.now() - timedelta(hours=1)
        self.recent_swaps = [
            swap for swap in self.recent_swaps 
            if swap.timestamp > cutoff_time
        ]
        
        return processed_swaps
    
    async def decode_swap_event(self, event: Dict) -> Optional[Dict]:
        """Swap eventini dekodlash"""
        try:
            # Bu yerda real event decoding logikasi bo'lishi kerak
            # Hozircha mock data qaytarish
            return {
                'timestamp': datetime.now(),
                'tx_hash': event.get('transactionHash', '').hex(),
                'dex': 'uniswap_v2',
                'token_in': 'WETH',
                'token_out': 'USDC',
                'amount_in': 1.0,
                'amount_out': 2000.0,
                'price': 2000.0,
                'gas_price': 20.0,
                'block_number': event.get('blockNumber', 0)
            }
        except Exception as e:
            logger.error(f"❌ Event dekodlashda xato: {str(e)}")
            return None
    
    async def calculate_metrics(self, swaps: List[SwapEvent]) -> OrderFlowMetrics:
        """Metrikalari hisoblash"""
        if not swaps:
            return OrderFlowMetrics(
                total_volume=0, buy_volume=0, sell_volume=0,
                buy_sell_ratio=0, avg_trade_size=0, large_trades_count=0,
                unique_wallets=0, price_impact=0, liquidity_score=0
            )
        
        # Asosiy metrikalari hisoblash
        total_volume = sum(swap.amount_in for swap in swaps)
        buy_volume = sum(swap.amount_in for swap in swaps if swap.token_out == 'USDC')
        sell_volume = total_volume - buy_volume
        
        buy_sell_ratio = buy_volume / sell_volume if sell_volume > 0 else 0
        avg_trade_size = total_volume / len(swaps)
        
        # Katta savdolar (o'rtacha hajmdan 5 marta katta)
        large_trades_count = len([s for s in swaps if s.amount_in > avg_trade_size * 5])
        
        # Unique wallets (hozircha mock)
        unique_wallets = len(set(swap.tx_hash[:10] for swap in swaps))
        
        # Price impact va liquidity score (mock)
        price_impact = sum(abs(swap.price - 2000) / 2000 for swap in swaps) / len(swaps)
        liquidity_score = max(0, 100 - price_impact * 1000)
        
        return OrderFlowMetrics(
            total_volume=total_volume,
            buy_volume=buy_volume,
            sell_volume=sell_volume,
            buy_sell_ratio=buy_sell_ratio,
            avg_trade_size=avg_trade_size,
            large_trades_count=large_trades_count,
            unique_wallets=unique_wallets,
            price_impact=price_impact,
            liquidity_score=liquidity_score
        )
    
    async def generate_signals(self, metrics: OrderFlowMetrics) -> Dict[str, Any]:
        """Signallarni generatsiya qilish"""
        signals = {
            'trend': 'NEUTRAL',
            'strength': 0.5,
            'recommendation': 'HOLD',
            'confidence': 0.5,
            'reasons': []
        }
        
        try:
            # Buy/Sell ratio tahlili
            if metrics.buy_sell_ratio > 1.5:
                signals['trend'] = 'BULLISH'
                signals['strength'] = min(0.8, metrics.buy_sell_ratio / 2)
                signals['recommendation'] = 'BUY'
                signals['reasons'].append(f'Kuchli buy pressure: {metrics.buy_sell_ratio:.2f}')
            elif metrics.buy_sell_ratio < 0.5:
                signals['trend'] = 'BEARISH'
                signals['strength'] = min(0.8, 1 / metrics.buy_sell_ratio / 2)
                signals['recommendation'] = 'SELL'
                signals['reasons'].append(f'Kuchli sell pressure: {metrics.buy_sell_ratio:.2f}')
            
            # Katta savdolar tahlili
            if metrics.large_trades_count > 5:
                signals['strength'] = min(1.0, signals['strength'] + 0.2)
                signals['reasons'].append(f'Katta savdolar: {metrics.large_trades_count}')
            
            # Likvidlik tahlili
            if metrics.liquidity_score < 50:
                signals['strength'] = max(0.3, signals['strength'] - 0.2)
                signals['reasons'].append(f'Past likvidlik: {metrics.liquidity_score:.1f}')
            
            # Confidence hisoblash
            signals['confidence'] = (
                min(1.0, metrics.total_volume / 1000) * 0.3 +
                min(1.0, metrics.unique_wallets / 100) * 0.3 +
                min(1.0, metrics.liquidity_score / 100) * 0.4
            )
            
        except Exception as e:
            logger.error(f"❌ Signal generatsiyasida xato: {str(e)}")
        
        return signals
    
    async def health_check(self):
        """Sog'lik tekshiruvi"""
        healthy_connections = 0
        
        for network, web3 in self.web3_connections.items():
            try:
                if web3.is_connected():
                    healthy_connections += 1
                else:
                    logger.warning(f"⚠️ {network} bog'lanishi yo'q")
            except Exception as e:
                logger.error(f"❌ {network} sog'lik tekshiruvida xato: {str(e)}")
        
        if healthy_connections == 0:
            raise Exception("Hech qanday tarmoqqa bog'lanish yo'q")
        
        logger.info(f"✅ OrderFlow: {healthy_connections}/{len(self.web3_connections)} tarmoq sog'lom")
    
    async def shutdown(self):
        """Komponentni to'xtatish"""
        logger.info("🛑 OrderFlow Manager to'xtatilmoqda...")
        
        # Event filterlarini tozalash
        for network, event_filter in self.event_filters.items():
            try:
                if hasattr(event_filter, 'uninstall'):
                    event_filter.uninstall()
                logger.info(f"✅ {network} event filter to'xtatildi")
            except Exception as e:
                logger.error(f"❌ {network} event filter to'xtatishda xato: {str(e)}")
        
        # Web3 bog'lanishlarini yopish
        self.web3_connections.clear()
        logger.info("✅ OrderFlow Manager to'xtatildi")
