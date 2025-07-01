"""
AI_SENTIMENT.PY
-----------------
- Matn tozalash (URL, emoji, reklama so‘zlari)
- Duplicate filtr
- Sentiment tahlil (TextBlob)
"""

import re
import emoji
import hashlib
from textblob import TextBlob

class TextCleaner:
    def __init__(self):
        self.spam_keywords = [
            "airdrop", "giveaway", "join now", "free crypto", "click here",
            "bonus", "referral", "invite", "promo", "pump", "dump", "moon",
            "telegram", "whatsapp", "discord"
        ]
        self.seen_hashes = set()

    def clean_text(self, text: str) -> str:
        # URL olib tashlash
        text = re.sub(r"http\S+|www.\S+", "", text)
        # Emoji olib tashlash
        text = emoji.replace_emoji(text, replace='')
        # Belgilar olib tashlash
        text = re.sub(r'[^\w\s]', '', text)
        return text.lower()

    def is_spam(self, text: str) -> bool:
        for word in self.spam_keywords:
            if word in text.lower():
                return True
        return False

    def is_duplicate(self, text: str) -> bool:
        hash_digest = hashlib.md5(text.encode('utf-8')).hexdigest()
        if hash_digest in self.seen_hashes:
            return True
        self.seen_hashes.add(hash_digest)
        return False

class SentimentAnalyzer:
    def __init__(self):
        self.cleaner = TextCleaner()

    def analyze_sentiment(self, text_list: list) -> dict:
        sentiments = []
        for text in text_list:
            clean = self.cleaner.clean_text(text)
            if self.cleaner.is_spam(clean) or self.cleaner.is_duplicate(clean):
                continue
            blob = TextBlob(clean)
            polarity = blob.sentiment.polarity
            if polarity > 0.1:
                sentiment = 'positive'
            elif polarity < -0.1:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
            sentiments.append(sentiment)
        if not sentiments:
            return {'overall': 'neutral'}
        # Overall sentiment
        return {
            'positive': sentiments.count('positive'),
            'negative': sentiments.count('negative'),
            'neutral': sentiments.count('neutral'),
            'overall': max(set(sentiments), key=sentiments.count)
        }

if __name__ == "__main__":
    cleaner = TextCleaner()
    analyzer = SentimentAnalyzer()
    test_texts = [
        "BTC is going to the moon! 🚀🚀",
        "Join our airdrop and get free crypto!",
        "Bitcoin price drops 10% overnight."
    ]
    print(analyzer.analyze_sentiment(test_texts))
