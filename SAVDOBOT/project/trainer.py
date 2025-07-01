# trainer.py - AI o'z-o'zini o'qitadi va strategiyani optimallashtiradi

import pandas as pd
import json
import logging
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression # Misol uchun oddiy model
from sklearn.metrics import accuracy_score, classification_report
import joblib # Modelni saqlash va yuklash uchun

# Log konfiguratsiyasi
LOGS_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOGS_DIR, exist_ok=True) # logs papkasini yaratish

# Asosiy trainer loglari (results.csv ga yoziladi)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(os.path.join(LOGS_DIR, "results.csv"), encoding='utf-8', mode='a'), # Natijalar CSV ga qo'shib yoziladi
                        logging.StreamHandler()
                    ])

# Xatolar uchun alohida logger
error_logger = logging.getLogger('errors')
error_handler = logging.FileHandler(os.path.join(LOGS_DIR, "errors.log"), encoding='utf-8')
error_handler.setLevel(logging.ERROR)
error_logger.addHandler(error_handler)
error_logger.propagate = False # Asosiy loggerga ikki marta yozmaslik uchun

OPTIMIZED_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "optimized_config.json")
LOCAL_MODEL_PATH = os.path.join(os.path.dirname(__file__), "local_sentiment_model.pkl")

def optimize_strategy(results_csv_path):
    """
    'results.csv' fayli asosida savdo strategiyasini optimallashtiradi.
    Args:
        results_csv_path (str): Oldingi backtest natijalari joylashgan CSV fayl yo'li.
    """
    logging.info(f"Strategiya optimallashtirilmoqda, ma'lumotlar: {results_csv_path}")
    try:
        if not os.path.exists(results_csv_path):
            logging.warning(f"Natijalar fayli topilmadi: {results_csv_path}. Optimallashtirish o'tkazilmadi.")
            return

        df = pd.read_csv(results_csv_path)

        # Optimallashtirish uchun ma'lumotlarni tayyorlash
        # Misol: oldingi savdo signallari va natijalar asosida
        # Bu yerda sizning savdo strategiyangizning parametrlari bo'ladi
        # va ularni optimallashtirish uchun mashinani o'rganish algoritmlari qo'llaniladi.

        # Soddalashtirilgan misol:
        # Agar sizning `results.csv` faylingizda 'type' va 'pnl_percent' ustunlari bo'lsa
        if 'type' in df.columns and 'pnl_percent' in df.columns:
            # Har bir savdoning natijasini (foyda/zarar) belgilash
            # 'actual_outcome' ustunini yaratish (foyda = 1, zarar = 0)
            df['actual_outcome'] = (df['pnl_percent'] > 0).astype(int)

            # Xususiyatlar (features) va maqsad (target)
            # Bu yerda sizning savdo signallaringiz yoki indikatorlaringiz xususiyat bo'lishi mumkin
            # Oddiy misol uchun, savdo turini (BUY/SELL) xususiyat sifatida ishlatamiz
            df['is_buy'] = (df['type'] == 'BUY').astype(int)
            df['is_sell'] = (df['type'] == 'SELL').astype(int)

            # Faqat savdo signallari mavjud bo'lgan qatorlarni tanlash
            # Faqat 'BUY' va 'SELL' savdolari bo'yicha natijalarni ko'rib chiqamiz
            X = df[['is_buy', 'is_sell']].copy()
            y = df['actual_outcome'].copy()

            # NaN qiymatlarni olib tashlash (agar mavjud bo'lsa)
            combined_df = pd.concat([X, y], axis=1).dropna()
            X = combined_df[['is_buy', 'is_sell']]
            y = combined_df['actual_outcome']


            if not X.empty and not y.empty and len(X) > 1: # Kamida 2 ta namuna bo'lishi kerak
                # Ma'lumotlarni o'qitish va test to'plamlariga ajratish
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y) # stratify qo'shildi

                # Oddiy logistik regressiya modelini o'qitish
                model = LogisticRegression(solver='liblinear', random_state=42) # solver qo'shildi
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                accuracy = accuracy_score(y_test, y_pred)
                report = classification_report(y_test, y_pred, zero_division=0) # zero_division qo'shildi

                logging.info(f"Model aniqligi: {accuracy:.2f}")
                logging.info(f"Tasniflash hisoboti:\n{report}")

                # Optimal sozlamalarni saqlash (misol uchun, model parametrlari)
                optimized_config = {
                    "model_accuracy": accuracy,
                    "model_coefficients": model.coef_.tolist(),
                    "model_intercept": model.intercept_.tolist(),
                    "last_optimization_date": str(pd.Timestamp.now())
                }
                with open(OPTIMIZED_CONFIG_PATH, "w", encoding="utf-8") as f:
                    json.dump(optimized_config, f, indent=4)
                logging.info(f"Optimal sozlamalar '{OPTIMIZED_CONFIG_PATH}' fayliga saqlandi.")

                # Modelni saqlash (sentiment modeliga o'xshash)
                joblib.dump(model, LOCAL_MODEL_PATH)
                logging.info(f"Mashinani o'rganish modeli '{LOCAL_MODEL_PATH}' fayliga saqlandi.")
            else:
                logging.warning("Optimallashtirish uchun yetarli ma'lumot topilmadi (X yoki y bo'sh yoki namuna soni kam).")
        else:
            logging.warning("`results.csv` faylida 'type' yoki 'pnl_percent' ustunlari topilmadi. Optimallashtirish o'tkazilmadi.")

    except FileNotFoundError:
        error_logger.error(f"Xato: {results_csv_path} fayli topilmadi. Iltimos, to'g'ri yo'lni kiriting.")
    except Exception as e:
        error_logger.error(f"Strategiyani optimallashtirishda kutilmagan xato yuz berdi: {e}", exc_info=True)

if __name__ == "__main__":
    # Misol uchun foydalanish
    # Bu yerda sizning `project/logs/results.csv` faylingiz bo'lishi kerak.
    # Agar mavjud bo'lmasa, `backtest.py` ni ishga tushirib, ma'lumotlar yaratishingiz mumkin.
    #
    # # Test uchun namunaviy results.csv faylini yaratish (faqat test uchun)
    # sample_results_data = {
    #     'time': pd.to_datetime(['2023-01-01', '2023-01-01', '2023-01-02', '2023-01-02', '2023-01-03']),
    #     'type': ['BUY', 'SELL', 'BUY', 'SELL', 'BUY'],
    #     'price': [100, 101, 102, 100, 103],
    #     'pnl_percent': [0, 1.0, 0, -2.0, 0],
    #     'capital': [10000, 10100, 10100, 9898, 9898]
    # }
    # sample_results_df = pd.DataFrame(sample_results_data)
    # sample_results_df.to_csv(os.path.join(LOGS_DIR, "results.csv"), index=False)

    optimize_strategy(os.path.join(LOGS_DIR, "results.csv"))
