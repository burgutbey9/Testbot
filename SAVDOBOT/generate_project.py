import os

def create_structure():
    folders = ['project', 'project/logs', 'project/data']
    files = [
        'main.py',
        'indicators.py',
        'api_manager.py',
        'risk_manager.py',
        'ai_sentiment.py',
        'orderflow.py',
        'backtest.py',
        'logger.py',
        '.env.example',
        'requirements.txt',
        'README.md',
        'PROJECT_IDEA.txt',
        'logs/trades.log',
        'logs/errors.log',
        'logs/api_rotation.log',
        'logs/results.csv',
        'data/sample_data.csv'
    ]

    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"📁 {folder} papka yaratildi.")

    for file in files:
        path = f"project/{file}"
        with open(path, 'w', encoding='utf-8') as f:
            f.write('')
        print(f"✅ {file} fayl yaratildi.")

if __name__ == "__main__":
    create_structure()
