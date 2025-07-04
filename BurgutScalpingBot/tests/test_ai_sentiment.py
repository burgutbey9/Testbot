from modules.ai_sentiment import analyze_sentiment

def test_ai_sentiment():
    result = analyze_sentiment("Bitcoin will pump!")
    assert result in ["positive", "negative", "neutral"]
