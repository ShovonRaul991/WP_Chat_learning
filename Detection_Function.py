import joblib
from sklearn.pipeline import Pipeline


language_dec = joblib.load(open("language_detection_model", "rb"))
sentiment_dec = joblib.load(open("Sentiment_detection_model", 'rb'))


def Detect_The_lang(text):
    text = [text]
    result = language_dec.predict(text)[0]
    return result


def Detect_The_senti(text):
    text = [text]
    result = sentiment_dec.predict(text)[0]
    return result


print(Detect_The_senti("The man is green"))
ex =  "أنا أحب زكا وفريقها الرائع"
print(Detect_The_lang(ex))