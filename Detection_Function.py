import joblib
from sklearn.pipeline import Pipeline


language_dec = joblib.load(open("Language_detection_model", "rb"))
sentiment_dec = joblib.load(open("Sentiment_detection_model", 'rb'))


def Detect_The_lang(text):
    text = [text]
    result = language_dec.predict(text)[0]
    return result


def Detect_The_senti(text):
    text = [text]
    result = sentiment_dec.predict(text)[0]
    return result


#print(Detect_The_senti("The man is green"))
#ex =  "أنا أحب زكا وفريقها الرائع"
#print(Detect_The_lang(ex))

'''
        # Langugae Detection Part
        st.title("Language Detection of WhatsApp Chat: ")
        eng, neng, c_eng, c_ncng = helper.message_language_count(selected_user, df)
        com, con = st.columns(2)

        with com:
            st.header("English message Count")
            st.title(num_messages)

        with con:
            st.header("Non English message Count")
            st.title(no_words)
    '''
