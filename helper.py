import pandas as pd
from urlextract import URLExtract
from wordcloud import WordCloud
from collections import Counter
import emoji
import Detection_Function

# from nltk.sentiment.vader import SentimentIntensityAnalyzer
# import nltk

# nltk.downloader.download('vader_lexicon')

# sentiments = SentimentIntensityAnalyzer()

extract = URLExtract()


def fetch_stats(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    num_message = df.shape[0]

    words = []
    for message in df['message']:
        words.extend(message.split())

    media_count = df[df['message'] == '<Media omitted>'].shape[0]
    delecetd_message_count = df[df['message'] == 'This message was deleted'].shape[0]

    urls = []
    for message in df['message']:
        urls.extend(extract.find_urls(message))

    return num_message, len(words), len(urls), media_count, delecetd_message_count


def most_busy_user(df):
    x = df['user'].value_counts().head()
    dfx = round((df['user'].value_counts() / df.shape[0]) * 100, 2).reset_index().rename(
        columns={'index': 'name', 'user': 'percentage'})
    return x, dfx


def create_word_cloud(selected_user, df):
    f = open('stop_hinglish.txt', 'r')
    stop_words = f.read()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    # temp = temp[temp['message'] != '<media omitted>\n']
    temp = temp[temp['message'] != '<Media omitted>\n']
    temp = temp[temp['message'] != 'This message was deteted\n']

    def remove_stop_words(message):
        y = []
        for word in message.lower().split():
            if word not in stop_words:
                y.append(word)
        return " ".join(y)

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    wc = WordCloud(width=500, height=500, min_font_size=5, background_color="white")
    temp['message'] = temp['message'].apply(remove_stop_words)
    df_wc = wc.generate(df['message'].str.cat(sep=" "))
    return df_wc


def most_common_words(selected_user, df):
    
    f = open('stop_hinglish.txt', 'r')
    stop_words = f.read()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    # temp = temp[temp['message'] != '<media omitted>\n']
    temp = temp[temp['message'] != '<Media omitted>\n']
    temp = temp[temp['message'] != 'This message was deteted\n']

    words = []

    for message in temp['message']:
        for word in message.lower().split():
            if word not in stop_words:
                words.append(word)

    word_df = pd.DataFrame(Counter(words).most_common(20))

    return word_df


def emoji_analysis(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    emojis = []
    for message in df['message']:
        x = emoji.distinct_emoji_list(message)
        emojis.extend([c for c in x])

    emoji_df = pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))))

    return emoji_df


def monthly_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()

    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i] + "-" + str(timeline['year'][i]))

    timeline['time'] = time

    return timeline


def daily_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    dailytimeline = df.groupby('only_date').count()['message'].reset_index()

    return dailytimeline


def week_activity_map(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['day_name'].value_counts()


def month_activity_map(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['month'].value_counts()


def activity_heatmap(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    user_heatmap = df.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)

    return user_heatmap


'''
def sentiment_analyse(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<media omitted>\n']
    temp = temp[temp['message'] != '<Media omitted>\n']

    sen_df = pd.DataFrame(df, columns=['date', 'user', 'message'])
    sen_df['positive'] = [sentiments.polarity_scores(i)['pos'] for i in sen_df['message']]
    sen_df['negative'] = [sentiments.polarity_scores(i)['neg'] for i in sen_df['message']]
    sen_df['neutral'] = [sentiments.polarity_scores(i)['neu'] for i in sen_df['message']]

    return sen_df

'''


def message_language_count(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    df['language'] = df['message'].apply(lambda x: Detection_Function.Detect_The_lang(x))

    temp = df[df['user'] != 'group_notification']
    # temp = temp[temp['message'] != '<media omitted>\n']
    temp = temp[temp['message'] != '<Media omitted>\n']
    temp = temp[temp['message'] != 'This message was deteted\n']
    temp['message'] = temp['message'].str.replace('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 'This is an url\n')
    temp = temp[temp['message'] != 'This is an url\n']

    df_eng = temp[temp['language']=='English']
    df_non_eng = temp[temp['language'] != 'English']

    eng_count = df_eng.shape[0]
    non_eng_count = df_non_eng.shape[0]

    return df_eng, df_non_eng, eng_count, non_eng_count



