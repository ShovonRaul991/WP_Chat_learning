import matplotlib.pyplot as plt
# import pandas as pd
import seaborn as sns
import streamlit as st

import helper
import preprocessor
import Detection_Function

st.sidebar.title("WhatsApp Chat Analyzer and Prediction Application")

# accu = Language_detection.model_creation()
# st.title("Accuracy of Language Detection model is"+str(accu))

uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")
    # st.text(data) to view the inputted data
    df = preprocessor.pre_process(data)

    st.title("The created DataFrame is: ")
    st.dataframe(df)

    # fetch unique user

    user_list = df['user'].unique().tolist()
    user_list.remove('group_notification')
    user_list.sort()
    user_list.insert(0, "Overall")

    selected_user = st.sidebar.selectbox("Show analysis with respect to: ", user_list)

    if st.sidebar.button("Show Analysis"):

        num_messages, no_words, no_urls, media_count, deleted_count = helper.fetch_stats(selected_user, df)
        col1, col2, col3 = st.columns(3)

        with col1:
            st.header("Total Messages")
            st.title(num_messages)

        with col2:
            st.header("Total Words")
            st.title(no_words)

        with col3:
            st.header("Total URLs")
            st.title(no_urls)
        # not working

        # monthly timeline
        st.title("Monthly Timeline")
        timeline = helper.monthly_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(timeline['time'], timeline['message'], color='green')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # daily timeline
        st.title("Daily Timeline")
        daily_timeline = helper.daily_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='black')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # activity map
        st.title('Activity Map')
        col1, col2 = st.columns(2)

        with col1:
            st.header("Most busy day")
            busy_day = helper.week_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values, color='purple')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        with col2:
            st.header("Most busy month")
            busy_month = helper.month_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values, color='orange')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        st.title("Weekly Activity Map")
        user_heatmap = helper.activity_heatmap(selected_user, df)
        fig, ax = plt.subplots()
        ax = sns.heatmap(user_heatmap)
        st.pyplot(fig)

        # group level
        if selected_user == "Overall":
            st.title("Most Busy User: ")
            x, y = helper.most_busy_user(df)
            fig, ax = plt.subplots()

            cola, colb = st.columns(2)

            with cola:
                st.dataframe(y)

            with colb:
                ax.bar(x.index, x.values, color='Red')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)

        # word_cloud
        st.title("WordCloud")
        wordcloud_image = helper.create_word_cloud(selected_user, df)
        fig, ax = plt.subplots()
        ax.imshow(wordcloud_image)
        st.pyplot(fig)

        # most_common_words
        st.title("Most Frequent Words: ")
        most_common_word = helper.most_common_words(selected_user, df)
        fig, ax = plt.subplots()
        ax.barh(most_common_word[0], most_common_word[1])
        plt.xticks(rotation='vertical')
        st.pyplot(fig)
        # st.dataframe(most_common_word)

        # emoji analysis
        st.title("Most Common Emojis: ")
        emoji_df = helper.emoji_analysis(selected_user, df)
        fig, ax = plt.subplots()
        ax.pie(emoji_df[1], labels=emoji_df[0])
        #st.pyplot(fig)
        st.dataframe(emoji_df)

    