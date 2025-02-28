import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import asyncio
from model_implementation import predict

if not hasattr(st.session_state, 'loop'):
    st.session_state.loop = asyncio.new_event_loop()
    asyncio.set_event_loop(st.session_state.loop)


async def analyze_sentiment(text):
    return await predict(text)


async def process_dataframe(df, text_column, sentiment_col):
    tasks = [analyze_sentiment(text) for text in df[text_column]]
    sentiments = await asyncio.gather(*tasks)
    df[sentiment_col] = sentiments
    return df


st.title("Manglish Sentiment Analysis")

if "df" not in st.session_state:
    st.session_state.df = None

if "uploaded_text" not in st.session_state:
    st.session_state.uploaded_text = None

if "text_input" not in st.session_state:
    st.session_state.text_input = ""

if "selected_option" not in st.session_state:
    st.session_state.selected_option = None

option = st.radio(
    "Select input method:", ("Upload CSV", "Upload Text File", "Enter Text")
)

if (
    "selected_option" not in st.session_state
    or option != st.session_state.selected_option
):
    st.session_state.df = None
    st.session_state.uploaded_text = None
    st.session_state.text_input = ""
    st.session_state.selected_option = option

if option == "Upload CSV":
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file is not None:
        st.session_state.df = pd.read_csv(uploaded_file)
        st.session_state.sentiment_processed = False

    if st.session_state.df is not None:
        df = st.session_state.df
        st.subheader("Raw Data")
        st.dataframe(df)

        df_string_columns = [col for col in df.columns if df[col].dtype == "object"]
        text_column = st.selectbox("Select column with text data", df_string_columns)

        sentiment_col = f"Sentiment_{text_column.capitalize()}"

        if not st.session_state.get("sentiment_processed", False):
            st.write("Processing sentiment analysis...")
            df = st.session_state.loop.run_until_complete(
                process_dataframe(df, text_column, sentiment_col)
            )
            st.session_state.df = df
            st.session_state.sentiment_processed = True
            st.success("Sentiment analysis complete!")

        sentiment_counts = st.session_state.df[sentiment_col].value_counts()

        col1, col2 = st.columns(2)
        with col1:
            fig = px.pie(
                names=sentiment_counts.index,
                values=sentiment_counts.values,
                title="Sentiment Distribution",
            )
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig_bar = px.bar(
                sentiment_counts,
                x=sentiment_counts.index,
                y=sentiment_counts.values,
                title="Sentiment Count",
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        st.subheader(f"Word Cloud for {text_column.capitalize()}")
        sentiment_option = st.selectbox(
            "Select sentiment", ["Positive", "Negative", "Neutral"]
        )

        text_data = " ".join(
            st.session_state.df[st.session_state.df[sentiment_col] == sentiment_option][
                text_column
            ]
        )
        wordcloud = WordCloud(width=800, height=400, background_color="black").generate(
            text_data
        )

        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        st.pyplot(plt)

        csv = (
            st.session_state.df[[text_column, sentiment_col]]
            .to_csv(index=False)
            .encode("utf-8")
        )
        st.download_button("Download", csv, "analyzed_data.csv", "text/csv")

    if st.button("Reset"):
        st.session_state.df = None
        st.session_state.sentiment_processed = False
        st.rerun()


elif option == "Upload Text File":
    uploaded_file = st.file_uploader("Upload TXT File", type=["txt"])

    if uploaded_file is not None:
        st.session_state.uploaded_text = uploaded_file.read().decode("utf-8")
        st.text_area(
            "Uploaded Text:", st.session_state.uploaded_text, height=150, disabled=True
        )

        if st.button("Run"):
            st.write("Processing...")
            result = st.session_state.loop.run_until_complete(
                analyze_sentiment(st.session_state.uploaded_text)
            )
            st.success("Processing complete!")
            st.subheader("Sentiment Result")
            st.write(f"The input file's sentiment is '{result}'.")

    if st.button("Reset"):
        st.session_state.uploaded_text = None
        st.rerun()

elif option == "Enter Text":
    data = st.text_area("Enter text:", value=st.session_state.get("text_input", ""))

    if st.button("Run"):
        if data.strip():
            st.session_state.text_input = data
            st.write("Processing...")
            result = st.session_state.loop.run_until_complete(analyze_sentiment(data))
            st.success("Processing complete!")
            st.subheader("Sentiment Result")
            st.write(f"The input text's sentiment is '{result}'.")
        else:
            st.warning("Please provide input before running.")

    if st.button("Reset"):
        st.session_state.text_input = ""
        st.rerun()
