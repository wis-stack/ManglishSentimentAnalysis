import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import asyncio
from model_implementation import predict

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)


async def analyze_sentiment(text):
    return await predict(text)


async def process_dataframe(df, text_column, sentiment_col):
    tasks = [analyze_sentiment(text) for text in df[text_column]]
    sentiments = await asyncio.gather(*tasks)
    df[sentiment_col] = sentiments
    return df


st.title("Manglish Sentiment Analysis")

if "processed_df" not in st.session_state:
    st.session_state.processed_df = None


def reset_app():
    for key in st.session_state.keys():
        del st.session_state[key]
    st.session_state.processed_df = None


option = st.radio(
    "Select input method:", ("Upload CSV", "Upload Text File", "Enter Text")
)

if option == "Upload CSV":
    uploaded_file = st.file_uploader(
        "Upload CSV File", type=["csv"], key="csv_uploader"
    )

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.subheader("Raw Data")
        st.dataframe(df)

        df_string_columns = [col for col in df.columns if df[col].dtype == "object"]
        text_column = st.selectbox(
            "Select column with text data", df_string_columns, key="csv_text_column"
        )
        sentiment_col = f"Sentiment_{text_column.capitalize()}"

        if (
            st.session_state.processed_df is None
            or st.session_state.text_column != text_column
        ):
            st.write("Processing sentiment analysis...")
            processed_df = loop.run_until_complete(
                process_dataframe(df, text_column, sentiment_col)
            )
            st.session_state.processed_df = processed_df
            st.session_state.text_column = text_column
            st.success("Sentiment analysis complete!")

        else:
            processed_df = st.session_state.processed_df
            st.success("Using cached analysis!")

        sentiment_counts = processed_df[sentiment_col].value_counts()

        st.subheader(f"Charts for {text_column.capitalize()}")
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
            "Select sentiment",
            ["Positive", "Negative", "Neutral"],
            key="csv_sentiment_option",
        )
        text_data = " ".join(
            processed_df[processed_df[sentiment_col] == sentiment_option][text_column]
        )
        wordcloud = WordCloud(width=800, height=400, background_color="black").generate(
            text_data
        )

        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        st.pyplot(plt)

        csv = (
            processed_df[[text_column, sentiment_col]]
            .to_csv(index=False)
            .encode("utf-8")
        )
        st.download_button("Download", csv, "analyzed_data.csv", "text/csv")

        st.button("Reset", on_click=reset_app)

    else:
        reset_app()

elif option == "Upload Text File":
    uploaded_file = st.file_uploader(
        "Upload TXT File", type=["txt"], key="txt_uploader"
    )

    if uploaded_file is not None:
        uploaded_text = uploaded_file.read().decode("utf-8")
        st.text_area(
            "Uploaded Text:",
            uploaded_text,
            height=150,
            disabled=True,
            key="txt_text_area",
        )

        if st.button("Run"):
            st.write("Processing...")
            result = loop.run_until_complete(analyze_sentiment(uploaded_text))
            st.success("Processing complete!")
            st.subheader("Sentiment Result")
            st.write(f"The input file's sentiment is '{result}'.")
            st.button("Reset", on_click=reset_app)

    else:
        reset_app()

elif option == "Enter Text":
    data = st.text_area("Enter text:", key="text_input")

    if st.button("Run"):
        if data.strip():
            st.write("Processing...")
            result = loop.run_until_complete(analyze_sentiment(data))
            st.success("Processing complete!")
            st.subheader("Sentiment Result")
            st.write(f"The input text's sentiment is '{result}'.")
        else:
            st.warning("Please provide input before running.")
        st.button("Reset", on_click=reset_app)

    else:
        reset_app()
