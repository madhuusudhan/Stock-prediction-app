import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from datetime import datetime as dt, date, timedelta
import plotly.graph_objs as go
from newsapi import NewsApiClient
from requests.exceptions import HTTPError

# Set up the page layout
st.set_page_config(layout="wide")

def get_stock_price_fig(df):
    fig = px.line(df, x="Date", y=["Close", "Open"], title="Closing and Opening Price vs Date")
    return fig

def get_more(df):
    df['EWA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    fig = px.scatter(df, x="Date", y="EWA_20", title="Exponential Moving Average vs Date")
    fig.update_traces(mode='lines+markers')
    return fig

def prediction(stock, start_date, end_date,  n_days):
    # Download the stock data
    df = yf.download(stock, start=start_date, end=end_date)
    df.reset_index(inplace=True)
    df['Day'] = df.index

    # Prepare data for training and testing
    days = list(range(len(df)))
    X = [[day] for day in days]
    Y = df[['Close']].values

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, shuffle=False)

    # Set up GridSearchCV
    # param_grid={
    #         'C': [0.001, 0.01, 0.1, 1, 100, 1000],
    #         'epsilon': [
    #             0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10,
    #             50, 100, 150, 1000
    #         ],
    #         'gamma': [0.0001, 0.001, 0.005, 0.1, 1, 3, 5, 8, 40, 100, 1000]
    #     }
    param_grid={
            'C': [0.1, 1, 10, 100, 1000],
            'epsilon': [0.01, 0.1, 1, 10],
            'gamma': [0.001, 0.01, 0.1, 1]
        }

    gsc = GridSearchCV(
            estimator=SVR(kernel='rbf'),
            param_grid=param_grid,
            cv=5,
            scoring='neg_mean_absolute_error',
            verbose=0,
            n_jobs=-1
        )
   
    grid_result = gsc.fit(x_train, y_train.ravel())
    best_params = grid_result.best_params_
    best_svr = SVR(kernel='rbf', C=best_params["C"], epsilon=best_params["epsilon"], gamma=best_params["gamma"], max_iter=-1)
    best_svr.fit(x_train, y_train.ravel())

   
    output_days = [[x_test[-1][0] + i] for i in range(1, n_days + 1)]
    predictions = best_svr.predict(output_days)

    dates = [date.today() + timedelta(days=i) for i in range(1, n_days + 1)]
    prediction_df = pd.DataFrame({'Date': dates, 'Predicted Close Price': predictions})

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=predictions, mode='lines+markers', name='Predicted Close Price'))
    fig.update_layout(title=f"Predicted Close Price for the next {n_days} days", xaxis_title="Date", yaxis_title="Close Price")

    return fig, prediction_df

def get_company_news(ticker):
    
    newsapi = NewsApiClient(api_key='1da03ff6dc5f4122bdda5cd8b7c19034')  # Replace 'YOUR_NEWSAPI_KEY' with your actual API key

    try:
       
        news = newsapi.get_everything(q=ticker, language='en', sort_by='publishedAt', page_size=5)
        articles = news['articles']
        news_list = [{'title': article['title'], 'url': article['url'], 'publishedAt': article['publishedAt']} for article in articles]
    except Exception as e:
        st.error(f"Error fetching news articles: {e}")
        news_list = []

  
    for article in news_list:
        try:
            
            article_url = article['url']
            article_image = get_article_image(article_url)
            article['image'] = article_image
        except Exception as e:
            st.warning(f"Error fetching image for article: {e}")
            article['image'] = None

    return news_list

def get_article_image(article_url):
    
    import requests
    from bs4 import BeautifulSoup

    # Fetch the article HTML content
    response = requests.get(article_url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find the article image URL using BeautifulSoup
    image_tag = soup.find('meta', property='og:image')
    if image_tag:
        article_image = image_tag['content']
    else:
        article_image = None

    return article_image


def calculate_growth(df):
    start_price = df['Close'].iloc[0]
    end_price = df['Close'].iloc[-1]
    growth_percentage = ((end_price - start_price) / start_price) * 100
    return growth_percentage

# Sidebar inputs



st.sidebar.title("MarketInsight App")
ticker_input = st.sidebar.text_input("Input stock code:", value="AAPL")
start_date = st.sidebar.date_input("Start date:", dt(2022, 1, 1))
end_date = st.sidebar.date_input("End date:", dt.now())
n_days = st.sidebar.number_input("Number of days for forecast:", min_value=1, max_value=365, value=30)


# Forecast button
if st.sidebar.button("Forecast"):
    if ticker_input:
        try:
            
            df = yf.download(ticker_input, start=start_date, end=end_date)
            if df.empty:
                st.error(f"No data found for {ticker_input}. Please check the stock code.")
            else:
                df.reset_index(inplace=True)
                st.title(f"Stock Data for {ticker_input}")

                
                st.subheader("Stock Prices")
                fig = get_stock_price_fig(df)
                st.plotly_chart(fig)

                
                st.subheader("Indicators")
                fig = get_more(df)
                st.plotly_chart(fig)

               
                st.subheader("Growth Percentage")
                growth_percentage = calculate_growth(df)
                st.write(f"Growth Percentage: {growth_percentage:.2f}%")

                
                st.subheader("Forecast")
                forecast_fig, prediction_df = prediction(ticker_input, start_date, end_date,  n_days)
                st.plotly_chart(forecast_fig)
                st.write(prediction_df)

               
                st.subheader("Recent News")
                news_list = get_company_news(ticker_input)
                for news in news_list:
                    formatted_date = news['publishedAt'].split('T')[0]
                    st.write(f"[{news['title']}]({news['url']}) - {formatted_date}")

        except HTTPError as e:
            st.error(f"HTTP error occurred: {e}")
        except Exception as e:
            st.error(f"An error occurred: {e}")

