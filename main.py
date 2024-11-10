from os.path import exists
from googleapiclient.discovery import build
import pandas as pd
from time import sleep
import traceback
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.docstore.document import Document
import re
from langchain_text_splitters import CharacterTextSplitter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from langchain.prompts import PromptTemplate
from fastapi import FastAPI
import uvicorn
from models.video import VideoResult
from resources.constants import Extract_Comments_Path

load_dotenv()

api_key = os.getenv("API_KEY")

YouTube = build('YouTube', 'v3', developerKey=os.getenv("API_KEY"))

# Initialize the SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

os.environ["GOOGLE_API_KEY"] = os.getenv("API_KEY")

# Initialize Model
llm = ChatGoogleGenerativeAI(model="gemini-pro")

def get_summary(text):
    # Define the Summarize Chain
    template = """Write a concise summary of the following:
    "{text}"
    CONCISE SUMMARY:"""

    prompt = PromptTemplate.from_template(template)

    llm_chain = LLMChain(llm=llm, prompt=prompt)
    stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")

    # Invoke Chain
    response = stuff_chain.invoke(text)
    return response["output_text"]

def get_comments(api_key, video_id):
    YouTube = build('YouTube', 'v3', developerKey=api_key)

    request = YouTube.commentThreads().list(
        part="snippet,replies",
        videoId=video_id,
        textFormat="plainText"
    )

    df = pd.DataFrame(columns=['comment', 'replies', 'date', 'user_name'])

    while request:
        replies = []
        comments = []
        dates = []
        user_names = []

        try:
            response = request.execute()

            for item in response['items']:
                # Extracting comments
                comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                comments.append(comment)

                user_name = item['snippet']['topLevelComment']['snippet']['authorDisplayName']
                user_names.append(user_name)

                date = item['snippet']['topLevelComment']['snippet']['publishedAt']
                dates.append(date)

                # counting number of reply of comment
                replycount = item['snippet']['totalReplyCount']

                # if reply is there
                if replycount > 0:
                    # append empty list to replies
                    replies.append([])
                    # iterate through all reply
                    for reply in item['replies']['comments']:
                        # Extract reply
                        reply = reply['snippet']['textDisplay']
                        # append reply to last element of replies
                        replies[-1].append(reply)
                else:
                    replies.append([])

            # create new dataframe
            df2 = pd.DataFrame({"comment": comments, "replies": replies, "user_name": user_names, "date": dates})
            df = pd.concat([df, df2], ignore_index=True)
            df.to_csv(f"{Extract_Comments_Path}+{video_id}_user_comments.csv", index=False, encoding='utf-8')
            sleep(2)
            request = YouTube.commentThreads().list_next(request, response)
            print("Iterating through next page")
        except Exception as e:
            print(str(e))
            print(traceback.format_exc())
            print("Sleeping for 10 seconds")
            sleep(10)
            df.to_csv(f"{Extract_Comments_Path} + {video_id}_user_comments.csv", index=False, encoding='utf-8')
            break
    return df

def merge_comments(comments):
    sentences = ""
    for index, comment in list(comments.items()):
        sentence = re.sub(r'[^a-zA-Z0-9\s]', '', comment)
        # Remove extra spaces
        sentences = sentences + ' '.join(sentence.split()) + "."
    return sentences

# Function to map sentiment scores to 1-10
def scale_sentiment_to_1_10(comment):
    sentiment = analyzer.polarity_scores(comment)
    compound_score = sentiment['compound']
    # Map compound score to a scale from 1 to 10
    if compound_score <= -0.5:
        return 1  # Highly negative
    elif compound_score <= 0:
        return 5  # Neutral
    else:
        return 10  # Positive

def sentiment_scale_analysis(comments):
    # Define the classification template
    classification_template = """
    You are a helpful assistant who categorizes user comments on a scale from 1 to 10 based on the tone and content of the comment:
    1. **1** = Extremely abusive, harmful, discriminatory, or threatening comments.
    2. **5** = Neutral or mildly critical comments, offering constructive criticism or general feedback.
    3. **10** = Highly positive, supportive, encouraging, or kind comments.

    Assign a score from 1 to 10 to the following comment based on its tone:
    Comment: "{text}"
    Score (1-10):
    """
    # Example of using this prompt for categorization
    prompt = PromptTemplate.from_template(classification_template)
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    response = llm_chain.run({"text": comments})
    return response

def get_viewcount_likes_dislikes(video_id):
    # Call the YouTube Data API
    request = YouTube.videos().list(
        part="statistics",
        id=video_id
    )
    response = request.execute()
    # Extracting likes and dislikes
    likes = response['items'][0]['statistics'].get('likeCount', 0)
    dislikes = response['items'][0]['statistics'].get('dislikeCount', 0)
    view_count = response["items"][0]["statistics"]["viewCount"]
    return view_count, likes, dislikes

def get_title(video_id):
    request = YouTube.videos().list(
        part="snippet",
        id=video_id
    )
    response = request.execute()
    # Extract the title from the response
    title = response["items"][0]["snippet"]["title"] if response["items"] else "Video not found"
    return title

app = FastAPI()

@app.post("/result/{video_id}")
def get_result_json(video_id: str):
    # You have to extract the video id from the YouTube url
    # video_id = "q_Q4o0sMBBA"  # (Scale: 10) FULL SPEECH: Trump projected winner of 2024 presidential election : https://www.YouTube.com/watch?v=q_Q4o0sMBBA
    # video_id = "XIwoWyG7vyQ" # (Scale: 10) South Korean K-pop star Sulli found dead... suspected suicide: https://www.YouTube.com/watch?v=XIwoWyG7vyQ
    # video_id = "mzJzCj9IqIE" #(Scale: 2) Trudeau and Poilievre debate inflation and the cost of living after U.S. election results : https://www.YouTube.com/watch?v=mzJzCj9IqIE
    # video_id = "w3ugHP-yZXw"  #(Scale: 1) GHOSTBUSTERS - Official Trailer (HD): https://www.YouTube.com/watch?v=w3ugHP-yZXw

    '''
    Storing YouTube video's comments in csv file, which takes time depends upon no of comments.
    If we have already fetch respective video's comments then directly read csv file, else
    fetch its comments using API.
    '''
    file_name = Extract_Comments_Path + video_id + "_user_comments.csv"
    if exists(file_name) == False:
        df = get_comments(api_key, video_id)
        print("YouTube video's comments fetched successfully")
    else:
        df = pd.read_csv(file_name)

    # Considering only top 500 commnets for analysis.
    top_1000_rows = df.head(500)
    # Merge all user comments
    merged_comments = merge_comments(top_1000_rows['comment'])

    '''Split merged user comments and create Documents for each chunk as input suppose 
        to have page_content attribute in specific format to get summary of input text.
    '''
    text_splitter = CharacterTextSplitter()
    texts = text_splitter.split_text(merged_comments)
    docs = [Document(page_content=t) for t in texts]
    summary = get_summary(docs)
    print("Successfully summarize user comments.")

    # Get title of give YouTube video
    title = get_title(video_id)
    print("Successfully fetch title of video.")

    # Get view count, no of likes and dislikes of respective YouTube video.
    view_count, likes, dislikes = get_viewcount_likes_dislikes(video_id)
    print("Successfully fetched YouTube video's statistics.")

    # Get sentiment score of user comments.
    sentiment_score = sentiment_scale_analysis(texts)
    print("Successfully calculate YouTube video's sentiment score.")

    # Create json object of all result.
    '''data = {
        "summary": summary,
        "view_count": view_count,
        "likes": likes,
        "dislikes": dislikes,
        "sentiment_score": sentiment_score
    }
    json_data = json.dumps(data, indent=4)'''

    video = VideoResult()
    video.video_id = video_id
    video.title = title
    video.summary = summary
    video.view_count = view_count
    video.no_of_likes = likes
    video.no_of_dislikes = dislikes
    video.sentiment_score = sentiment_score
    return video

if __name__ == '__main__':
    uvicorn.run(app, port=36114, host='127.0.0.1')

