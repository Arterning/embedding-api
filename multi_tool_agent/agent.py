import datetime
import os
from zoneinfo import ZoneInfo
from google.adk.agents import Agent
import praw
from dotenv import load_dotenv

load_dotenv()

def get_reddit_reviews(topic: str) -> dict:
    """
    Fetches and summarizes Reddit posts about a given topic to gauge sentiment or find reviews.

    Args:
        topic (str): The topic to search for on Reddit (e.g., "mac mini").

    Returns:
        dict: A dictionary containing the status and a report summarizing the findings, or an error message.
    """
    try:
        reddit = praw.Reddit(
            client_id=os.getenv("REDDIT_CLIENT_ID"),
            client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
            user_agent=os.getenv("REDDIT_USER_AGENT") or "SaaSOpportunityFinder/1.0",
        )
        # Search for the topic in relevant subreddits, focusing on reviews
        submissions = reddit.subreddit("all").search(
            f"{topic} review", limit=5, sort="relevance"
        )

        report_parts = []
        for submission in submissions:
            title = submission.title
            # Try to get selftext, otherwise take top 3 comments
            content = submission.selftext
            if not content:
                submission.comments.replace_more(limit=0)
                comments = [
                    comment.body
                    for comment in submission.comments[:3]
                    if not comment.stickied
                ]
                content = "\n".join(comments)

            report_parts.append(f"Title: {title}\nContent:\n{content[:500]}...")

        if not report_parts:
            return {
                "status": "success",
                "report": f"No relevant Reddit reviews found for '{topic}'.",
            }

        return {
            "status": "success",
            "report": "\n\n---\n\n".join(report_parts),
        }
    except Exception as e:
        return {"status": "error", "error_message": f"An error occurred: {e}"}


def get_weather(city: str) -> dict:
    """Retrieves the current weather report for a specified city.

    Args:
        city (str): The name of the city for which to retrieve the weather report.

    Returns:
        dict: status and result or error msg.
    """
    if city.lower() == "new york":
        return {
            "status": "success",
            "report": (
                "The weather in New York is sunny with a temperature of 25 degrees"
                " Celsius (77 degrees Fahrenheit)."
            ),
        }
    else:
        return {
            "status": "error",
            "error_message": f"Weather information for '{city}' is not available.",
        }


def get_current_time(city: str) -> dict:
    """Returns the current time in a specified city.

    Args:
        city (str): The name of the city for which to retrieve the current time.

    Returns:
        dict: status and result or error msg.
    """

    if city.lower() == "new york":
        tz_identifier = "America/New_York"
    else:
        tz_identifier = "UTC"

    tz = ZoneInfo(tz_identifier)
    now = datetime.datetime.now(tz)
    report = (
        f'The current time in {city} is {now.strftime("%Y-%m-%d %H:%M:%S %Z%z")}'
    )
    return {"status": "success", "report": report}


root_agent = Agent(
    name="multi_capability_agent",
    model="gemini-2.0-flash",
    description=(
        "Agent for answering questions about time, weather, and summarizing Reddit reviews."
    ),
    instruction=(
        "You are a helpful agent who can answer user questions about the time and "
        "weather in a city, and also find and summarize reviews from Reddit on a given topic."
    ),
    tools=[get_weather, get_current_time, get_reddit_reviews],
)