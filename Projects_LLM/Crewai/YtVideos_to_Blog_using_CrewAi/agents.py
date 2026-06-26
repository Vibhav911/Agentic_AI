from crewai import Agent
from tools import yt_tool
import os
from dotenv import load_dotenv

load_dotenv()


os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ["OPENAI_MODEL_NAME"] ="gpt-4o-mini-2024-07-18"

# Create a senior blog content researcher agent
blog_researcher = Agent(
    role = "Blog Researcher from youtube videos",
    goal = "get the relevant content for the topic {topic} from Youtube channel",
    verbose= True,
    memory = True,
    backstory= (
        """
        Expoert in understanding videos in AI, Data Science, Machine Leanring 
        and Gen AI and provide suggestions for blog content.
        """
    ),
    tools=[yt_tool],
    allow_delegation=True
)


# Creating a senior blog content writer agent with YT Tool
blog_writer = Agent(
    role = "Blog Writer",
    goal= "Narrate compelling tech stories about the video {topic} from Yt Channel",
    verbose= True,
    memory= True,
    backstory= (
        """
        With a flair of simplifying complex topics, you craft engaging narratives
        that captivate and educate, bringing new discoveries to light in an accessible 
        manner
        """
    ),
    tools=[],
    allow_delegation=False
)