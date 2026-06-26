from crewai import Task
from tools import yt_tool  
from agents import blog_researcher, blog_writer

# Research task
research_task = Task(
    description= (
        "Identify the video {topic}"
        "Get detailed information about the video from the channel"
    ),
    expected_output= (
        "A comprehensive three paragraph long report based on the {topic} of the video content"
    ),
    tools= [yt_tool],
    agent= blog_researcher
)


# Writing task 
write_task = Task(
    description= (
        "Write a blog post based on the research done on {topic}"
        "Make sure to include all the relevant information"
    ),
    expected_output= (
        """
        A well-structured blog post with an introduction, body, and conclusion 
        on the topic {topic} and create the content for the blog post.
        """
    ),
    tools=[yt_tool],
    agent= blog_writer,
    async_execution=False,
    output_file="new_blog_post.md"  # Example of output customization
)