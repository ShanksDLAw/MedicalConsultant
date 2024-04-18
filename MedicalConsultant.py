#The aim of this project is to build a preliminary medical consultant using crew ai

#Importing the necessary Packages needed to work on this project
import os 
from crewai import Agent, Task, Crew
from langchain_community.llms import Ollama
from crewai_tools import SerperDevTool
from dotenv import load_dotenv, find_dotenv

# Load environment variables
_ = load_dotenv(find_dotenv())
SERPER_API_KEY = os.getenv('SERPER_API_KEY')

llm = Ollama(model='openhermes')

search_tool = SerperDevTool()
    
#Setting up the Medical Agent researcher
medical_researcher = Agent(
    llm = llm,
    role = "Medical Doctor",
    goal = "Provide thorough Medical advise for treatment",
    backstory = "You are an Experienced Medical Doctor. In this case you offer medical advise on every ailment asked",
    allow_delegation = False,
    tools = [search_tool],
    verbose = True,
)

#Setting up the task
task_A = Task(
    description = "Search the internet and find medical treatment and answer in a friendly way",
    expected_output = "A friendly report for each medical treatment asked",
    agent = medical_researcher,
    output_file = "medical_output.txt",
)

crew = Crew(agents=[medical_researcher], tasks=[task_A], verbose=5)
output = crew.kickoff()
print(output)
