import streamlit as st
from crewai import Agent, Task, Crew
import litellm
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="Travel Assistant Chatbot", layout="centered")
st.title("Hello, I am your Travel Assistant!")

# User Input
destination = st.text_input("Enter your destination:")

# Dummy Function for Flights
def check(destination):
    print(f"Fetching flight details for: {destination}")
    if destination.lower() == 'japan':
        flights = [
            {'airline': 'MiddleEast', 'price': '1299$', 'departure': '10/4/2025'},
            {'airline': 'Turkish', 'price': '1500$', 'departure': '11/4/2025'}
        ]
    elif destination.lower() == 'canada':
        flights = [
            {'airline': 'Emirates', 'price': '1900$', 'departure': '12/5/2025'},
            {'airline': 'Turkish', 'price': '1880$', 'departure': '13/5/2025'}
        ]
    else:
        flights = [{'airline': 'None', 'price': 'N/A', 'departure': 'N/A'}]

    output = "Available Flights:\n"
    for f in flights:
        output += f"- Airline: {f['airline']}, Price: {f['price']}, Departure: {f['departure']}\n"
    return output

# LLM-backed Functions (Kept for hotels, tours, and advice)
def checkhotels(destination):
    res = litellm.completion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": f"List famous hotels in {destination}"}],
        temperature=0.3,
        api_key=API_KEY
    )
    return res['choices'][0]['message']['content']

def tour(destination):
    res = litellm.completion(
        model="gpt-3.5-turbo",
        temperature=0.3,
        messages=[{"role": "user", "content": f"List places to be visited in {destination}"}],
        api_key=API_KEY
    )
    return res['choices'][0]['message']['content']

def advices(destination):
    res = litellm.completion(
        model="gpt-3.5-turbo",
        temperature=0.3,
        messages=[{"role": "user", "content": f"List advices before traveling to {destination}"}],
        api_key=API_KEY
    )
    return res['choices'][0]['message']['content']

# Agents are defined once (llm=None disables GPT use)
flight_agent = Agent(
    role="Reservation Specialist",
    goal="Find the flights for the user",
    llm=None,
    backstory="I specialize in helping users find and book flights to various destinations."
)

hotel_agent = Agent(
    role="Hotel Specialist",
    goal="Find the best hotels for the user",
    llm=None,
    backstory="I specialize in recommending the best hotels and accommodations for travelers."
)

tour_agent = Agent(
    role="Tour Specialist",
    goal="Find the best tours for the user",
    llm=None,
    backstory="I provide information on the best places to visit and special tours for tourists."
)

advice_agent = Agent(
    role="Advice Specialist",
    goal="Provide the user with advice",
    llm=None,
    backstory="I give tips and important advice to help tourists have a smooth and enjoyable trip."
)

# ✅ Run only after destination is entered
if destination:
    # Define tasks AFTER destination input
    flight_task = Task(
        description="Find available airlines with price and departure",
        expected_output="Provide a brief summary of the best flight option for the user",
        agent=flight_agent,
        llm=None,  # No LLM for flights
        task_function=lambda: check(destination)  # Ensure the dummy function is used here
    )

    hotel_task = Task(
        description=f"Find best hotels",
        expected_output="A numbered list of 5 hotel names with brief descriptions",
        agent=hotel_agent,
        llm=None,
        task_function=lambda: checkhotels(destination)
    )

    tour_task = Task(
        description=f"Give me special tour and places to visit",
        expected_output="A numbered list of 5 places names with brief descriptions",
        agent=tour_agent,
        llm=None,
        task_function=lambda: tour(destination)
    )

    advice_task = Task(
        description=f"Give me advice I should pay attention to during my visit",
        expected_output="Provide a list of advice the tourist should know before traveling",
        agent=advice_agent,
        llm=None,
        task_function=lambda: advices(destination)
    )

    # Define the Crew with tasks
    crew = Crew(
        agents=[flight_agent, hotel_agent, tour_agent, advice_agent],
        tasks=[flight_task, hotel_task, tour_task, advice_task],
        process="sequential"
    )

    
    result = crew.kickoff()  

    if "history" not in st.session_state:
        st.session_state.history = []

    # Collect and display results from all tasks
    for task_output in result.tasks_output:
        st.session_state.history.append({"question": destination, "answer": task_output.raw})

    for chat in st.session_state.history:
        st.markdown(f"**You:** {chat['question']}")
        st.markdown(f"**Bot:** {chat['answer']}")
        st.markdown("----------")

