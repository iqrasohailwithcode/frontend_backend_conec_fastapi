from agents import Agent, Runner, OpenAIChatCompletionsModel, set_tracing_disabled
from fastapi import FastAPI
from openai import AsyncOpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os

load_dotenv()
set_tracing_disabled(disabled=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
BASE_URL = os.getenv("BASE_URL")

client = AsyncOpenAI(api_key=GEMINI_API_KEY, base_url=BASE_URL)

model = OpenAIChatCompletionsModel(
    openai_client=client,
    model="gemini-2.5-flash"
)

main_agent = Agent(
    name="Python Assistant",
    instructions="You are a python assistant always respond about Python.",
    model=model,
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Hello from Sohail!"}

class ChatMessage(BaseModel):
    message: str

@app.post("/chat")
async def chat_with_agent(request: ChatMessage):
    result = await Runner.run(main_agent, request.message)
    return {"response": result.final_output}
