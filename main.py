from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from model import generate_analysis

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/analizar")
def analizar(ticker: str):
    return generate_analysis(ticker)