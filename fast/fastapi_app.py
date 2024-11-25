from fastapi import FastAPI
import time
from pydantic import BaseModel

app = FastAPI()

class InputData(BaseModel):
	number: int


@app.get("/")
async def root():
	return {"message":"Welcome to FastAPI"}

@app.post("/predict")
async def predict(data: InputData):
	time.sleep(2)
	result = {"input": data.number, "prediction": data.number * 2}
	return result
