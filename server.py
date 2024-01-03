import sentenceCompleter
import fastapi


app = fastapi.FastAPI()

# run with - uvicorn server:app --host 0.0.0.0 --port 8080 --reload
@app.on_event("startup")
async def startup_event():
  print("Starting Up")

@app.on_event("shutdown")
async def startup_event():
  print("Shutting down")


@app.get("/")
def on_root():
  return { "message": "Hello App" }


@app.post("/tell_me_stories")
async def tell_me_stories(request: fastapi.Request):
  text = (await request.json())["text"] #get text input from website
  result = sentenceCompleter.generate(text) # generate text output 
  print("text", text)
  print("result", result)
  return result 
