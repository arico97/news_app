import os
from news import News 
from fastapi import APIRouter, Request, FastAPI 


app = FastAPI()
api_router = APIRouter()

@api_router.get("/", status_code=200)
def version(): 
    return {"version": "V0"}


@api_router.post("/topics")
async def predic(request: Request): 
    body = await request.json()
    q = body["q"] 
    language = body["language"]

    print('start topic endpoint')

    topics=News(q,language).make_topics()
    
    print(topics)
    return {"success": True,
            "topics":topics}

app.include_router(api_router)