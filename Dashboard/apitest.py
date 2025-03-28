from typing import Optional

from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
    return {"item_id": item_id, "q":q}


if __name__ == '__apitest__':
    uvicorn.run(app)
    
print("API test script is running!")

