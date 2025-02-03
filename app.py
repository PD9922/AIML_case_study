from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from chatbot import answer_query

app = FastAPI(title="Ubuntu Q&A Chatbot", description="Chatbot using FAISS vector store and gpt-4o-mini LLM")

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str

@app.post("/chat", response_model=QueryResponse)
def chat_endpoint(request: QueryRequest):
    try:
        answer = answer_query(request.query)
        return QueryResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# For running via "uvicorn app:app" command
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)