from fastapi import FastAPI
from routes import router  # If routes.py is in the same package; use "from routes import router" if not

app = FastAPI()
app.include_router(router)
