from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import pickle

app = FastAPI()

try:
    with open("MoviePredict_LinearRegressionModel.pkl", 'rb') as file:
        ml_model = pickle.load(file)
except Exception as e:
    print(f"Error loading the model: {e}")

class Movie(BaseModel):
    budget : float
    imdb_score : float
    facebook_likes : int

@app.get("/index")
def index():
    return{'message': 'Hello, given following parameters: budget, imdb_score, facebook_likes'}

@app.post("/predict")
async def predict(data : Movie):
    try:
        data = data.dict()
        budget = data['budget']
        imdb_score = data['imdb_score']
        facebook_likes = data['facebook_likes']
        prediction = ml_model.predict([[budget, imdb_score, facebook_likes]])
        return {'gross_prediction', prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app=app, host="0.0.0.0", port = 8000)
