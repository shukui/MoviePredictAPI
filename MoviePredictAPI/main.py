from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import pickle

# pydantic model
class MovieFeature(BaseModel):
    # Budget of the movie
    budget : float
    # IMDb rating of the movie
    imdb_score : float
    # The number of Facebook likes
    facebook_likes : int

# Import Linear Regression Model
try:
    with open("MoviePredict_LinearRegressionModel.pkl", 'rb') as file:
        ml_model = pickle.load(file)
        print(ml_model)
except Exception as e:
    print(f"Error loading the model: {e}")

# FastAPI
app = FastAPI()

@app.get("/index")
def index():
    return{'message': r'POST /predict {"budget": 180000000, "imdb_score": 8.5, "facebook_likes": 356789}'}

@app.post("/predict")
async def predict(movie_feature : MovieFeature):
    try:
        budget = movie_feature.budget
        imdb_score = movie_feature.imdb_score
        facebook_likes = movie_feature.facebook_likes
        prediction = ml_model.predict([[budget, imdb_score, facebook_likes]])
        return {'gross_prediction', prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app = app, host = "0.0.0.0", port = 8000)