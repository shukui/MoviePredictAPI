# Movie Gross Prediction

This project is a machine learning application that predicts the gross income of a movie based on its budget, IMDb score, and Facebook likes.

## Overview

The project consists of two main components:

- A linear regression model ( MoviePredict_LinearRegressionModel.pkl ) that is trained on a dataset of movies with their budget, IMDb score, Facebook likes, and gross income. 
- A REST API that exposes the model as a web service using FastAPI framework and pedantic for validation. The API has one endpoint (/predict) that accepts a POST request with a JSON body that contains the budget, IMDb score, and Facebook likes of a movie. 
  	{
      "budget": 180000000,
      "imdb_score": 8.5,
      "facebook_likes": 356789
    }
- The API returns a JSON response with the predicted gross income of the movie.
    {
      "gross_prediction": 500000.98
    }
