import os
import sys
import pandas as pd
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from houseprice1.pipelines.prediction_pipeline import PredictPipeline
from houseprice1.pipelines.training_pipeline import TrainPipeline
from src.houseprice1.exception import CustomException
from src.houseprice1.logger import logging

app = FastAPI(
	title="House Price Prediction API"
)

class HousePredictionRequest(BaseModel):
	OverallQual: int = Field(description="Overall material and finish quality")
	GrLivArea: int = Field(description="Above grade (ground) living area square feet")
	TotalBsmtSF: int = Field(description="Total square feet of basement area")
	FullBath: int = Field(default=1,description="Full bathrooms above grade")
	YearBuilt: int = Field(default=1900, description="Original construction date")
	TotRmsAbvGrd: int = Field(description="Total rooms above grade")
	Fireplaces: int = Field(default=0, description="Number of fireplaces")
	WoodDeckSF: int = Field(description="Wood deck area in square feet")
	OpenPorchSF: int = Field(description="Open porch area in square feet")
	ExterQual: str = Field(default='Po', description="Evaluates the quality of the material on the exterior ")
	BsmtQual: str = Field(default='Po', description="Evaluates the height of the basement")
	KitchenQual: str = Field(default='Po',description="Kitchen quality")
	GarageFinish: str = Field(default='Unf',description="Interior finish of the garage")
	MSZoning: str = Field(description="The general zoning classification")
	CentralAir: str = Field(default='N',description="Central air conditioning")
	Neighborhood: str = Field(default='OldTown',description="Physical locations within Ames city limits")

@app.post("/predict")
async def predict_endpoint(request: HousePredictionRequest):
	try:
		data_dict = request.model_dump()

		input_df = pd.DataFrame([data_dict])
		predict_pipeline = PredictPipeline()

		final_price = predict_pipeline.predict(input_df)

		return {
            "status": "success",
            "predicted_price": float(final_price),
            "unit": "USD"
        }
	
	except Exception as e:
		custom_error = CustomException(e, sys)
		logging.error(custom_error.error_message)

		raise HTTPException(
            status_code=500, 
            detail="System error. Please check the request values!!!"
        )

@app.post("/train")
async def re_train_model():
	try:
		train_pipeline = TrainPipeline()
		train_pipeline.run_pipeline()

		return {
            "status": "success",
            "message": "successful trained!",
            "unit": "USD"
        }
	except Exception as ex:
		raise HTTPException(
            status_code=500, 
            detail="Trained error. Please check the request values!!!"
        )
	
if __name__ == "__main__":
	uvicorn.run(app, port=8000)

