"""Common Serving Functions for Both Classification and Regression"""

from __future__ import annotations
import builtins
import json
import logging
import os
import random
import joblib
# install required packages. Move to docker file in the future.
from io import BytesIO, StringIO
from typing import Any
import numpy as np
import pandas as pd
import xgboost as xgb

import shared.data_processing as dp
import shared.constants as mc

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

def _getTopFeatures(model, xArray, modelPredictors):
    booster = model.get_booster()
    dmatrix = xgb.DMatrix(xArray)
    contribs = booster.predict(dmatrix, pred_contribs=True)
    
    # Sum contributions across classes
    summedContribs = np.sum(contribs, axis=1)
    
    # Remove the bias term (last column)
    summedContribs = summedContribs[:, :-1]
    
    # Get the absolute values of contributions
    absContribs = np.abs(summedContribs)
    # Get the indices of the top 6 contributing features for each sample
    topIndices = np.argsort(-absContribs, axis=1)[:, :mc.NUM_TOP_FEATURES]
    explainList = [] 
    for rowNumber in range(len(topIndices)):
        indices = topIndices[rowNumber,:]
        values = absContribs[rowNumber,:]
        featureNames = [dp.getPrefix(modelPredictors[i]) for i in indices]
        featureValues = [values[i] for i in indices]  
        explainList.append([featureNames, featureValues])
    return explainList
    
def model_loading(
    model_dir: str,
    model_type: str | None = 'classification',
) -> dict:
    """Load model from previously saved artifact."""
    allFiles = os.listdir(model_dir)
    logger.info(f'All the files in model_dir is {allFiles}')
    modelName, encoderName, modelParamsName = None, None, None 
    # Set model_prefix and initialize model
    if model_type == 'classification':
        model_prefix = mc.CLASSIFICATION_OUTCOME
        model = xgb.XGBClassifier() 
    elif model_type == 'regression':
        model_prefix = mc.REGRESSION_OUTCOME
        model = xgb.XGBRegressor()
    else:
        msg = f"Only support model_type classifcation and regression, not {model_type}"
        raise ValueError(msg)
    for fileName in allFiles:
        if fileName.startswith(model_prefix):
            modelName = fileName 
        elif fileName.startswith('encoder'): 
            encoderName = fileName 
        elif fileName.startswith('modelParams'): 
            modelParamsName = fileName 
    if not (modelName and encoderName and modelParamsName): 
        msg = f"One of the model artificats is missing in {model_dir}."
        raise ValueError(msg)
    model.load_model(os.path.join(model_dir, modelName))
    # Load one-hot encoder
    encoder = joblib.load(os.path.join(model_dir, encoderName))
    # Load parameters
    with open(os.path.join(model_dir, modelParamsName), 'r') as f:
        modelParams = json.load(f)
    return {'model':model, 'encoder':encoder, 'modelParams':modelParams}


def model_inference(
    modelDct: dict,
    request_body: str,
    input_content_type: str,
    output_content_type: str = "application/json",
) -> builtins.tuple[str, str]:
    """
    Transform function to process input data and generate predictions.

    Args:
        modelDct: The trained model object.
        request_body: The input data.
        input_content_type: The content type of the input data.
        output_content_type: The desired content type of the output.

    Returns:
        The prediction output and the corresponding content type.
    """
    if input_content_type == "application/x-parquet":
        buf = BytesIO(request_body)
        data = pd.read_parquet(buf)

    elif input_content_type == "text/csv":
        buf = StringIO(request_body)
        data = pd.read_csv(buf)

    elif input_content_type == "application/json":
        buf = StringIO(request_body)
        data = pd.read_json(buf)

    elif input_content_type == "application/jsonl":
        buf = StringIO(request_body)
        data = pd.read_json(buf, orient="records", lines=True)

    else:
        msg = f"{input_content_type} input content type not supported."
        raise ValueError(msg)

    model = modelDct['model'] 
    encoder = modelDct['encoder'] 
    modelParams = modelDct['modelParams'] 
    medianValues = modelParams.get(mc.MEDIAN_VALUES_STR, None) 
    modelPredictors = modelParams.get(mc.PREDICTORS_STR, []) 
    dummiedPredictors = modelParams.get(mc.DUMMIED_PREDICTORS_STR, []) 

    predictionDf = data.pipe(
        func=dp.baseTransformDf,
        medianValues=medianValues,        
    ).pipe(
        func=dp.dummyPredictors, 
        enc=encoder
    ).pipe(
        func=dp.setMissingPredictorValuesToZero,
        modelPredictors=modelPredictors
    )

    xArray = predictionDf[modelPredictors] 
    visitIds = predictionDf[mc.VISIT_ID_STR].tolist() 
    # Predict
    yArray = model.predict(xArray)
    
    explainList = _getTopFeatures(model, xArray, modelPredictors)
    predictions = pd.DataFrame({mc.VISIT_ID_STR:visitIds, 
                            mc.PREDICTED_DISCHARGE_DATE_OUTCOME_RESPONSE_STR:yArray.tolist(),
                            mc.OUTCOME_EXPLANATION_STR:explainList})
    output = {"predictions": predictions.to_dict("records")}
    
    if "application/json" in output_content_type:
        output = json.dumps(output, indent=4)
        output_content_type = "application/json"
    else:
        msg = f"{output_content_type} content type not supported"
        raise ValueError(msg)

    return output, output_content_type
