import os
import json
import pickle
import joblib
import pandas as pd
from flask import Flask, jsonify, request
from peewee import (
    Model, TextField, BooleanField, IntegrityError
)
from playhouse.shortcuts import model_to_dict
from playhouse.db_url import connect
import logging

class CustomRailwayLogFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "time": self.formatTime(record),
            "level": record.levelname,
            "message": record.getMessage()
        }
        return json.dumps(log_record)
    

def get_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO) # this should be just "logger.setLevel(logging.INFO)" but markdown is interpreting it wrong here...
    handler = logging.StreamHandler()
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    formatter = CustomRailwayLogFormatter()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


logger = get_logger()

DB = connect(os.environ.get('DATABASE_URL') or 'sqlite:///predictions.db')


class Prediction(Model):
    observation_id = TextField(unique=True)
    observation = TextField()
    label = BooleanField()
    predicted_outcome = BooleanField()
    
    class Meta:
        database = DB

DB.create_tables([Prediction], safe=True)

# End database stuff
########################################

########################################
# Unpickle the previously-trained model

with open('columns.json') as fh:
    columns = json.load(fh)

pipeline = joblib.load('pipeline.pickle')

with open('dtypes.pickle', 'rb') as fh:
    dtypes = pickle.load(fh)

# End model un-pickling
########################################

########################################
# Input validation functions

def check_valid_column(observation):
    """
    Validates that our observation only has valid columns
        
    Returns:
    - assertion value: True if all provided columns are valid, False otherwise
    - error message: empty if all provided columns are valid, False otherwise
    """
    
    valid_columns = {
        "id", "name", "sex", "dob", "race",
        "juv_fel_count", "juv_misd_count", "juv_other_count", "priors_count",
        "c_case_number", "c_charge_degree", "c_charge_desc", "c_offense_date",
        "c_arrest_date", "c_jail_in"
    }
    
    keys = set(observation.keys())
    
    if len(valid_columns - keys) > 0: 
        missing = valid_columns - keys
        error = "Missing columns: {}".format(missing)
        return False, error
    
    if len(keys - valid_columns) > 0: 
        extra = keys - valid_columns
        error = "Unrecognized columns provided: {}".format(extra)
        return False, error    

    return True, ""

########################################
# Begin webserver stuff

app = Flask(__name__)


@app.route('/will_recidivate', methods=['POST'])
def will_recidivate():
    observation = request.get_json()
    
    columns_ok, error = check_valid_column(observation)
    if not columns_ok:
        response = {'error': error}
        return jsonify(response)

    _id = observation['id']
    obs = pd.DataFrame([observation])
    
    label = pipeline.predict(obs)[0]
    response = {'id': _id, 'outcome': bool(label)}
    
    p = Prediction(
        observation_id=_id,
        label=bool(label),  # Store the boolean label directly
        observation=json.dumps(observation),  # Store observation as JSON string
        predicted_outcome=bool(label)  # Store the predicted outcome
    )
    
    try:
        p.save()
        logger.info(f"Observation saved: {_id}")
    except IntegrityError:
        error_msg = 'Observation ID: "{}" already exists'.format(_id)
        response['error'] = error_msg
        logger.warning(error_msg)
        DB.rollback()
        
    return jsonify(response)


@app.route('/recidivism_result', methods=['POST'])
def recidivism_result():
    observation = request.get_json()
    _id = observation['id']
    outcome = bool(observation['outcome'])  # Ensure outcome is boolean
    
    try:
        p = Prediction.get(Prediction.observation_id == _id)
        p.label = outcome
        p.save()
        
        obs_dict = json.loads(p.observation)
        predicted_outcome = p.predicted_outcome  # Retrieve the predicted outcome from the model
        
        response = {
            'id': _id,
            'outcome': outcome,
            'predicted_outcome': predicted_outcome
        }
        
        logger.info(f"Recidivism result for ID {_id}: {response}")
        return jsonify(response)
    
    except Prediction.DoesNotExist:
        error_msg = 'Observation ID: "{}" does not exist'.format(_id)
        logger.warning(error_msg)
        return jsonify({'error': error_msg})

    
@app.route('/list-db-contents')
def list_db_contents():
    contents = [model_to_dict(obs) for obs in Prediction.select()]
    logger.info(f"Database contents: {contents}")
    return jsonify(contents)

# End webserver stuff
########################################

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=5000)
