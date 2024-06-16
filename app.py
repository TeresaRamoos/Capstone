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
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database setup
DB = connect(os.environ.get('DATABASE_URL') or 'sqlite:///predictions.db')

class Prediction(Model):
    observation_id = TextField(unique=True)
    observation = TextField()
    label = BooleanField()
    predicted_outcome = BooleanField()
    
    class Meta:
        database = DB

DB.create_tables([Prediction], safe=True)

# Unpickle the previously-trained model
with open('columns.json') as fh:
    columns = json.load(fh)

pipeline = joblib.load('pipeline.pickle')

with open('dtypes.pickle', 'rb') as fh:
    dtypes = pickle.load(fh)

# Input validation
def check_valid_column(observation):
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

# Flask app setup
app = Flask(__name__)

@app.route('/will_recidivate', methods=['POST'])
def will_recidivate():
    observation = request.get_json()
    columns_ok, error = check_valid_column(observation)
    if not columns_ok:
        response = {'error': error}
        return jsonify(response), 400

    _id = observation['id']

    if Prediction.select().where(Prediction.observation_id == _id).exists():
        response = {'error': f'Observation ID: "{_id}" already exists'}
        return jsonify(response), 400

    try:
        obs = pd.DataFrame([observation], columns=columns)
    except ValueError as e:
        return jsonify({'error': 'DataFrame creation failed, missing required columns.'}), 400
    
    try:
        label = pipeline.predict(obs)[0]
    except KeyError as e:
        return jsonify({'error': f'Prediction failed, missing columns in data: {str(e)}'}), 400
    
    response = {'id': _id, 'outcome': bool(label)}
    
    p = Prediction(
        observation_id=_id,
        label=bool(label),
        observation=json.dumps(observation),
        predicted_outcome=bool(label)
    )
    
    try:
        p.save()
    except IntegrityError:
        response = {'error': f'Observation ID: "{_id}" already exists'}
        DB.rollback()
        return jsonify(response), 400
    
    return jsonify(response)

@app.route('/recidivism_result', methods=['POST'])
def recidivism_result():
    observation = request.get_json()
    _id = observation['id']
    outcome = bool(observation['outcome'])
    
    try:
        p = Prediction.get(Prediction.observation_id == _id)
        p.label = outcome
        p.save()
        
        response = {
            'id': _id,
            'outcome': outcome,
            'predicted_outcome': p.predicted_outcome
        }
        return jsonify(response)
    
    except Prediction.DoesNotExist:
        return jsonify({'error': f'Observation ID: "{_id}" does not exist'}), 400

@app.route('/list-db-contents')
def list_db_contents():
    contents = [model_to_dict(obs) for obs in Prediction.select()]
    return jsonify(contents)

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
