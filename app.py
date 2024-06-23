import os
import json
import pickle
import joblib
import pandas as pd
import hashlib
import base64
from flask import Flask, jsonify, request
from peewee import Model, TextField, BooleanField, IntegrityError
from playhouse.shortcuts import model_to_dict
from playhouse.db_url import connect
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

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
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    formatter = CustomRailwayLogFormatter()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

logger = get_logger()

DB = connect(os.environ.get('DATABASE_URL') or 'sqlite:///predictions.db')

class BooleanFieldWithNone(BooleanField):
    def db_value(self, value):
        return None if value is None else bool(value)

    def python_value(self, value):
        return None if value is None else bool(value)

class Prediction(Model):
    observation_id = TextField(unique=True)
    observation = TextField()
    outcome = BooleanFieldWithNone(null=True)
    predicted_outcome = BooleanField()

    class Meta:
        database = DB

DB.create_tables([Prediction], safe=True)

with open('columns.json') as fh:
    columns = json.load(fh)

pipeline = joblib.load('pipeline.pickle')

with open('dtypes.pickle', 'rb') as fh:
    dtypes = pickle.load(fh)

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

def preprocess_data(df):
    df = pd.DataFrame([df])

    # Convert to correct types and handle null values
    df['dob'] = pd.to_datetime(df['dob'], errors='coerce')
    df['c_offense_date'] = pd.to_datetime(df['c_offense_date'], errors='coerce')
    df['c_arrest_date'] = pd.to_datetime(df['c_arrest_date'], errors='coerce')
    df['c_jail_in'] = pd.to_datetime(df['c_jail_in'], errors='coerce')

    df['unified_date'] = df['c_arrest_date'].combine_first(df['c_offense_date'])

    df['age_at_unified_date'] = (df['c_jail_in'] - df['dob']).dt.days // 365
    df['age_at_unified_date'] = df['age_at_unified_date'].fillna(df['age_at_unified_date'].mean())
    
    avg_time_offense_arrest = (df['c_arrest_date'] - df['c_offense_date']).dt.days.mean()
    avg_time_since_jail_in = (pd.to_datetime('today') - df['c_jail_in']).dt.days.mean()
    avg_time_to_jail = (df['c_jail_in'] - df['unified_date']).dt.days.mean()

    df['time_offense_arrest'] = (df['c_arrest_date'] - df['c_offense_date']).dt.days.fillna(avg_time_offense_arrest)
    df['time_since_jail_in'] = (pd.to_datetime('today') - df['c_jail_in']).dt.days.fillna(avg_time_since_jail_in)
    df['time_to_jail'] = (df['c_jail_in'] - df['unified_date']).dt.days.fillna(avg_time_to_jail)
    
    df['total_juv_crimes'] = df['juv_fel_count'].fillna(0) + df['juv_misd_count'].fillna(0) + df['juv_other_count'].fillna(0)
    df['total_adult_crimes'] = df['priors_count'].fillna(0) - df['total_juv_crimes']

    freq_encoding = df['c_charge_desc'].value_counts(normalize=True)
    df['c_charge_desc_freq'] = df['c_charge_desc'].map(freq_encoding).fillna(0)

    df['offense_month'] = df['c_offense_date'].dt.month.fillna(df['c_offense_date'].dt.month.mean())
    df['offense_day_of_week'] = df['c_offense_date'].dt.dayofweek.fillna(df['c_offense_date'].dt.dayofweek.mean())
    df['arrest_month'] = df['c_arrest_date'].dt.month.fillna(df['c_arrest_date'].dt.month.mean())
    df['arrest_day_of_week'] = df['c_arrest_date'].dt.dayofweek.fillna(df['c_arrest_date'].dt.dayofweek.mean())

    return df

def encode_name(name):
    return hashlib.sha256(name.encode('utf-8')).hexdigest()

def generate_id_from_observation(observation, prefix="id_"):
    observation_str = json.dumps(observation, sort_keys=True)
    md5_hash = hashlib.md5(observation_str.encode('utf-8')).digest()
    base64_id = base64.urlsafe_b64encode(md5_hash).decode('utf-8').rstrip('=')
    return f"{prefix}{base64_id}"

app = Flask(__name__)

@app.route('/will_recidivate/', methods=['POST'])
def will_recidivate():
    observation = request.get_json()

    logger.debug(f"Received observation: {observation}")

    if observation.get('name'):
        observation['name'] = encode_name(observation['name'])
        logger.debug(f"Encoded name: {observation['name']}")

    if observation.get('id') is None:
        observation['id'] = generate_id_from_observation(observation)
        logger.debug(f"Generated ID: {observation['id']}")
    
    columns_ok, error = check_valid_column(observation)
    if not columns_ok:
        response = {'error': error}
        logger.warning(f"Invalid columns in observation: {error}")
        return jsonify(response)

    _id = observation['id']

    # Check if the ID already exists
    existing_prediction = Prediction.get_or_none(Prediction.observation_id == _id)
    if existing_prediction:
        logger.warning(f"Observation ID: \"{_id}\" already exists")
        response = {
            'error': f'Observation ID: "{_id}" already exists',
            'id': _id,
            'outcome': existing_prediction.predicted_outcome
        }
        return jsonify(response)

    obs = preprocess_data(observation)
    
    outcome = pipeline.predict(obs)[0]
    response = {'id': _id, 'outcome': bool(outcome)}
    
    try:
        with DB.atomic():
            Prediction.create(
                observation_id=_id,
                outcome=None,  # Initially set to None
                observation=json.dumps(observation),
                predicted_outcome=bool(outcome)
            )
        logger.info(f"Observation saved: {_id}")
    except IntegrityError:
        error_msg = f'Observation ID: "{_id}" already exists'
        response['error'] = error_msg
        logger.warning(error_msg)
        
    return jsonify(response)

@app.route('/recidivism_result/', methods=['POST'])
def recidivism_result():
    observation = request.get_json()
    _id = observation['id']
    outcome = observation.get('outcome')
    
    try:
        p = Prediction.get(Prediction.observation_id == _id)
        p.outcome = outcome
        p.save()
        
        response = {
            'id': _id,
            'outcome': outcome,
            'predicted_outcome': p.predicted_outcome
        }
        
        logger.info(f"Recidivism result for ID {_id}: {response}")
        return jsonify(response)
    
    except Prediction.DoesNotExist:
        error_msg = f'Observation ID: "{_id}" does not exist'
        logger.warning(error_msg)
        return jsonify({'error': error_msg})

@app.route('/list-db-contents')
def list_db_contents():
    contents = [model_to_dict(obs) for obs in Prediction.select()]
    logger.info(f"Database contents: {contents}")
    return jsonify(contents)

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=5000)
