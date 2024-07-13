import os
import json
import pickle
import gzip
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

def load_model(file_path):
    with gzip.open(file_path, 'rb') as f_in:
        model = pickle.load(f_in)
    if not hasattr(model, 'predict'):
        raise ValueError("Loaded object is not a valid model or pipeline with predict method")
    return model

# Use the load_model function to load the compressed model
pipeline = load_model('pipeline.pickle.gz')

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


# Define the frequency encoding function
def frequency_encoding(df, column):
    freq_encoding = df[column].value_counts(normalize=True)
    df[f'{column}_freq'] = df[column].map(freq_encoding)
    return df

def validate_observation(observation):
    errors = []

    # Null/None Checks
    if observation.get('c_jail_in') is None:
        errors.append("c_jail_in cannot be None")
    if all(observation.get(field) is None for field in ['dob', 'priors_count', 'sex', 'race']):
        errors.append("At least one of 'dob', 'priors_count', 'sex', or 'race' must not be None")

    # Value Range Checks
    if 'priors_count' in observation and observation['priors_count'] is not None and observation['priors_count'] < 0:
        errors.append("priors_count must be non-negative")

    # Date Validity Checks
    for date_field in ['dob', 'c_offense_date', 'c_arrest_date', 'c_jail_in']:
        if observation.get(date_field) is not None:
            try:
                pd.to_datetime(observation[date_field], errors='raise')
            except (ValueError, TypeError):
                errors.append(f"{date_field} must be a valid date")

    if 'c_jail_in' in observation and 'c_offense_date' in observation and observation.get('c_jail_in') and observation.get('c_offense_date'):
        if pd.to_datetime(observation['c_jail_in']) < pd.to_datetime(observation['c_offense_date']):
            errors.append("c_jail_in cannot be before c_offense_date")

    # Category Value Checks
    if 'sex' in observation and observation['sex'] not in [None, 'Male', 'Female']:
        errors.append("sex must be 'Male' or 'Female'")

    if 'race' in observation:
        valid_races = {'Asian', 'Native American', 'Hispanic', 'Caucasian', 'African-American', 'Other'}
        if observation['race'] not in valid_races:
            errors.append("race must be valid")

    return errors

def preprocess_data(df):
    
    df = pd.DataFrame([df])

    
    df['dob'] = pd.to_datetime(df['dob'], errors='coerce')
    df['c_offense_date'] = pd.to_datetime(df['c_offense_date'], errors='coerce')
    df['c_arrest_date'] = pd.to_datetime(df['c_arrest_date'], errors='coerce')
    df['c_jail_in'] = pd.to_datetime(df['c_jail_in'], errors='coerce')

    df['unified_date'] = df['c_arrest_date'].combine_first(df['c_offense_date'])

    # Feature engineering
    df['age_jail'] = (df['c_jail_in'] - df['dob']).dt.days // 365
    df['age_arrest'] = (df['unified_date'] - df['dob']).dt.days // 365
    df['total_juv_crimes'] = df['juv_fel_count'] + df['juv_misd_count'] + df['juv_other_count']
    df['total_adult_crimes'] = df['priors_count'] - df['total_juv_crimes']

    # Frequency encoding for 'c_charge_desc'
    df = frequency_encoding(df, 'c_charge_desc')

    # Create age bins
    age_bins = [16, 25, 35, 45, 96]
    age_labels = ['16-24', '25-34', '35-44', '45+']
    df['age_group'] = pd.cut(df['age_jail'], bins=age_bins, labels=age_labels, right=False)

    # Create bins for priors_count
    priors_bins = [0, 1, 2, 3, 5, float('inf')]
    priors_labels = ['0', '1', '2', '3-4', '5+']
    df['priors_count_bin'] = pd.cut(df['priors_count'], bins=priors_bins, labels=priors_labels, right=False)

    # Extract more granular date features
    df['offense_month'] = df['c_offense_date'].dt.month
    df['offense_day_of_week'] = df['c_offense_date'].dt.dayofweek
    df['arrest_month'] = df['c_arrest_date'].dt.month
    df['arrest_day_of_week'] = df['c_arrest_date'].dt.dayofweek

    df['time_to_jail'] = (df['c_jail_in'] - df['c_offense_date']).dt.days
    df['race_priors_interaction'] = df['race'].astype(str) + '_' + df['priors_count_bin'].astype(str)
    df['jail_month'] = df['c_jail_in'].dt.month
    df['jail_year'] = df['c_jail_in'].dt.year

    # Replace specific races with 'Other'
    df['race'] = df['race'].replace({
        'Asian': 'Other',
        'Native American': 'Other',
        'Hispanic': 'Other'
    })

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
    
    validation_errors = validate_observation(observation)
    if validation_errors:
        response = {'error': validation_errors}
        logger.warning(f"Validation errors in observation: {validation_errors}")
        return jsonify(response), 422

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
        return jsonify(response), 409

    obs = preprocess_data(observation)
    obs = obs[columns]
    
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
        return jsonify(response), 409
        
    return jsonify(response), 201

@app.route('/recidivism_result/', methods=['POST'])
def recidivism_result():
    observation = request.get_json()
    logger.info(f"Received observation: {observation}")
    _id = observation['id']
    outcome = observation.get('outcome')
    
    try:
        p = Prediction.get(Prediction.observation_id == _id)
        p.outcome = outcome
        p.save()
        
        obs_dict = json.loads(p.observation)
        predicted_outcome = p.predicted_outcome
        
        response = {
            'id': _id,
            'outcome': outcome,
            'predicted_outcome': p.predicted_outcome
        }
        
        logger.info(f"Recidivism result for ID {_id}: {response}")
        return jsonify(response), 200
    
    except Prediction.DoesNotExist:
        error_msg = f'Observation ID: "{_id}" does not exist'
        logger.warning(error_msg)
        return jsonify({'error': error_msg}), 404

@app.route('/list-db-contents')
def list_db_contents():
    contents = [model_to_dict(obs) for obs in Prediction.select()]
    logger.info(f"Database contents: {contents}")
    return jsonify(contents)

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=5000)
