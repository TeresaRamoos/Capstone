import os
import json
import joblib
import pandas as pd
from flask import Flask, jsonify, request
from peewee import (
    Model, BooleanField, CharField,
    TextField
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

########################################
# Begin database stuff
DB = connect(os.environ.get('DATABASE_URL') or 'sqlite:///predictions.db')

class Prediction(Model):
    observation_id = CharField(primary_key=True, max_length=50)
    observation = TextField()
    pred_class = BooleanField()
    true_class = BooleanField(null=True)

    class Meta:
        database = DB


DB.create_tables([Prediction], safe=True)

# End database stuff
########################################


########################################
# model

with open('columns.json') as fh:
    columns = json.load(fh)

pipeline = joblib.load('pipeline.pickle')

# End model
########################################

########################################
# Input validation functions


def check_request(request):
    """
        Validates that our request is well formatted
        
        Returns:
        - assertion value: True if request is ok, False otherwise
        - error message: empty if request is ok, False otherwise
    """
    
    if "id" not in request:
        error = "Field `id` missing from request: {}".format(request)
        return False, error
    
    if "observation" not in request:
        error = "Field `observation` missing from request: {}".format(request)
        return False, error
    
    return True, ""



def check_valid_column(observation):
    """
        Validates that our observation only has valid columns
        
        Returns:
        - assertion value: True if all provided columns are valid, False otherwise
        - error message: empty if all provided columns are valid, False otherwise
    """
    
    valid_columns = {
      "name",
      "sex",
      "dob", 
      "race", 
      "juv_fel_count", 
      "juv_misd_count",
      "juv_other_count",
      "priors_count",
      "c_case_number",
      "c_charge_degree",
      "c_charge_desc",
      "c_offense_date",
      "c_arrest_date",
      "c_jail_in",
      "is_recid",
      "r_case_number",
      "r_charge_degree",
      "r_charge_desc",
      "r_offense_date",
      "is_violent_recid",
      "vr_case_number",
      "vr_offense_date",
      "vr_charge_degree",
      "vr_charge_desc"
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



def check_categorical_values(observation):
    """
        Validates that all categorical fields are in the observation and values are valid
        
        Returns:
        - assertion value: True if all provided categorical columns contain valid values, 
                           False otherwise
        - error message: empty if all provided columns are valid, False otherwise
    """
    
    valid_category_map = {
        #"InterventionReasonCode": ["V", "E", "I"],
        #"SubjectRaceCode": ["W", "B", "A", "I"],
        #"SubjectSexCode": ["M", "F"],
        #"SubjectEthnicityCode": ["H", "M", "N"],
        #"SearchAuthorizationCode": ["O", "I", "C", "N"],
        #"ResidentIndicator": [True, False],
        "Race": ['African-American','Caucasian', 'Hispanic', 'Other', 'Asian', 'Native American'], 
        #"day_of_week": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    }
    
    for key, valid_categories in valid_category_map.items():
        if key in observation:
            value = observation[key]
            if value not in valid_categories:
                error = "Invalid value provided for {}: {}. Allowed values are: {}".format(
                    key, value, ",".join(["'{}'".format(v) for v in valid_categories]))
                return False, error
        else:
            error = "Categorical field {} missing"
            return False, error

    return True, ""



# End input validation functions
########################################

########################################
# Begin webserver stuff
app = Flask(__name__)


def process_observation(observation):
    logger.info("Processing observation, %s", observation)
    # A lot of processing
    return observation


@app.route('/predict', methods=['POST'])
def predict():
    obs_dict = request.get_json()
    logger.info('Observation: %s', obs_dict)
    _id = obs_dict['observation_id']
    observation = obs_dict

    if not _id:
        logger.warning('Returning error: no id provided')
        return jsonify({'error': 'observation_id is required'}), 400
    if Prediction.select().where(Prediction.observation_id == _id).exists():
        prediction = Prediction.get(Prediction.observation_id == _id)

        # Update the prediction
        prediction.observation = str(observation)
        prediction.save()

        logger.warning('Returning error: already exists id %s', _id)
        return jsonify({'error': 'observation_id already exists'}), 400

    try:
        obs = pd.DataFrame([observation], columns=columns)
    except ValueError as e:
        logger.error('Returning error: %s', str(e), exc_info=True)
        default_response = {'observation_id': _id, 'label': False}
        return jsonify(default_response), 200
    
    label = bool(pipeline.predict(obs))
    response = {'observation_id': _id, 'label': label}
    p = Prediction(
        observation_id=_id,
        observation=request.data,
        pred_class=label,
    )
    p.save()
    logger.info('Saved: %s', model_to_dict(p))
    logger.info('Prediction: %s', response)

    return jsonify(response)


@app.route('/update', methods=['POST'])
def update():
    obs = request.get_json()
    logger.info('Observation:', obs)
    _id = obs['observation_id']
    label = obs['label']

    if not _id:
        logger.warning('Returning error: no id provided')
        return jsonify({'error': 'observation_id is required'}), 400
    if not Prediction.select().where(Prediction.observation_id == _id).exists():
        logger.warning(f'Returning error: id {_id} does not exist in the database')
        return jsonify({'error': 'observation_id does not exist'}), 400
    
    p = Prediction.get(Prediction.observation_id == _id)
    p.true_class = label
    p.save()
    logger.info('Updated: %s', model_to_dict(p))

    response = {'observation_id': _id, 'label': label}
    return jsonify(response)



@app.route('/list-db-contents')
def list_db_contents():
    return jsonify([
        model_to_dict(obs) for obs in Prediction.select()
    ])

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=8000)
