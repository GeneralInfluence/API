#!/usr/bin/env python
"""API to Data Community DC Data Lake built on Flask"""

from flask import Flask, request, Response, make_response, logging
from werkzeug.exceptions import NotFound, Unauthorized, UnsupportedMediaType, BadRequest
import boto
import os, inspect
import ujson as json
import uuid
import redis
import amqp
import urllib, urllib2
import sys
import traceback
import logging
from logging import FileHandler
import math
import magic
import xlrd
from logging.handlers import RotatingFileHandler
import ast
from datetime import timedelta
from functools import update_wrapper
from flask.ext.cors import CORS
import time


# from twisted.internet import task
# from twisted.internet import reactor


# Install Dependencies:
# sudo apt-get update
# sudo apt-get install python-setuptools python-libxml2
# sudo easy_install pip
# pip install flask boto redis celery filemagic xlrd
# pip install -U flask-cors
# Get the latest ThreeScale python library here: https://github.com/3scale/3scale_ws_api_for_python
# and follow the directions to install

from config import *

# Assumes custom packages are either in the same directory or the parent directory for importing
this_dir = os.path.realpath( os.path.abspath( os.path.split( inspect.getfile( inspect.currentframe() ))[0]))
parent_dir = os.path.realpath(os.path.abspath(os.path.join(this_dir,"../")))
sys.path.insert(0, this_dir)
sys.path.insert(0, parent_dir)

from IdeaNets.models.lstm.scode.lstm_class import LSTM as lstm
from Preprocessing.loadCleanly import sheets as sh

ROOT = "/v1/gizmoduck"
app = Flask(__name__)
app.config.from_object(__name__)
CORS(app)
# CORS(app, resources={ROOT + r"/*": {"origins": "*"}})

HANDLER = FileHandler('./api.log')
HANDLER.setLevel(logging.INFO)
app.logger.addHandler(HANDLER)

if USESSL:
    from OpenSSL import SSL
    ctx = SSL.Context(SSL.SSLv23_METHOD)
    ctx.use_privatekey_file('ssl/ssl.key')
    ctx.use_certificate_file('ssl/ssl.cert')

IDEANETS = {}
# app.logger.info("Loading the model now.")
# model = "untrustworthy"
# model_path = os.path.realpath(os.path.abspath(os.path.join(this_dir,MODEL_MAPS[model])))
# IDEANETS[model] = lstm() # I can see the type of model to use being an input at some point.
# IDEANETS[model].load_pickle(model_path)

MAX_MEGABYTES = 2
app.config['MAX_CONTENT_LENGTH'] = MAX_MEGABYTES * 1024 * 1024

ACCEPTED_MIMETYPES = ["text/plain",
                      "text/csv",
                      "application/vnd.ms-excel",
                      "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]

ERROR_MESSAGES = {404: "The requested resource cannot be found. Please check the API documentation and try again.",
                  401: "You are not authenticated to use this resource. Please provide a valid app_id and app_key pair.",
                  413: "The file that was submitted is too large. Please submit a file smaller than %i megabytes." % MAX_MEGABYTES,
                  415: "The file that was submitted is an unsupported type. Please submit valid plain text, CSV, or an Excel spreadsheet.",
                  500: "An error has occurred in the analysis. Please contact support@gosynapsify.com for assistance.",
                  400: "The request is missing required parameters or otherwise malformed. Please check the API documentation and try again."
                  }


# Connections
RED = redis.Redis(host=REDISHOST, port=REDISPORT, db=0)
# S3 = boto.connect_s3(aws_access_key_id=AWSKEY, aws_secret_access_key=AWSSECRET).get_bucket(BUCKET)

# datagramRecieved = False
# timeout = 1.0 # One second
# def testTimeout():
#     global datagramRecieved
#     if not datagramRecieved:
#         reactor.stop()
#     datagramRecieved = False

# import socket, errno, time
# def maintain_socket():
#     # setup socket to listen for incoming connections
#     s = socket.socket()
#     # 127.0.0.1
#     # http://0.0.0.0:9999
#     # s.bind(('localhost:9999', 1234))
#     s.bind(('localhost', 1234))
#     s.listen(1)
#     remote, address = s.accept()
#
#     print "Got connection from: ", address
#
#     while 1:
#         try:
#             remote.send("message to peer\n")
#             time.sleep(1)
#         except socket.error, e:
#             if isinstance(e.args, tuple):
#                 print "errno is %d" % e[0]
#                 if e[0] == errno.EPIPE:
#                    # remote peer disconnected
#                    print "Detected remote disconnect"
#                 else:
#                    # determine and handle different error
#                    pass
#             else:
#                 print "socket error ", e
#             remote.close()
#             break
#         except IOError, e:
#             # Hmmm, Can IOError actually be raised by the socket module?
#             print "Got IOError: ", e
#             break


def make_key(app_id, app_key, task_id):
    """This is a naming convention for both redis and s3"""
    return app_id + "/" + app_key + "/" + task_id

def check_credentials():
    """Check the credentials with 3scale. Return app_id and app_key pair if valid. 
       Raise Unauthorized error if not."""
    # request is a global variable from flask and contains all info about the request from the client.

    app.logger.info("Checking Credentials.")
    app_id  = str(request.args.get("app_id"))
    app.logger.info("app_id loaded")
    app_key = str(request.args.get("app_key"))
    app.logger.info("app_key loaded")
    app.logger.info("Checking " + app_id + " against " + str(APP_IDS))
    if app_id == None or app_key == None:
        app.logger.info("They have to give us an id and key to work with!")
        raise Unauthorized()
    if (app_id in APP_IDS) & (app_key in APP_KEYS[app_id]):
        app.logger.info("Are they trying to hack us?")
        return app_id, app_key
    raise Unauthorized()

def check_credentials_and_status(task_id):
    """Check the credentials and status in one go for use with retrieval functions. 
       Raise NotFound error if status isn't complete."""
    app_id, app_key = check_credentials()
    status = get_status(app_id, app_key, task_id)
    if status["documentStatus"] != "COMPLETE":
        app.logger.error("Status failed in check_credentials_and_status")
        raise NotFound()
    return app_id, app_key, status

def validate_parameters(params, expected_params):
    """Validate a list of parameters. Return BadRequest error if invalid."""
    for key, ptype in expected_params.items():
        try:
            ptype(params[key])
        except KeyError:
            raise BadRequest()

def validate_input_file(handle, mimetype):
    """Validate that an input file is a known text encoding if text or a readable Excel spreadsheet. 
       Raise UnsupportedMediaType error if not."""
    handle.seek(0)
    if mimetype in ("text/plain", "text/csv"):
        buf = handle.read(512)
        if len(buf) > 0:
            merlin = magic.Magic(mime_encoding=True)
            encoding = merlin.from_buffer(buf)
            if encoding == "binary":
                raise UnsupportedMediaType()
    else:
        try:
            xlrd.open_workbook(file_contents=handle.read())
        except AssertionError: # This is the error created if we can't open the spreadsheet
            raise UnsupportedMediaType()
    handle.seek(0)

def make_custom_response(data, code, headers=None):
    """Create a complete JSON response from an object using the Flask Response type"""
    response = Response(json.dumps(data)+"\n", status=code, mimetype='application/json')
    response.headers["Server"] = "General Influence API"
    if headers != None:
        for key, value in headers.items():
            response.headers[key] = value
    return response

# def crossdomain(origin=None, methods=None, headers=None,
#                 max_age=21600, attach_to_all=True,
#                 automatic_options=True):
#     if methods is not None:
#         methods = ', '.join(sorted(x.upper() for x in methods))
#     if headers is not None and not isinstance(headers, basestring):
#         headers = ', '.join(x.upper() for x in headers)
#     if not isinstance(origin, basestring):
#         origin = ', '.join(origin)
#     if isinstance(max_age, timedelta):
#         max_age = max_age.total_seconds()
#
#     def get_methods():
#         if methods is not None:
#             return methods
#
#         options_resp = current_app.make_default_options_response()
#         return options_resp.headers['allow']
#
#     def decorator(f):
#         def wrapped_function(*args, **kwargs):
#             if automatic_options and request.method == 'OPTIONS':
#                 resp = current_app.make_default_options_response()
#             else:
#                 resp = make_response(f(*args, **kwargs))
#             if not attach_to_all and request.method != 'OPTIONS':
#                 return resp
#
#             h = resp.headers
#
#             h['Access-Control-Allow-Origin'] = origin
#             h['Access-Control-Allow-Methods'] = get_methods()
#             h['Access-Control-Max-Age'] = str(max_age)
#             if headers is not None:
#                 h['Access-Control-Allow-Headers'] = headers
#             return resp
#
#         f.provide_automatic_options = False
#         return update_wrapper(wrapped_function, f)
#     return decorator

@app.errorhandler(401)
@app.errorhandler(404)
@app.errorhandler(413)
@app.errorhandler(415)
@app.errorhandler(400)
def handle_user_error(err):
    """Handle all 400 level errors. Don't send a support email since this is the user's problem."""
    message = {"status" : "fail",
        "data" : {"reason" : ERROR_MESSAGES[err.code]}}
    app.logger.warning("A user error occurred. code: %i, path: %s", err.code, request.path)
    # return make_custom_response(message, err.code)
    return make_response(str(message))

@app.errorhandler(500)
def handle_internal_error(err):
    """Handle all 500 level errors. Send a detailed email about the error. Respond with a helpful message."""
    message = {"status" : "fail",
        "data" : {"reason" : ERROR_MESSAGES[500]}}
    exc_type, exc_value, exc_traceback = sys.exc_info()
    exc = traceback.format_exception(exc_type, 
                                     exc_value, 
                                     exc_traceback)
    exc = ''.join(exc)
    error_message = {
           "api_user":"",
           "api_key":"",
           "to": SUPPORTEMAIL,
           "from": SUPPORTEMAIL,
           "subject": "General Influence API Error",
           "text": "An error ocurred in the General Influence IdeaNet API: \n\nENDPOINT: " \
                + request.path + " " + request.method + "\n\n" + exc +
                "\n\n" + "app_id:" + str(request.args.get("app_id")) +
                "\n\n" + "app_key:" + str(request.args.get("app_key")) }
    data = urllib.urlencode(error_message)
    # urllib2.urlopen(url="https://api.sendgrid.com/api/mail.send.json", data=data).read()  # NEED A NEW EMAIL SOLUTION
    # smtp_handler = logging.handlers.SMTPHandler(mailhost=("smtp.gmail.com", 465),
    #                                             fromaddr="sean@general-influence.com",
    #                                             toaddrs="sean.moore.gonzalez@gmail.com",
    #                                             subject="General Influence API Error")
    #
    #
    # logger = logging.getLogger()
    # logger.addHandler(smtp_handler)
    app.logger.error("ERROR: " + error_message["text"])      
    # return make_custom_response(message, 500)
    return make_response(str(message))

def post_initial_status(key):
    """Post the first status message to Redis. Redis messages are used by the queue endpoint."""
    key = "jobs/" + key
    message = {"percentComplete": 0,
               "taskStatusUpdate": "Initializing...",
               "documentStatus": "QUEUED"}
    value = json.dumps(message)
    RED.set(key, value)
    # Publish to a channel so that clients such as the webapp can subscribe instead of poll.
    RED.publish(key, value)

def get_status(app_id, app_key, task_id):
    """Get the status from Redis as a dictionary."""
    key = "jobs/" + make_key(app_id, app_key, task_id)
    return json.loads(RED.get(key))

def submit_job(app_id, app_key, task_id, mimetype, text_col, dedupe):
    """Submit a job to the queue for the Celery worker. Create the required JSON message and post it to RabbitMQ."""
    # These are the args that the Python function in the adjunct processor will use.
    kwargs = {"app_id": app_id,
              "app_key": app_key,
              "task_id": task_id,
              "format": mimetype,
              "text_col": text_col,
              "dedupe": dedupe,
              "s3_endpoint": S3ENDPOINT,
              "bucket": BUCKET,
              "redis_port": REDISPORT,
              "redis_host": REDISHOST}
    # Recreate a celery message manually so that we don't need to import celery_tasks.py which has heavy dependencies. 
    job = {"id": task_id,
           # "task": "synapsify_adjunct.celery_tasks.synapsify_master",
           "task": "dc2_master",
           "kwargs": kwargs}
    # Connect to RabbitMQ and post.
    conn = amqp.Connection(host=RMQHOST, port=RMQPORT, userid=RMQUSERNAME, password=RMQPASSWORD, virtual_host=RMQVHOST, insist=False)
    cha = conn.channel()
    msg = amqp.Message(json.dumps(job))
    msg.properties["content_type"] = "application/json"
    cha.basic_publish(routing_key=RMQEXCHANGE,
                        msg=msg)
    cha.close()
    conn.close()

def get_models(models):
    """
    The models would be stored in S3, or perhaps for now just on the server in a file.
    Eventually they might be in a database, but either way I need to reduce their size because they take a while to load.
    That being said, I really should have a login capability where we use the user information to gather and load the necessary models.
    I could have those models loaded already using celery
    :return:
    """
    models_loaded = []
    for model in models:
        if (model in MODEL_MAPS):
            models_loaded.append(model)
            if (model not in IDEANETS):
                model_path = os.path.realpath(os.path.abspath(os.path.join(this_dir,MODEL_MAPS[model])))
                IDEANETS[model] = lstm() # I can see the type of model to use being an input at some point.
                IDEANETS[model].load_pickle(model_path)
                app.logger.info("Model(s) loaded: " + str(model))

    return models_loaded

@app.route(ROOT + '/health_check/')
def health_check():
    """
    To verify that errors are with the endpoints, not the server configuration
    :return:
    """
    return "<h1 style='color:red'>Server is working!</h1>"

# The @app.route function decorators map endpoints to functions.
# @crossdomain(origin='*')
# @cross_origin()
@app.route(ROOT + '/initialize/', methods=['POST', 'GET'])
def initialize_session():
    """
    Need to get things loaded so the user response is quick!

    The POST method gives the user information for model retrieval
    The GET method gives the status of loading the models
    :return:
    """

    # l = task.LoopingCall(testTimeout)
    # l.start(timeout) # call every second
    #
    # reactor.run()

    if request.method == 'POST':
        # maintain_socket()
        app.logger.info("Got a file POST request") # Log that we got a request
        app_id, app_key = check_credentials() # Extract and validate credentials.
        params = ast.literal_eval(request.data)
        models = params['models']
        models_loaded = get_models(models)
        data = {"status":"success",
                "models_loaded":models_loaded}

    if request.method == 'GET':
        print "This is a placeholder for getting model loading status"
        data = {"status":"success",
                "models_loaded":[]}

    # l.stop() # will stop the looping calls
    app.logger.info(request.data)
    # return make_custom_response(data,200)
    return make_response(str(data))

# @crossdomain(origin='*')
# @cross_origin()
@app.route(ROOT + '/classify_one/', methods=['POST', 'GET'])
def classify_json():
    """
    The POST method receives a json and calls the classifier
    The GET method simply returns the mdoels available to this user.

    :param request.data: dictionary with fields 'sentences' and 'model',
        which indicates which ideanet model to use on the given sentences.
    :return:
    """
    app.logger.info("Testing testing 123")
    if request.method == 'GET':
        ### FORMALITIES FORMALITIES FORMALITIES FORMALITIES
        app.logger.info("Got a file GET request") # Log that we got a request
        app.logger.info(request.data)

        app_id, app_key = check_credentials() # Extract and validate credentials.
        app.logger.info("Credentials accepted.")

        data = ast.literal_eval(request.data)
        model_initialized = True
        try:
            sentences = data['sentences']
            app.logger.info("Classifying sentence(s): " + sentences)
            models = data['model']
            predictions = []
            probabilities = []
            for model in models:
                if (model in MODEL_MAPS):
                    app.logger.info(model)
                    if model not in IDEANETS:
                        app.logger.info("Initializing IDEANET.")
                        model_initialized = False
                        models_loaded = get_models(models)
                        if model in models_loaded:
                            app.logger.info("IDEANET " +model+ " successfully initialized!")
                            model_initialized = True
                    if model_initialized:
                        ideanet = IDEANETS[model]
                        this_pred = ideanet.classify(sentences)
                        probabilities += this_pred['prob_0_1']
                        predictions.append(this_pred['pred'])
                else:
                    app.logger.info("Model not in MODEL_MAPS: " + model)
            data = {"status":"success",
                    "predictions":predictions,
                    "probabilities":probabilities}

        except:
            data = {"status":"failure",
                    "predictions":[],
                    "probabilities":[]}

    # if request.method == 'GET':
        # This is a placeholder for a more informative endpoint.
    app.logger.info("data: " + str(data))
    # time.sleep(5)
    # return make_custom_response(data, 200)
    return make_response(str(data))

# The @app.route function decorators map endpoints to functions.
@app.route(ROOT + '/mass', methods=['POST', 'GET'])
def classify_files():
    """
    The POST method allows API clients to use their model(s)
    The GET method allows API clients to get a list of available models.

    :return:
    """
    if request.method == "POST":
        # Log that we got a request
        app.logger.error("Got a file POST request")
        # Extract and validate credentials.
        app_id, app_key = check_credentials()

        try:
            submitted_file = request.files.get('file')
            ctype = submitted_file.content_type
            if ctype not in ACCEPTED_MIMETYPES:
                app.logger.error("Unsupported Media Type: %s", ctype)
                raise UnsupportedMediaType

            data = json.loads(request.data)
            if 'textcol' in data:
                textcol = data['textcol']
            validate_input_file(submitted_file, ctype)
            header, rows = sh.get_spreadsheet_rows(submitted_file,textcol,dedupe=True)

            model = data['model']
            ideanet = IDEANETS[model]
            predictions = ideanet.classify(rows)
            data = {"status":"success",
                    "data":predictions}

        except:
            data = {"status":"failure",
                    "data":{}}


    # elif request.method == "GET":

@app.route(ROOT + '/train', methods=['POST', 'GET'])
def train_model():
    """
    The POST method is where API clients provide data to train their model
    The GET method provides training status (statii?) and details for calling their model when complete."""
    if request.method == "POST":
        # Log that we got a request
        app.logger.error("Got a file POST request")
        # Extract and validate credentials. 
        app_id, app_key = check_credentials()

        # Get the file, validate the type, and make sure the file itself is valid.
        # The EntityTooLarge error is raised automatically by Flask.
        submitted_file = request.files.get('file')
        size = len(submitted_file.read())
        ctype = submitted_file.content_type
        if ctype not in ACCEPTED_MIMETYPES:
            app.logger.error("Unsupported Media Type: %s", ctype)
            raise UnsupportedMediaType
        validate_input_file(submitted_file, ctype)

        # Validate the parameters and set a default.
        if ctype != "text/plain":
            validate_parameters(request.form, {"text_col": int})
            text_col = request.form.get("text_col")
        else:
            text_col = 0
        if request.form.get("dedupe"):
            dedupe = request.form.get("dedupe")
        else:
            dedupe = False

        # If we've reached this point, everything looks good so generate a task id.
        task_id = str(uuid.uuid4())

        # Post initial status to Redis, upload to s3, and submit the job to RabbitMQ.
        key = make_key(app_id, app_key, task_id)
        post_initial_status(key)
        S3.new_key("input/"+key).set_contents_from_file(submitted_file)
        submit_job(app_id, app_key, task_id, ctype, text_col, dedupe)

        # Finally, return a message to the client and write to the log file.
        data = {"status": "success",
                "data": {"job_id": task_id,
                         "file_size": size,
                         "mime_type": ctype,
                         "links": [{"rel": "queue",
                                    "href": ROOT + "/queue/" + task_id,
                                    "type": "application/json"}]}}

        app.logger.error("File successfully submitted. type: %s, size: %i, app_id: %s, task_id: %s, dedupe: %s", ctype, size, app_id, task_id, dedupe)
        # return make_custom_response(data, 202, headers = {"Location": ROOT + "/queue/" + task_id})
        return make_response(str(data))

    if request.method == "GET":
        # Extract and validate credentials. 
        app_id, app_key = check_credentials()

        # Get the list of previous task ids from s3.
        outputs = set([key.name.split("/")[-2] for key in S3.list(prefix="output/" + app_id + "/" + app_key)])

        # optionally paginate results
        if request.args.get("max_results"):
            per_page = int(request.args.get("max_results"))
            offset = 0
            if request.args.get("offset"): offset = int(request.args.get("offset"))
            outputs = list(outputs)[:offset + per_page]
        if len(outputs) == 0:
            raise NotFound() # should this be a 404?

        # Return the list in a JSON response.    
        data = {"status": "success",
                "data": {}}
        data["data"]["links"] = []
        for task_id in outputs:
            data["data"]["links"].append({"rel": task_id, 
                                          "href": ROOT + "/" + task_id,
                                          "type" : "application/json"})

        # return make_custom_response(data, 200)
        return make_response(str(data))


@app.route(ROOT + '/queue/<string:task_id>', methods=['GET'])
def queue(task_id):
    """This endpoint is where clients poll to find the status of their jobs."""
    # Extract and validate credentials. 
    app_id, app_key = check_credentials()

    # Get status from Redis.
    status = get_status(app_id, app_key, task_id)

    # Build the response.
    data = {"status": "success",
            "data": status}
    data["job_id"] = task_id
    headers = {}
    code = 200 # default
    # If the job is complete, give a link to the listing endpoint.
    if status["documentStatus"] == "COMPLETE":
        headers["Location"] = ROOT + "/" + task_id
        data["data"]["links"] = [{"rel": task_id,
                                 "href": ROOT + "/" + task_id,
                                 "type": "application/json"}]

    # If the adjunct had an error, log it and return a 500 status.
    elif status["documentStatus"] == "FAIL":
        app.logger.info("Looks like there was a DC2 Master error:\n%s", status)
        data["sad_face"] = ":_("
        try:
            data["taskStatusUpdate"] += " Please contact support@gosynapsify.com."
        except KeyError:
            app.logger.error("Key ERROR! No taskStatusUpdate")
            data["taskStatusUpdate"] = " Please contact support@gosynapsify.com."
        data["error"] = data["taskStatusUpdate"]
        app.logger.warning("An error ocurred in the adjunct: %s", status["error"])
        code = 500

    # If the job is queued or processing, give a link to this same endpoint
    else: 
        headers["Location"] = ROOT + "/queue/" + task_id
        data["data"]["links"] = [{"rel": "queue",
                                 "href": ROOT + "/queue/" + task_id,
                                 "type": "application/json"}]

    app.logger.info("Polling task_id: %s, status: %s, percentComplete: %i", task_id, status["documentStatus"], status["percentComplete"])
    # return make_custom_response(data, code, headers)
    return make_response(str(data))

@app.route(ROOT + '/<string:task_id>', methods=['GET'])
def list_results(task_id):
    """This endpoint lists the locations of the results."""
    # Extract and validate credentials. Verify that the job is done.
    app_id, app_key, status = check_credentials_and_status(task_id)

    # Get the list of outputs for this task id
    # outputs = [key for key in S3.list(prefix="output/" + make_key(app_id, app_key, task_id)) if key.content_type == "text/csv"]
    outputs = [key for key in S3.list(prefix="output/" + make_key(app_id, app_key, task_id)) if key.name[-3:] == "csv"]
    # outputs = []
    # [outputs.append(key) for key in S3.list(prefix="output/" + make_key(app_id, app_key, task_id))]

    if len(outputs) == 0:
        app.logger.error("From list_results, document was complete but cannot find files!")
        raise NotFound()    

    # Build the JSON response with the locations of the results.
    data = {"status": "success",
            "data": {} }
    data["data"]["links"] = []
    for key in outputs:
        name = key.name.split("/")[-1]
        data["data"]["links"].append({"rel": name, 
                                      "href": ROOT + "/" + task_id + "/" + name,
                                      "type" : "text/csv",
                                      "size" : key.size,
                                      "completion_date": key.last_modified}) #,
                                      # For some reason expiry_date doesn't work so it's gone for now
                                      #"expiry_date": key.expiry_date})
    
    # return make_custom_response(data, 200)
    return make_response(str(data))

def get_file(task_id, name):
    """Helper function for spitting out a file on s3"""
    # Validate credentials and check status
    app_id, app_key, status = check_credentials_and_status(task_id)
    key = "output/" + make_key(app_id, app_key, task_id) + "/" + name
    app.logger.info("Results delivered. name: %s, app_id: %s, task_id: %s", name, app_id, task_id)
    # Return results directly from s3
    return S3.get_key(key).get_contents_as_string(), 200

@app.route(ROOT + '/<string:task_id>/comments.csv', methods=['GET'])
def comments(task_id):
    """Return the comments.csv results"""
    return get_file(task_id, "comments.csv")

@app.route(ROOT + '/<string:task_id>/graph.json', methods=['GET'])
def graph(task_id):
    """Return the graph.json results"""
    return get_file(task_id, "graph.json")

@app.route('/')
def info():
    """Return some info, useful for testing deployment"""
    return "This is the Data Community DC Data Lake API. Please see the documentation for use."


if __name__ == '__main__':
    #handler = RotatingFileHandler(LOGFILE, maxBytes=1024*1024, backupCount=10)
    handler = FileHandler(LOGFILE)
    handler.setLevel(logging.INFO)
    app.logger.addHandler(handler)
    if USESSL:
        #app.run(host='0.0.0.0', port=FLASKPORT, ssl_context=ctx)
        app.run(host='0.0.0.0', port=FLASKPORT, ssl_context='adhoc', threaded=True)
    else:
        # app.run()
        app.run(host='0.0.0.0', port=FLASKPORT, debug=DEBUG, threaded=True)


