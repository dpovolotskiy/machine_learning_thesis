import flask
from flask import Flask, request
from flask_restful import Resource, Api
import os
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

from captions_generator import get_predict_web



app = Flask(__name__)
api = Api(app)


class Prediction(Resource):
    def post(self):
        with open("current_processing_image.jpg", "wb") as image:
            image.write(request.data)
        caption = get_predict_web("current_processing_image.jpg")
        if os.path.exists("current_processing_image.jpg"):
            os.remove("current_processing_image.jpg")
        return flask.jsonify({'caption': caption})


api.add_resource(Prediction, '/prediction')


if __name__ == '__main__':
     app.run(threaded=False)