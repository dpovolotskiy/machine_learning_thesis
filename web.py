# -*- coding: utf-8 -*-
import os
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

import flask
from flask import Flask, flash, render_template
from flask_wtf import FlaskForm
from werkzeug.utils import redirect
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired
import wget

from captions_generator import get_predict_web

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret string'


class CaptioningForm(FlaskForm):
    url_image = StringField('URL-адрес изображения:', validators=[DataRequired()])
    download = SubmitField('Download')


class ResultForm(FlaskForm):
    url_image = ""
    caption = ""


@app.route('/', methods=['GET', 'POST'])
def index():
    form = CaptioningForm()
    if form.validate_on_submit():
        flash('URL {}'.format(form.url_image.data))
        flask.session["url"] = form.url_image.data
        wget.download(flask.session["url"], "downloaded_image.jpg")
        flask.session["prediction"] = get_predict_web("downloaded_image.jpg", "my_model_15.h5")
        if os.path.exists("downloaded_image.jpg"):
            os.remove("downloaded_image.jpg")
        return redirect("/result")
    return render_template('caption.html', title='Download', form=form)


@app.route('/result')
def result():
    form = ResultForm()
    form.url_image = flask.session["url"]
    form.caption = flask.session["prediction"]
    return render_template('result.html', title='Result', form=form)


if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=False)
