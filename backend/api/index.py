import os
from time import time

from flask import Blueprint, request, jsonify, render_template
from werkzeug.utils import secure_filename
from werkzeug.datastructures import CombinedMultiDict

from app.models import check_auth, authorize, config
from app.models import MyForm
from api.correct_photo import correct_photo

index_bp = Blueprint('index', __name__)


def allowed_file(filename):
    """Checking the downloaded file for a valid extension"""

    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower(
           ) in config['APP']['ALLOWED_EXTENSIONS']


@index_bp.route('/', methods=['GET', 'POST'])
def index():
    """Index Page"""
    form = MyForm()
    if form.validate_on_submit():
        file = form.file_.data
        filename = secure_filename(file.filename)
        if allowed_file(filename):
            path = os.path.join("app/static/files/" + filename.split('.')[0])

            try:
                # Create target Directory
                os.mkdir(path)
            except FileExistsError as e:
                pass

            file.save(os.path.join(path, filename))
            del file
            path = '/app/backend/app/static/files/' + filename.split('.')[0]

            tic = time()
            copy, pallete = correct_photo(path, filename)
            toc = time()

            if type(copy) != str:
                form.error = str(copy)
            else:
                form.path = '/static/files/{}/'.format(filename.split('.')[0])
                form.success = True
                form.orig = filename
                form.copy = copy
                form.pallete = pallete
                form.time_seconds = '{:.4f}'.format(toc - tic)

    return render_template('index.html', form=form)
