import os
import tempfile
from flask import Flask
from flask import render_template, request
from dog_app import dog_human_detector

app = Flask(__name__)

# Index webpage.  It just runs the function `dog_human_detector` to the
# supplied image, and allows the user to select a new image to
# analyze.
upload_folder = os.path.join(os.getcwd(), 'upload_folder')


def allowed_file(filename):
    return '.' in filename and (filename.rsplit('.', 1)[1] in allowed_extensions)


@app.route('/')
@app.route('/index', methods=['POST'])
def index():
    try:
        file = request.files['file']
        file.save(os.path.join(upload_folder, file.filename))
        img_path = os.path.join(upload_folder, file.filename)

        breed = dog_human_detector(img_path)
        print(img_path)
        print(breed)
        return render_template('index.html', result=breed)
    except:
        return render_template('index.html')


def main():
    app.run(host='127.0.0.1', port=8080, debug=False, threaded=False)


if __name__ == '__main__':
    main()
