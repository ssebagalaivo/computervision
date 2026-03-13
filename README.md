# VGG16 Flask Image Classifier

This project packages a pretrained `VGG16` convolutional neural network into a Flask web application. Users upload an image, the server preprocesses it to `224 x 224`, runs ImageNet inference, and renders the top predictions with confidence scores.

## Features

- Pretrained `VGG16` weights loaded lazily on first prediction
- Upload form for `PNG`, `JPG`, `JPEG`, and `WEBP` images
- Confidence bars for the top 5 ImageNet classes
- Deployment-ready `Dockerfile`, `Procfile`, and `gunicorn` entry point

## Project Layout

```text
.
├── app
│   ├── __init__.py
│   ├── config.py
│   ├── model.py
│   ├── routes.py
│   ├── static/css/styles.css
│   └── templates/
├── Dockerfile
├── Procfile
├── requirements.txt
├── runtime.txt
└── wsgi.py
```

## Local Setup

The most important setup detail is the Python version.

- `Python 3.14` will not work for this project because TensorFlow does not publish compatible wheels for it yet.
- Use `Python 3.11` or `Python 3.12`.
- On macOS, install `tensorflow`, not `tensorflow-cpu`.

If your machine only has Python 3.14 right now, install Python 3.12 first, then create a fresh virtual environment.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -m flask --app wsgi run --debug
```

Open `http://127.0.0.1:5000`.

If `python3.12` is not available yet, install Python 3.12 from python.org or Homebrew, then rerun the commands above.

## Docker Run

```bash
docker build -t vgg16-flask-app .
docker run --rm -p 8000:8000 -e PORT=8000 vgg16-flask-app
```

Open `http://127.0.0.1:8000`.

## Deployment Notes

- The `Procfile` supports platforms that launch `gunicorn` directly.
- The `Dockerfile` pre-downloads VGG16 ImageNet weights during image build so the first request is fast in production.
- For a platform like Render, Railway, or Fly.io, deploy from this directory and use the included `Dockerfile`.
- Set `SECRET_KEY` in production instead of relying on the default placeholder value.
- The included [runtime.txt](/Users/praisewebsolutions/computervision/runtime.txt) targets Python `3.11.10` for hosted deployments.

## How Prediction Works

1. Flask receives the uploaded file.
2. Pillow converts the image to RGB and resizes it to `224 x 224`.
3. TensorFlow applies `preprocess_input` for VGG16.
4. The model returns ImageNet probabilities, and the app shows the top 5 classes.
