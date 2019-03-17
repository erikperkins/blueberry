from flask import Flask, jsonify, request, Response
from mnist.cnn import digit_image, mnist_input
from mnist.cnn import Classifier
from base64 import b64encode, b64decode
from io import BytesIO
from werkzeug.wsgi import FileWrapper

app = Flask(__name__)

cnn = Classifier()

@app.route("/")
def root():
  return "Hello, Flask!\n"

@app.route("/mnist/<int:id>", methods = ["GET"])
def getMnistClassification(id):
  image = digit_image(id)
  bytes = BytesIO()
  image.save(bytes, format = "PNG")
  b64_image = b64encode(bytes.getvalue()).decode()

  classification = cnn.classify(mnist_input(id = id))
  response = { 'classification': classification, 'image': b64_image }
  return jsonify(response)

@app.route("/mnist", methods = ["POST"])
def postMnistClassification():
  b64_image = request.get_json()['image'].encode('ascii')
  buffer = BytesIO(bytearray(b64decode(b64_image)))

  classification = cnn.classify(mnist_input(buffer = buffer))
  return jsonify({ 'classification': classification })

@app.route("/mnist/image/<int:id>.png", methods = ["GET"])
def getMnistImagePng(id):
  image = digit_image(id)
  bytes = BytesIO()

  image.save(bytes, 'PNG')
  bytes.seek(0)
  wrapper = FileWrapper(bytes)

  return Response(wrapper, mimetype = 'image/png', direct_passthrough = True)
