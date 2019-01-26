from flask import Flask, send_file, jsonify, request
from mnist.cnn import digit_image, mnist_input
from mnist.cnn import Classifier
from base64 import b64encode, b64decode
from StringIO import StringIO
import cStringIO

app = Flask(__name__)

cnn = Classifier()

@app.route("/")
def root():
  return "Hello, Flask!\n"

@app.route("/mnist/<int:id>", methods = ["GET"])
def getMnistClassification(id):
  image = digit_image(id)
  buffer = cStringIO.StringIO()
  image.save(buffer, format = "PNG")

  b64_image = b64encode(buffer.getvalue())
  classification = cnn.classify(mnist_input(id = id))

  response = { 'classification': classification, 'image': b64_image }
  return jsonify(response)

@app.route("/mnist", methods = ["POST"])
def postMnistClassification():
  b64_image = request.get_json()['image'].encode('ascii')
  buffer = StringIO(bytearray(b64decode(b64_image)))

  classification = cnn.classify(mnist_input(buffer = buffer))
  return jsonify({ 'classification': classification })

@app.route("/mnist/image/<int:id>.png", methods = ["GET"])
def getMnistImagePng(id):
  image = digit_image(id)
  io = StringIO()

  image.save(io, 'PNG')
  io.seek(0)

  return send_file(io, mimetype = "image/png")
