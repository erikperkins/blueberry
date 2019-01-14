from flask import Flask, Response, send_file, jsonify, request, make_response
from network.mnist import MnistNetwork, digit_image, mnist_input
from base64 import b64encode, b64decode
from messaging.tasks import make_celery
from StringIO import StringIO
from time import sleep
import cStringIO


app = Flask(__name__)

app.config.update(
  CELERY_BROKER_URL = 'amqp://guest:guest@rabbitmq:5672',
  CELERY_RESULT_BACKEND = 'rpc'
)

@app.route("/")
def root():
  return "Hello, Flask!\n"


@app.route("/mnist/<int:id>.json", methods = ["GET"])
def getMnistClassificationJson(id):
  image = digit_image(id)
  buffer = cStringIO.StringIO()
  image.save(buffer, format = "PNG")

  b64_image = b64encode(buffer.getvalue())
  classification = MnistNetwork().classify(mnist_input(id = id))
  response = { 'classification': classification, 'image': b64_image }
  return jsonify(response)


@app.route("/mnist/new/classification.json", methods = ["POST"])
def postMnistNewClassification():
  b64_image = request.get_json()['image'].encode('ascii')
  buffer = StringIO(bytearray(b64decode(b64_image)))

  classification = MnistNetwork().classify(mnist_input(buffer = buffer))
  return jsonify({ 'classification': classification })


@app.route("/mnist/image/<int:id>.json", methods = ["GET"])
def getMnistImageJson(id):
  image = digit_image(id)
  buffer = cStringIO.StringIO()
  image.save(buffer, format = "PNG")

  encoded_image = b64encode(buffer.getvalue())
  return jsonify({ 'id': id, 'image': encoded_image })


@app.route("/mnist/image/<int:id>.png", methods = ["GET"])
def getMnistImagePng(id):
  image = digit_image(id)
  io = StringIO()

  image.save(io, 'PNG')
  io.seek(0)
  return send_file(io, mimetype = "image/png")


if __name__ == "__main__":
  app.run(host = "0.0.0.0", port = 80, debug = False)
