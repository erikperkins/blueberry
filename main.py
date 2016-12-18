from flask import Flask, Response, send_file, jsonify
from StringIO import StringIO
import json

from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image
from pylab import gray
from numpy import random, uint8

app = Flask(__name__)
MNIST = input_data.read_data_sets("./MNIST_data/", one_hot = True)

@app.route("/")
def hello():
  return "Hello, World!"

@app.route("/names/<string:name>")
def getName(name):
  data = { 'name': name }
  response = jsonify(data)
  response.status_code = 200
  response.headers['Link'] = 'http://www.backpasture.net'
  return response

@app.route("/image", methods = ["GET"])
def getImage():
  id = random.randint(0, 10000)
  record = MNIST.test.images[id]
  array = record.reshape((28, 28))
  image = Image.fromarray(uint8(255 * (1.0 - array)))

  io = StringIO()
  image.save(io, 'PNG')
  io.seek(0)
  return send_file(io, mimetype = "image/png")

if __name__ == "__main__":
  app.run(host = "0.0.0.0", port = 3005, debug = True)
