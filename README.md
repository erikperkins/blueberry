# flask_app
Service for machine learning applications


To get data from S3, do

import boto3
s3 = boto3.resource('s3')
data = s3.Object('bucket-name', 'path/to/object').get()['Body'].read()


To read an image directly into memory, do

import boto3
import Image
from StringIO import StringIO

s3 = boto3.resource('s3')
body = s3.Object('bucket-name', 'path/to/object').get()['Body'].read() # octet-stream
bytes = bytearray(body) # Convert octet-stream to bytearray
image = Image.open(StringIO(bytes)) # Convert bytearray to image
image.show()
