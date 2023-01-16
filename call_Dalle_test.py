import openai
import requests
import json
import base64
import os

# DALL-E API endpoint
DALLE_ENDPOINT_GENERE = 'https://api.openai.com/v1/images/generations'
DALLE_ENDPOINT_VARIA =  'https://api.openai.com/v1/images/variations'
# DALL-E API key
API_KEY = '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'

# Path to the image file
image_path = 'C:/kobbi/endProject/TSKinFace_Data/Azura_Test/test/FMD-1-D.jpg'
headers = {
    'Authorization': f'Bearer {API_KEY}'
}
# Open the image file and read its contents
with open(image_path, 'rb') as image_file:
    image_data = image_file.read()
openai.api_key = API_KEY

"""res = openai.Image.create(
  prompt="A badass panda cook some space noodles in the space photorealistic 4k nikon",
  n=2,
  size="1024x1024"
)
print(res)"""
fofu = openai.Image.create_variation(
  image= image_path,
  n=2,
  size="256x256" 
)
print(fofu)
"""encoded_image = base64.b64encode(image_data).decode('utf-8')

# Create a JSON payload for the API request
payload = json.dumps({'image': encoded_image})

# Set the headers for the API request
headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {API_KEY}'
}

# Set the parameters for the API request
params = {
    'model': 'image-alpha-001'
}

# Make the API request
response = requests.post(DALLE_ENDPOINT_GENERE, headers=headers, params=params, data=image_data)

# Get the vector representation of the image from the API response
if response.status_code == 200:
    vector = response.json()['data'][0]['vector']
    print(vector)
else:
    print(response.status_code)
    print(response.text)"""
