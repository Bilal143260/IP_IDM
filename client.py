import requests
from PIL import Image
import io
import base64
from io import BytesIO

def image_to_base64(pil_image: Image.Image) -> str:
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")  # Adjust format as needed
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def base64_to_image(base64_string):
    try:
        # Check and remove the prefix if it's there
        if base64_string.startswith('data:image'):
            base64_string = base64_string.split(',')[1]
        
        # Decode the base64 string
        try:
            image_data = base64.b64decode(base64_string)
        except base64.binascii.Error as e:
            print(f"Error decoding base64 string: {e}")
            return None
        
        # Load the image data into a PIL Image object
        try:
            image = Image.open(BytesIO(image_data))
            # Convert the image to RGB
            rgb_image = image.convert('RGB')
            return rgb_image
        except IOError as e:
            print(f"Error loading image data: {e}")
            return None

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

source_image = Image.open('/home/bilal/IDM-VTON/samples/17/image.png')
target_image = Image.open('/home/bilal/IDM-VTON/samples/17/JEANS_3A-enhanced.png')
target_mask = Image.open('/home/bilal/IDM-VTON/samples/18/m6 (1).png')

# Server URL
url = "http://0.0.0.0:7860/generate_image"

# Convert images to base64 and prepare the data
data = {
    "source_image": image_to_base64(source_image),
    "target_image": image_to_base64(target_image),
    "target_mask": image_to_base64(target_mask)
}

# Send the data to the server as JSON
response = requests.post(url, json=data)

print("Response Status:", response.status_code)
# print("Response Body:", response.text)
response_data = response.json()  # Parse JSON to get the data
# Assuming the image data is returned with the key 'image'
image_base64 = response_data['image']
image = base64_to_image(image_base64)
image.save("generate.png")