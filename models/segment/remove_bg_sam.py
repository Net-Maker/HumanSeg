from PIL import Image
from rembg import new_session, remove

input_path = 'test.jpg'
output_path = './output_org.png'

input = Image.open(input_path)

output = remove(input)
output.save(output_path)