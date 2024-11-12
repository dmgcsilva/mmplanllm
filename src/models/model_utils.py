from PIL import Image
import requests
from io import BytesIO


def truncate_caption(caption: str) -> str:
    """Truncate captions at periods and newlines."""
    caption = caption.strip('\n')
    trunc_index = caption.find('\n') + 1
    if trunc_index <= 0:
        trunc_index = caption.find('.') + 1
    if trunc_index > 0:
        caption = caption[:trunc_index]
    return caption

def get_image_from_url(url: str):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img = img.resize((224, 224))
    img = img.convert('RGB')
    return img