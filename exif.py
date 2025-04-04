from PIL import Image
from PIL.ExifTags import TAGS
from pprint import pprint

image_file = "/home/dzoey/Pictures/Photos/Google Photos/Joe in Kenya/20170429_173859.jpg"

 # Open the image file
image = Image.open(image_file)
        
# Get the EXIF data
exif_data = image._getexif()

if exif_data is None:
    print("get_exif_data: File " + image_file + "  no EXIF data found");
else:
    # Convert EXIF data to a readable dictionary.  Change items of type bytes into a string so it can be serialized
    exif_dict = {}
    for tag_id, value in exif_data.items():
        # Get the tag name, if possible
        tag = TAGS.get(tag_id, tag_id)
        for t in value.keys():
            if type(value[t]) == dict:
                for t1 in value[t].keys():
                    if type(value[t][t1]) == bytes:
                        value[t][t1] = value[t][t1].decode("utf-8")
            elif type(value[t]) == bytes:
                value[t] = value[t].decode("utf-8") 


        exif_dict[tag] = value

pprint(exif_dict)