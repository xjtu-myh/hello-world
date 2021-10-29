# Improting Image class from PIL module
from PIL import Image
import os
import sys

# Opens a image in RGB mode
infile="/Users/apple/Downloads/style transfer/test-image/星空.jpeg"
im = Image.open(infile)


# Size of the image in pixels (size of orginal image)
# (This is not mandatory)
width, height = im.size
width=max(width,height)
newsize = (width,width)
im = im.resize(newsize)
# Shows the image in image viewer
print(sys.argv)


f = os.path.splitext(infile)
print(f)

outfile = f[0]+ "_resized"+".jpg"
if infile != outfile:



        im.save(outfile)