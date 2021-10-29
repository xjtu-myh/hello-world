import os
from PIL import Image

# 源目录
project_dir = os.path.dirname("/Users/apple/Downloads/style transfer/test-image/chicago_resized.jpg")
input ="/Users/apple/Downloads/mug"

# 输出目录

output = "/Users/apple/Downloads/mug"

def modify():
    # 切换目录
    os.chdir(input)

    # 遍历目录下所有的文件
    for image_name in os.listdir(os.getcwd()):
        if image_name==(".DS_Store" or 'mug1'):
            pass
        else:
            print(image_name)
            im = Image.open(os.path.join(input, image_name))
            width, height = im.size
            width = max(width, height)
            newsize = (width, width)
            im = im.resize(newsize)
            im=im.convert('RGB')

            im.save(image_name+'.jpg')


if __name__ == '__main__':
    modify()