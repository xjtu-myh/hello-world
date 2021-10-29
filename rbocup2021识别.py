import sensor, image, time,pyb
from pyb import UART
import nn


sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QQVGA)
sensor.skip_frames(time = 2000)
sensor.set_auto_gain(False) # must be turned off for color tracking
sensor.set_auto_whitebal(False) # must be turned off for color tracking
clock = time.clock()
uart=UART(3,9600)
net = nn.load('/lenet.network')

# 十个标签
labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

while(True):
    clock.tick()
    img = sensor.snapshot().lens_corr(1.8)
    #找长方形
    for r in img.find_rects(threshold=10000):
        img.draw_rectangle(r.rect(), color=(255, 0, 0))
        for p in r.corners():
            img.draw_circle(p[0], p[1], 5, color=(0, 255, 0))


            uart.write("rect")


  # 二维码
    for code in img.find_qrcodes():
        img.draw_rectangle(code.rect(), color=(255, 0, 0))
        message = code.payload()
        if message == '2021':
            print("qr_code")
            uart.write("qr_code")
  #颜色块
    blob=img.find_blobs([red])
    if blob:
        uart.write("red")
        print("red")


    # copy()表示创建一个图像副本储存在MicroPython堆中而不是帧缓冲区
    # 二值化是为了方便处理，阈值可以自己设定
    out = net.forward(img.copy().binary([(150, 255)], invert=True))

    # 挑选列表中的最大值
    max_id = out.index(max(out))

    # 将0-1之间的值扩大到百分制
    score = int(out[max_id] * 100)

    # 70分以上算识别成功
    if (score < 70):
        score_str = "Nothing"
    else:
        uart.write("digits")




