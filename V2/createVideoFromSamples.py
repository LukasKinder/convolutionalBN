import cv2
import os
import sys

from PIL import Image

image_folder = 'samples'
video_name = 'gibbsSampling.avi'

if True:

    for file in [img for img in os.listdir(image_folder) if not img.endswith(".png")]:
        filename, extension  = os.path.splitext(file)

        new_file = os.path.join(image_folder,"{}.png".format(filename))
        with Image.open(os.path.join(image_folder, filename)) as im:
            im.save(new_file)

if True:
    

    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    print("{0} images".format(len(images)))
    if (len(images) == 0):
        sys.exit()
    

    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape


    fourcc = cv2.VideoWriter_fourcc(*'DIVX') 
    print(fourcc)
    print(frame.shape)
    video = cv2.VideoWriter(video_name, fourcc, 30, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()

if True:
    os.popen("ffmpeg -i '{input}' -ac 2 -b:v 2000k -c:a aac -c:v libx264 -b:a 160k -vprofile high -bf 0 -strict experimental -f mp4 '{output}.mp4'".format(input =video_name , output = video_name ))
