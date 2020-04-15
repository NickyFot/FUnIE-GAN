import os
import sys
import argparse
import cv2

parser = argparse.ArgumentParser(description='Enhance underwater images')
parser.add_argument('--inputdir', type=str, required=True, help='Video folder to enhance')
args = parser.parse_args(sys.argv[1:])

for root, dirs, files in os.walk(args.inputdir):
    for file in files:
        if not file.lower().endswith('.mp4'):
            continue
        filename = os.path.join(*[root, file])
        vidcap = cv2.VideoCapture(filename)
        outdir = os.path.join(*[root, file.lower().replace('.mp4', '')])
        if not os.path.isdir(outdir):
          os.mkdir(outdir)
        success, image = vidcap.read()
        count = 0
        while success:
            outfile = os.path.join(outdir, "frame%d.jpg" % count)
            cv2.imwrite(outfile, image)     # save frame as JPEG file
            success, image = vidcap.read()
            count += 1