try:
    import decord
except ImportError:
    raise ImportError(
        'Please run "pip install decord" to install Decord first.')

import io
from mmcv.fileio import FileClient
from PIL import Image
import time
import os

folders = os.listdir('/home/develop/AML-bing_clip/data/kinetics600/val')
file_client = FileClient('disk')
for f in folders:
    tic = time.time()
    if not os.path.exists(os.path.join('/home/develop/AML-bing_clip/data/kinetics600/midframes', f)):
        os.mkdir(os.path.join('/home/develop/AML-bing_clip/data/kinetics600/midframes', f))
    filenames = os.listdir(os.path.join('/home/develop/AML-bing_clip/data/kinetics600/val', f))
    for filename in filenames:
        video_name = filename.split('.')[0]
        img_name = os.path.join('/home/develop/AML-bing_clip/data/kinetics600/midframes', f, video_name+'.jpg')
        if os.path.exists(img_name):
            continue
        try:
            file_obj = io.BytesIO(file_client.get(os.path.join('/home/develop/AML-bing_clip/data/kinetics600/val', f, filename)))
            container = decord.VideoReader(file_obj, num_threads=10)
        except Exception as e:
            print(f"Failed: {filename} ", e)
            continue
        total_frames = len(container)
        mid_frame_id = total_frames // 2
        mid_frame = container[mid_frame_id].asnumpy()
        img = Image.fromarray(mid_frame)
        img.save(img_name, quality=100)
    toc = time.time()
    print(f, toc-tic)