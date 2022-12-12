import cv2
import time
import torch
import numpy as np

import psutil
from pynvml.smi import nvidia_smi

from detection.detector import FaceDetector

from utils import create_facebank, add_person, load_facebank_pth, draw_bbox

print("CUDA: {}".format(torch.cuda.is_available()))
print("DEVICE: {}".format(torch.cuda.get_device_name(0)))

if __name__ == '__main__':
    nvsmi = nvidia_smi.getInstance()
    gpu_start = nvsmi.DeviceQuery('memory.free, memory.total')
    ram_start = psutil.virtual_memory()[4]

    detector = FaceDetector()

    embeddings, names = load_facebank_pth('test')

    cap = cv2.VideoCapture("videos/test_afl_2.mp4")
    num_frames = 0
    time_start = time.time()

    gif_list = []
    while True:
        # if (num_frames % 2 == 0) and (num_frames != 0):
        #     num_frames += 1
        #     continue
        ret, frame = cap.read()
        k = cv2.waitKey(1)
        if k % 256 == 27 or not ret:  # ESC
            break
        num_frames += 1

        detections = detector.detect(frame)

        faces = []


        if num_frames == 20:
            nvsmi = nvidia_smi.getInstance()
            gpu_loop = nvsmi.DeviceQuery('memory.free, memory.total')
            ram_loop = psutil.virtual_memory()[4]


        #cv2.imshow('recognition_test', frame)
        # gif_list.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    loop_time = time.time() - time_start

    cap.release()
    cv2.destroyAllWindows()

    fps = num_frames / loop_time
    gpu_memory_usage = round(float(gpu_start['gpu'][0]['fb_memory_usage']['free']) - float(gpu_loop['gpu'][0]['fb_memory_usage']['free']), 3)
    ram_usage = round((ram_start - ram_loop) / 1024 / 1024, 3)

    print("fps: {} || ram_usage: {} || gpu_memory_usage: {} MiB".format(round(fps, 3), ram_usage, gpu_memory_usage))

    # imageio.mimsave('norm.gif', gif_list, fps=round(fps, 0))
    # print("gif saved")

