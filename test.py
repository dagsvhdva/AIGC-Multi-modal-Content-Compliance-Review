"""
eval pretained model.
"""
import argparse
import random
import sys
import io
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.utils.data
import yaml
from tqdm import tqdm
from model.video_detector.detectors import DETECTOR
from PIL import Image
from torchvision import transforms as T
from metrics.utils import get_test_metrics

torch.multiprocessing.set_sharing_strategy('file_system')
parser = argparse.ArgumentParser(description='Deepfake Detection Test Args')
parser.add_argument('--config_file', type=str, default='model/video_detector/config/detector/lsda.yaml', help='path to detector YAML file')
args = parser.parse_args(args=[])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def inference(model, data_dict):
    predictions = model(data_dict, inference=True)
    return predictions

def video_process(video_path):
    # prepare the model config
    with open(args.config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    # prepare the model (detector)
    model_class = DETECTOR['lsda']
    if torch.cuda.is_available():
        model = model_class(config).to(device)
    else:
        model = model_class(config)
    ckpt = torch.load('model/video_detector/pretrained/ckpt_best.pth', map_location=device)
    model.load_state_dict(ckpt, strict=False)
    if ckpt:
        print('===> Load checkpoint done!')
    else:
        print('===> Fail to load the pre-trained weights!')
    # start testing
    model.eval()
    data_dict = {}
    cap = cv2.VideoCapture(video_path)
    print(video_path)
    if cap:
        print('视频成功导入！！！')
    else:
        print('视频未成功导入！！！')
    cap = cv2.VideoCapture(video_path)
    sum = 0.0
    num = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        print('帧加载成功！！！')
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_CUBIC)
        data_tensor = T.ToTensor()(np.array(Image.fromarray(np.array(img, dtype=np.uint8)))).unsqueeze(dim=0)
        data_tensor = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(data_tensor)
        if torch.cuda.is_available():
            data_tensor = data_tensor.to(device)
        data_dict["image"] = data_tensor
        data_dict["label"] = torch.ones_like(data_tensor)
        # data_dict["label_spe"] = torch.ones_like(data_tensor)
        predictions = inference(model, data_dict)
        print('预测结果：', predictions)
        # 将预测结果叠加到原始帧上
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        text = "prob of deepfake:{:.3f}".format(predictions["prob"].item())
        sum =sum+predictions["prob"].item()
        num =num+1
        (text_width, text_height) = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX, 2.0, 2)[0]
        # 在文字上方绘制白色矩形框
        cv2.rectangle(frame, (50, 50 - text_height - 10), (50 + text_width + 10, 50 + 10), (255, 255, 255), -1)
        # 在矩形框上绘制红色文字
        cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 2.0, (236, 108, 146), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        if ret:
            print('变量存储成功！！！')
        else:
            print('变量存储失败！！！')
        img_array = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + img_array + b'\r\n')

    cap.release()

def get_prob(video_path):
    # prepare the model config
    with open(args.config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    # prepare the model (detector)
    model_class = DETECTOR['lsda']
    if torch.cuda.is_available():
        model = model_class(config).to(device)
    else:
        model = model_class(config)
    ckpt = torch.load('model/video_detector/pretrained/ckpt_best.pth', map_location=device)
    model.load_state_dict(ckpt, strict=False)
    if ckpt:
        print('===> Load checkpoint done!')
    else:
        print('===> Fail to load the pre-trained weights!')
    # start testing
    model.eval()
    data_dict = {}
    cap = cv2.VideoCapture(video_path)
    print(video_path)
    if cap:
        print('视频成功导入！！！')
    else:
        print('视频未成功导入！！！')
    cap = cv2.VideoCapture(video_path)
    sum = 0.0
    num = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        print('帧加载成功！！！')
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_CUBIC)
        data_tensor = T.ToTensor()(np.array(Image.fromarray(np.array(img, dtype=np.uint8)))).unsqueeze(dim=0)
        data_tensor = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(data_tensor)
        if torch.cuda.is_available():
            data_tensor = data_tensor.to(device)
        data_dict["image"] = data_tensor
        data_dict["label"] = torch.ones_like(data_tensor)
        # data_dict["label_spe"] = torch.ones_like(data_tensor)
        predictions = inference(model, data_dict)
        print('预测结果：', predictions)
        # 将预测结果叠加到原始帧上
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # text = "prob of deepfake:{:.3f}".format(predictions["prob"].item())
        sum =sum+predictions["prob"].item()
        num =num+1
        ret, buffer = cv2.imencode('.jpg', frame)
        if ret:
            print('变量存储成功！！！')
        else:
            print('变量存储失败！！！')

    cap.release()
    prob = round(sum / num, 4)
    print(prob)
    return float(prob)
