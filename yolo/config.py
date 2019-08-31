#  -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 12:08:15 2018

@author: lenovo
"""

'''
配置参数
'''

import os


# 
#  数据集路径，和模型检查点文件路径
# 

DATA_PATH = 'data'           # 所有数据所在的根目录

PASCAL_PATH = os.path.join(DATA_PATH, 'pascal_voc')   # VOC2012数据集所在的目录

CACHE_PATH = os.path.join(PASCAL_PATH, 'cache')       # 保存生成的数据集标签缓冲文件所在文件夹

OUTPUT_DIR = os.path.join(PASCAL_PATH, 'output')      # 保存生成的网络模型和日志文件所在的文件夹

WEIGHTS_DIR = os.path.join(PASCAL_PATH, 'weights')    # 检查点文件所在的目录

# WEIGHTS_FILE = None

WEIGHTS_FILE = os.path.join(WEIGHTS_DIR, 'YOLO_small.ckpt')

# VOC 2012数据集类别名
CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
           'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
           'train', 'tvmonitor']
# 使用水平镜像，扩大一倍数据集？
FLIPPED = True

'''
网络模型参数
'''

# 图片大小
IMAGE_SIZE = 448

# 单元格大小S  一共有CELL_SIZExCELL_SIZE个单元格  
CELL_SIZE = 7

# 每个单元格边界框的个数B
BOXES_PER_CELL = 2
# 泄露修正线性激活函数系数
ALPHA = 0.1
# 控制台输出信息
DISP_CONSOLE = False

# 损失函数 的权重设置
OBJECT_SCALE = 1.0   # 有目标时，置信度权重
NOOBJECT_SCALE = 1.0 # 没有目标时，置信度权重
CLASS_SCALE = 2.0    # 类别权重
COORD_SCALE = 5.0    # 边界框权重



'''
训练参数设置
'''

GPU = ''
# 学习率
LEARNING_RATE = 0.0001
# 退化学习率衰减步数
DECAY_STEPS = 30000
# 衰减率
DECAY_RATE = 0.1
STAIRCASE = True
# 批量大小
BATCH_SIZE = 4
# 最大迭代次数
MAX_ITER = 15000
# 日志文件保存间隔步
SUMMARY_ITER = 10
# 模型保存间隔步
SAVE_ITER = 50


'''
测试时的相关参数
'''
# 格子有目标的置信度阈值
THRESHOLD = 0.2
# 非极大值抑制 IoU阈值
IOU_THRESHOLD = 0.5