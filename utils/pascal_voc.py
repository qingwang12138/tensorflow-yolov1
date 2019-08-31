import os
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import pickle
import copy
import yolo.config as cfg


class pascal_voc(object):
    def __init__(self, phase, rebuild=False):
        self.devkil_path = os.path.join(cfg.PASCAL_PATH, 'VOCdevkit')
        self.data_path = os.path.join(self.devkil_path, 'VOC2012')  #数据路径连接
        self.cache_path = cfg.CACHE_PATH
        self.batch_size = cfg.BATCH_SIZE
        self.image_size = cfg.IMAGE_SIZE
        self.cell_size = cfg.CELL_SIZE
        self.classes = cfg.CLASSES
        self.class_to_ind = dict(zip(self.classes, range(len(self.classes)))) #生成类的字典：{'aeroplane'：0，.....}
        self.flipped = cfg.FLIPPED
        self.phase = phase
        self.rebuild = rebuild  # rebuild：是否重新创建数据集的标签文件，保存在缓存文件夹下
        self.cursor = 0  #从gt_labels加载数据，cursor表明当前读取到第几个
        self.epoch = 1
        self.gt_labels = None
        self.prepare()   #这里调用了load_labels()函数生成了gt_labels图片标签列表，其中load_labels()函数又调用了load_pascal_annotation函数来制作标签

    def get(self):   #生成一个batch_size的图片和对应的lable
        images = np.zeros(
            (self.batch_size, self.image_size, self.image_size, 3))
        labels = np.zeros(
            (self.batch_size, self.cell_size, self.cell_size, 25))
        count = 0
        while count < self.batch_size:
            imname = self.gt_labels[self.cursor]['imname']
            flipped = self.gt_labels[self.cursor]['flipped']
            images[count, :, :, :] = self.image_read(imname, flipped)
            labels[count, :, :, :] = self.gt_labels[self.cursor]['label']
            count += 1
            self.cursor += 1
            if self.cursor >= len(self.gt_labels):
                np.random.shuffle(self.gt_labels)
                self.cursor = 0
                self.epoch += 1
        return images, labels

    def image_read(self, imname, flipped=False):  #读取图片，并把图像resieze到image_size,因为opencv读取的bgr格式转化为rgb格式，同时将图像归一化到[-1,1]
        image = cv2.imread(imname)
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image = (image / 255.0) * 2.0 - 1.0
        if flipped:  #图片翻转
            image = image[:, ::-1, :]  #将列进行翻转，即水平翻转
        return image

    def prepare(self):  #调用load_label函数生成gt_labels，如果flipped图片进行了水平翻转，那么对应的box的坐标也要翻转
        gt_labels = self.load_labels()
        if self.flipped:
            print('Appending horizontally-flipped training examples ...')
            gt_labels_cp = copy.deepcopy(gt_labels)   #复制gt_labels
            for idx in range(len(gt_labels_cp)):
                gt_labels_cp[idx]['flipped'] = True
                gt_labels_cp[idx]['label'] =\
                    gt_labels_cp[idx]['label'][:, ::-1, :]   #列倒序，因为水平翻转
                for i in range(self.cell_size):
                    for j in range(self.cell_size):
                        if gt_labels_cp[idx]['label'][i, j, 0] == 1:  #若某个cell中有物体，则将该box的水平坐标进行翻转
                            gt_labels_cp[idx]['label'][i, j, 1] = \
                                self.image_size - 1 -\
                                gt_labels_cp[idx]['label'][i, j, 1]
            gt_labels += gt_labels_cp
        np.random.shuffle(gt_labels)  #在一维上打错列表元素的顺序，相当于打乱元素的顺序
        self.gt_labels = gt_labels
        return gt_labels

    def load_labels(self):  #所有图像都读取出来，调用函数load_pascal_annotation做成label，并存在gt_labels里面。同时保存在pickle里面
        cache_file = os.path.join(
            self.cache_path, 'pascal_' + self.phase + '_gt_labels.pkl')

        if os.path.isfile(cache_file) and not self.rebuild:
            print('Loading gt_labels from: ' + cache_file)
            with open(cache_file, 'rb') as f:    #以二进制格式打开一个文件用于只读
                gt_labels = pickle.load(f)
            return gt_labels

        print('Processing gt_labels from: ' + self.data_path)

        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)

        if self.phase == 'train':   #根据不同的模式访问不同的文件
            txtname = os.path.join(
                self.data_path, 'ImageSets', 'Main', 'trainval.txt')
        else:
            txtname = os.path.join(
                self.data_path, 'ImageSets', 'Main', 'test.txt')
        with open(txtname, 'r') as f:  #生成图片索引列表
            self.image_index = [x.strip() for x in f.readlines()]

        gt_labels = []  #为读取的每一张图片制作标签并存在gt_labels中
        for index in self.image_index:
            label, num = self.load_pascal_annotation(index)
            if num == 0:
                continue
            imname = os.path.join(self.data_path, 'JPEGImages', index + '.jpg')
            gt_labels.append({'imname': imname,
                              'label': label,
                              'flipped': False})
        print('Saving gt_labels to: ' + cache_file)  #gt_labels的元素是字典，每一个字典元素中包含图片名和对应的label
        with open(cache_file, 'wb') as f:
            pickle.dump(gt_labels, f)
        return gt_labels

    def load_pascal_annotation(self, index): #从PASCAL VOC的XML file中解析出Box坐标
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """

        imname = os.path.join(self.data_path, 'JPEGImages', index + '.jpg')
        im = cv2.imread(imname)
        h_ratio = 1.0 * self.image_size / im.shape[0]  #因为图像被resize所以进行坐标变换
        w_ratio = 1.0 * self.image_size / im.shape[1]
        # im = cv2.resize(im, [self.image_size, self.image_size])

        label = np.zeros((self.cell_size, self.cell_size, 25))
        filename = os.path.join(self.data_path, 'Annotations', index + '.xml')
        tree = ET.parse(filename)
        objs = tree.findall('object')

        for obj in objs:
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based，减去1为了让像素索引从0开始，min和max是控制框在图像边界内
            x1 = max(min((float(bbox.find('xmin').text) - 1) * w_ratio, self.image_size - 1), 0)
            y1 = max(min((float(bbox.find('ymin').text) - 1) * h_ratio, self.image_size - 1), 0)
            x2 = max(min((float(bbox.find('xmax').text) - 1) * w_ratio, self.image_size - 1), 0)
            y2 = max(min((float(bbox.find('ymax').text) - 1) * h_ratio, self.image_size - 1), 0)
            cls_ind = self.class_to_ind[obj.find('name').text.lower().strip()] #调用字典中元素，得到对应类别的数字编号
            boxes = [(x2 + x1) / 2.0, (y2 + y1) / 2.0, x2 - x1, y2 - y1] #将左上右下坐标转变为框的中心点坐标和长和宽
            x_ind = int(boxes[0] * self.cell_size / self.image_size)   #计算框的中心落在哪个cell中
            y_ind = int(boxes[1] * self.cell_size / self.image_size)
            if label[y_ind, x_ind, 0] == 1:
                continue
            label[y_ind, x_ind, 0] = 1  #label里面的第一个值是表示置信度，第2-5是bbox的坐标，后面表示类别ont-hot
            label[y_ind, x_ind, 1:5] = boxes
            label[y_ind, x_ind, 5 + cls_ind] = 1

        return label, len(objs)
