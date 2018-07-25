import h5py, os
import gzip
import torch
import  six.moves.cPickle as pickle
from PIL import Image
from torch.utils import data
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
from torch import cat, t
import numpy as np

class MySVHN(data.Dataset):
    
    def __init__(self, root, split='train', process=False):
        self.split_list = {'train', 'test', 'extra', 'error', 
                    'error0', 'error1', 'error2', 'error3', 'error4', 
                    'error5', 'error6', 'error7', 'error8', 'error9'}
        self.root = os.path.expanduser(root)
        self.split = split
        self.process = process

        if self.split.startswith('error'):
            self.process = False
        
        if self.split not in self.split_list:
            raise ValueError('Wrong split entered!')
        
        if self.process:
            self.data, self.labels = self.process_data(split)
            self.data = self.data / 255.
        else:
            f = gzip.open(os.path.join(os.path.split(__file__)[0], '%spkl.gz' % self.split), 'rb')
            data_set = pickle.load(f)
            self.data = data_set.pop('images') / 255.
            self.labels = data_set.pop('labels')
    

    def __getitem__(self, index):
        img, target = self.data[index], int(self.labels[index])
        return torch.Tensor(img), target

    
    def __len__(self):
        return len(self.data)


    def process_data(self, data_type):
        print('Process %s data...' % data_type)
        f = h5py.File('./%s/digitStruct.mat' % data_type, 'r')
        digitStructName = f['digitStruct']['name']
        digitStructBbox = f['digitStruct']['bbox']

        def getName(n):
            return ''.join([chr(c[0]) for c in f[digitStructName[n][0]].value])


        def bboxHelper(attr):
            attr = [attr.value[0][0]] if len(attr) <= 1 else [f[attr.value[j].item()].value[0][0] for j in range(len(attr))]
            return attr


        def getBbox(n):
            bbox = {}
            bb = digitStructBbox[n].item()
            bbox['height'] = bboxHelper(f[bb]["height"])
            bbox['label'] = bboxHelper(f[bb]["label"])
            bbox['left'] = bboxHelper(f[bb]["left"])
            bbox['top'] = bboxHelper(f[bb]["top"])
            bbox['width'] = bboxHelper(f[bb]["width"])
            return bbox


        print ('... creating image box bound dict for %sing data' % data_type)
        image_dict = {}
        for i in range(len(digitStructName)):
            image_dict[getName(i)] = getBbox(i)
            if i % 5000 == 0:
                print('     image dict processing: %i/%i complete' %(i,len(digitStructName)))
        print('... dict processing complete')

        print('... processing image data and labels')
        names = []
        for item in os.listdir('./%s' % data_type):
            if item.endswith('.png'):
                names.append(item)
        x, y = [], []
        for i in range(len(names)):
            process = image_dict[names[i]]
            length = len(process['label'])
            y += process['label']
            image = Image.open('./%s/' % data_type + names[i])
            for j in range(length):
                left, right = int(process['left'][j]), int(process['left'][j] + process['width'][j])
                top, bottom = int(process['top'][j]), int(process['top'][j] + process['height'][j])
                img = image.crop(box=(left, top, right, bottom))
                img_tensor = ToTensor()(img)
                x.append(img_tensor)
            if i % 5000 == 0:
                print('     image processing: %i/%i complete' %(i,len(names)))
        print('... image processing complete')

        side = 32
        crop = transforms.CenterCrop(side)
        topil = transforms.ToPILImage()
        totensor = transforms.ToTensor()

        def get_fill(img):
            channel0 = cat((img[0][0], t(img[0])[0][1:], img[0][-1][1:], t(img[0])[-1][1:-1]), 0).median()
            channel1 = cat((img[1][0], t(img[1])[0][1:], img[1][-1][1:], t(img[1])[-1][1:-1]), 0).median()
            channel2 = cat((img[2][0], t(img[2])[0][1:], img[2][-1][1:], t(img[2])[-1][1:-1]), 0).median()
            return (int(channel0.tolist() * 255), int(channel1.tolist() * 255), int(channel2.tolist() * 255))


        def trans(num):
            img = topil(num)
            width, height = img.size

            short = int(side / width * height) if width >= height else int(side * width / height)
            resize = transforms.Resize(size=(short, side)) if width >= height else transforms.Resize(size=(side, short))
            img = resize(img)

            pad = transforms.Pad(side - short, fill=get_fill(num))
            return totensor(crop(pad(img)))

        x_array = []
        y_array = []
        print('... transform processed images')
        for i in range(len(x)):
            try:
                x_array.append(np.array(topil(trans(x[i]))))
                y_array.append(int(y[i]) % 10)
            except:
                continue
            if i % 5000 == 0:
                print('     image transforming: %i/%i complete' % (i,len(x)))

        x_array = np.array(x_array)
        y_array = np.array(y_array)
        x_array = np.rollaxis(x_array, 3, 1)
        print(len(x_array), len(y_array))
        print('... image transforming complete')


        print('... pickling data')
        out = {'labels': y_array, 'images': x_array}
        output_file = data_type + 'pkl.gz'
        out_path = './' + output_file
        p = gzip.open(out_path, 'wb')
        pickle.dump(out, p)
        p.close()
        return (x_array, y_array)
    
if __name__ == '__main__':
    train_data = MySVHN('./', split='train', process=True)
    test_data = MySVHN('./', split='test', process=True)
    extra_data = MySVHN('./', split='extra', process=True)
