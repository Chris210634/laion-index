import torch
import open_clip
from tqdm import tqdm
import PIL
import torch.nn.functional as F
import argparse

def get_argparser():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--modelname', default = 'ViT-L-14', type =str)
    parser.add_argument('--pretrained', default = 'openai', type =str)
    parser.add_argument('--cache_dir', default = '/projectnb/textconv/cliao25/data', type =str)
    parser.add_argument('--img_list_filename', default = '', type =str)
    parser.add_argument('--d', default = 768, type =int)
    parser.add_argument('--dataset', default = '', type =str)
    parser.add_argument('--bs', default = 64, type =int)
    return parser

class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, imgs, transform, metadata=None):
        self.transform = transform
        self.imgs = imgs
        self.metadata = metadata
        
        from PIL import ImageFile
        ImageFile.LOAD_TRUNCATED_IMAGES = True

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        def img_load(index):
            try:
                imraw = PIL.Image.open(self.imgs[index])
            except PIL.UnidentifiedImageError:
                width, height = 400, 300
                white_image = PIL.Image.new("RGB", (width, height), color="white")
                imraw = white_image
            # convert gray to rgb
            try:
                if len(list(imraw.split())) == 1 : imraw = imraw.convert('RGB')
                if imraw.mode != 'RGB' : imraw = imraw.convert('RGB')
            except OSError:
                print(self.imgs[index])
            if self.transform is not None:
                im = self.transform(imraw)
            return im

        im = img_load(index)
        return im
    
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, text_list, tokenizer):
        self.text_list = text_list
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.text_list)

    def __getitem__(self, index):
        im = self.tokenizer(self.text_list[index]).view(-1)
        return im