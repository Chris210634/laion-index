import torch
import open_clip
from tqdm import tqdm
import PIL
import torch.nn.functional as F
import argparse
from utils import *

args = get_argparser().parse_args()
print(args)
    
def get_features(dl, model, d=512):
    n = len(dl.dataset)
    features = torch.zeros(n, d)
    p = 0
    for x in tqdm(dl):
        with torch.cuda.amp.autocast(enabled=True):
            with torch.no_grad():
                z = model.encode_text(x.cuda()).cpu()
        b = len(x)
        assert z.shape[0] == b
        features[p : p+b, :] = z
        p += b
    assert p == n
    return F.normalize(features.float())

model, _, preprocess  = open_clip.create_model_and_transforms(
    args.modelname, 
    pretrained=args.pretrained,
    cache_dir=args.cache_dir
)

tokenizer = open_clip.get_tokenizer(args.modelname)
text_list = torch.load(args.img_list_filename)

dset = TextDataset(text_list, tokenizer)
dl = torch.utils.data.DataLoader(
    dset,
    num_workers=8,
    batch_size=args.bs,
    pin_memory=True,
    shuffle=False
)

model.cuda()
model.eval()
f = get_features(dl, model, d=args.d)
torch.save(f, args.img_list_filename + '.features')