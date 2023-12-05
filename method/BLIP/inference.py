from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from models.blip import blip_decoder
import os
from tqdm import tqdm
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--device', default='cuda:2')
parser.add_argument('--model_size', default='base')
parser.add_argument('--image_path', default='', help='path to one of the three category images')
args = parser.parse_args()

device = args.device
image_path = args.image_path

def load_image(image, image_size, device):
    raw_image = Image.open(image).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])
    image = transform(raw_image).unsqueeze(0).to(device)
    return image


image_size = 384
# blip base model
if args.model_size == 'base':
    model_name = 'model_base_caption_capfilt_large.pth'
    model = blip_decoder(pretrained=f'checkpoints/{model_name}', image_size=image_size, vit='base', detox_model=True)
# blip large model
elif args.model_size == 'large':
    model_name = 'model_large_caption.pth'
    model = blip_decoder(pretrained=f'checkpoints/{model_name}', image_size=image_size, vit='large', detox_model=True)
else:
    raise ValueError('model_size must be base or large')

detox_path = 'output/detox'
epoch = 0
it = 5000
if os.path.exists(detox_path):
    toxicity_classification_mlp_ckpt = torch.load(os.path.join(detox_path, f'toxicity_classification_mlp_epoch{epoch}_it{it}.pt'), map_location='cpu')
    detoxification_mlp_ckpt = torch.load(os.path.join(detox_path, f'detoxification_mlp_epoch{epoch}_it{it}.pt'), map_location='cpu')
    model.detox_model.toxicity_classification_mlp.load_state_dict(toxicity_classification_mlp_ckpt)
    model.detox_model.detoxification_mlp.load_state_dict(detoxification_mlp_ckpt)
    print(f'Loaded toxicity_classification_mlp from {os.path.join(detox_path, f"toxicity_classification_mlp_epoch{epoch}_it{it}.pt")} \
            and detoxification_mlp from {os.path.join(detox_path, f"detoxification_mlp_epoch{epoch}_it{it}.pt")}')

model.eval()
model = model.to(device)

image_list = os.listdir(image_path)
image_list.sort()
print(f'loaded images from {image_path}')

cls_name = image_path.split('/')[-2]
write_path = f'blip_large_caption_filtering_{cls_name}.json'
captions_dict = {}
for idx, image_name in tqdm(enumerate(image_list), total=len(image_list)):
    image = load_image(image_path + image_name, image_size=image_size, device=device)
    with torch.no_grad():
        # beam search
        # caption = model.generate(image, sample=False, num_beams=3, max_length=20, min_length=5)
        # nucleus sampling
        caption = model.generate(image, sample=True, top_p=0.9, max_length=20, min_length=5, num_return_sequences=10)
        # print('caption: ' + caption[0])
        img_name = cls_name + '/' + image_name
        captions_dict[img_name] = caption
    if idx % 50 == 0 or idx == len(image_list) - 1:
        with open(write_path, 'w') as f:
            json.dump(captions_dict, f)