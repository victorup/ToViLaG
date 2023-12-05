import os
import json

from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url

from PIL import Image

from data.utils import pre_caption

class detox_dataset(Dataset):
    def __init__(self, transform, max_words=30, prompt=''):        
        self.transform = transform
        self.prompt = prompt
        self.max_words = max_words

        import jsonlines
        import re
        import random
        codebase = re.findall(r'.*codebase/', os.getcwd())[0]
        self.data = []
        detox_data_path = codebase + 'stable-diffusion/detoxification_data/train/'
        image_files = []
        captions = []
        toxicity_labels = []
        with jsonlines.open(f'{detox_data_path}metadata.jsonl', 'r') as reader:  # new_dict = {"file_name": image, "text": text, 'toxicity_label': 0}
            for obj in reader:
                image_files.append(detox_data_path + obj['file_name'])
                captions.append(obj['text'])
                toxicity_labels.append(obj['toxicity_label'])

        # combine pairs
        zipped_list = list(zip(image_files, captions, toxicity_labels))
        print(f'training set: {len(zipped_list)}')
        random.shuffle(zipped_list)
        for i in zipped_list:
            self.data.append({'file_name': i[0], 'text': i[1], 'toxicity_label': i[2]})
        print('training sample:', self.data[0])
        print(f'training total data: {len(self.data)}')
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):    
        item = self.data[index]
        image = Image.open(item['file_name']).convert('RGB')   
        image = self.transform(image)
        caption = self.prompt + pre_caption(item['text'], self.max_words) 
        toxicity_label = item['toxicity_label']
        return image, caption, toxicity_label
    