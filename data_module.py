import torch
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import CLIPImageProcessor
import json
from PIL import Image
import random
import numpy as np
import os


# function copied from:
# https://github.com/huggingface/diffusers/blob/main/examples/research_projects/dreambooth_inpaint/train_dreambooth_inpaint.py
def prepare_mask_and_masked_image(image, mask):
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

    mask = np.array(mask.convert("L"))
    mask = mask.astype(np.float32) / 255.0
    mask = mask[None, None]
    # because masks are already binary
    # mask[mask < 0.5] = 0
    # mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)

    masked_image = image * (mask < 0.5)

    return mask, masked_image


class MyDataset(Dataset):
    def __init__(
        self,
        json_file,
        tokenizer,
        tokenizer_2,
        size=1024,
        center_crop=True,
        t_drop_rate=0.05,
        i_drop_rate=0.05,
        ti_drop_rate=0.05,
        image_root_path="",
    ):
        super().__init__
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.size = size
        self.center_crop = center_crop
        self.t_drop_rate = t_drop_rate
        self.i_drop_rate = i_drop_rate
        self.ti_drop_rate = ti_drop_rate
        self.image_root_path = image_root_path

        self.data = json.load(open(json_file))

        self.transforms = transforms.Compose(
            [
                # transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.clip_image_processor = CLIPImageProcessor()

    def __getitem__(self, index):
        item = self.data[index]  # single object from json file
        text = item["item"]
        target_img = item["target"]
        cloth = item["source"]
        mask = item["mask"]
        pose = item["pose"]

        raw_target_img = Image.open(os.path.join(self.image_root_path, target_img)).resize(
            (768, 1024)
        )
        raw_cloth_img = Image.open(os.path.join(self.image_root_path, cloth)).resize(
            (768, 1024)
        )
        raw_mask = Image.open(os.path.join(self.image_root_path, mask)).resize(
            (768, 1024)
        )
        raw_pose = Image.open(os.path.join(self.image_root_path, pose)).resize(
            (768, 1024)
        )

        cloth_img_tensor = self.transforms(raw_cloth_img.convert("RGB"))

        pose_tensor = self.transforms(raw_pose)

        target_img_tensor = self.transforms(raw_target_img.convert("RGB"))
        mask_tensor, masked_img_tensor = prepare_mask_and_masked_image(
            raw_target_img, raw_mask
        )

        clip_cloth_img = self.clip_image_processor(
            images=raw_cloth_img, return_tensors="pt"
        ).pixel_values

        prompt_ref = f"a photo of {text}"
        prompt_try = f"model is wearing a {text}"

        # get text and tokenize
        text_input_ids_ref = self.tokenizer(
            prompt_ref,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids

        text_input_ids_2_ref = self.tokenizer_2(
            prompt_ref,
            max_length=self.tokenizer_2.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids

        text_input_ids_try = self.tokenizer(
            prompt_try,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids

        text_input_ids_2_try = self.tokenizer_2(
            prompt_try,
            max_length=self.tokenizer_2.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids

        # drop
        drop_image_embed = 0
        rand_num = random.random()
        if rand_num < self.i_drop_rate:
            drop_image_embed = 1
        elif rand_num < (self.i_drop_rate + self.t_drop_rate):
            text = ""
        elif rand_num < (self.i_drop_rate + self.t_drop_rate + self.ti_drop_rate):
            text = ""
            drop_image_embed = 1

        crop_coords_top_left = torch.tensor([0, 0])
        original_size = torch.tensor([1024, 768])

        return {
            "cloth_image": cloth_img_tensor,  # to be used in garment unet
            "target_image": target_img_tensor,  # output
            "pose_image": pose_tensor,
            "text_input_ids_ref": text_input_ids_ref,
            "text_input_ids_2_ref": text_input_ids_2_ref,
            "text_input_ids_try": text_input_ids_try,
            "text_input_ids_2_try": text_input_ids_2_try,
            "clip_cloth_img": clip_cloth_img,  # input
            "masked_img": masked_img_tensor,  # input
            "mask": mask_tensor,  # input
            "drop_image_embed": drop_image_embed,
            "original_size": original_size,
            "crop_coords_top_left": crop_coords_top_left,
            "target_size": torch.tensor([1024, 768]),
        }

    def __len__(self):
        return len(self.data)


def collate_fn(data):
    cloth_images = torch.stack([example["cloth_image"] for example in data])
    target_images = torch.stack([example["target_image"] for example in data])
    pose_images = torch.stack([example["pose_image"] for example in data])
    text_input_ids_ref = torch.cat(
        [example["text_input_ids_ref"] for example in data], dim=0
    )
    text_input_ids_2_ref = torch.cat(
        [example["text_input_ids_2_ref"] for example in data], dim=0
    )
    text_input_ids_try = torch.cat(
        [example["text_input_ids_try"] for example in data], dim=0
    )
    text_input_ids_2_try = torch.cat(
        [example["text_input_ids_2_try"] for example in data], dim=0
    )
    clip_cloth_imgs = torch.cat([example["clip_cloth_img"] for example in data], dim=0)
    masked_imgs = torch.cat([example["masked_img"] for example in data])
    masks = torch.cat([example["mask"] for example in data])
    drop_image_embeds = [example["drop_image_embed"] for example in data]
    original_size = torch.stack([example["original_size"] for example in data])
    crop_coords_top_left = torch.stack(
        [example["crop_coords_top_left"] for example in data]
    )
    target_size = torch.stack([example["target_size"] for example in data])

    return {
        "cloth_images": cloth_images,
        "target_images": target_images,
        "pose_images": pose_images,
        "text_input_ids_ref": text_input_ids_ref,
        "text_input_ids_2_ref": text_input_ids_2_ref,
        "text_input_ids_try": text_input_ids_try,
        "text_input_ids_2_try": text_input_ids_2_try,
        "clip_cloth_images": clip_cloth_imgs,
        "masked_images": masked_imgs,
        "masks": masks,
        "drop_image_embeds": drop_image_embeds,
        "original_size": original_size,
        "crop_coords_top_left": crop_coords_top_left,
        "target_size": target_size,
    }


if __name__ == "__main__":

    from transformers import CLIPTokenizer

    pretrained_model_name = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
    json_file = "/home/bilal/datasets/tested_data/data_2.json"
    image_root_path = "/home/bilal/datasets/tested_data"

    tokenizer = CLIPTokenizer.from_pretrained(
        pretrained_model_name, subfolder="tokenizer"
    )
    tokenizer_2 = CLIPTokenizer.from_pretrained(
        pretrained_model_name, subfolder="tokenizer_2"
    )

    dataset = MyDataset(
        json_file=json_file,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        image_root_path=image_root_path,
    )

    # shapes before dataloader
    print("------using only dataset--------")
    print(f"Lenght of dataset: {len(dataset)}")
    print(f"Shape of cloth image: {dataset[1]['cloth_image'].shape}")
    print(f"Shape of target image: {dataset[1]['target_image'].shape}")
    print(f"Shape of pose: {dataset[1]['pose_image'].shape}")
    print(f"Shape of clip cloth image: {dataset[1]['clip_cloth_img'].shape}")
    print(f"Shape of masked image: {dataset[1]['masked_img'].shape}")
    print(f"Shape of mask: {dataset[1]['mask'].shape}")
    print(f"Shape of text_input_ids_ref: {dataset[1]['text_input_ids_ref'].shape}")
    print(f"Shape of text_input_ids_2_ref: {dataset[1]['text_input_ids_2_ref'].shape}")
    print(f"Shape of text_input_ids_try: {dataset[1]['text_input_ids_try'].shape}")
    print(f"Shape of text_input_ids_2_try: {dataset[1]['text_input_ids_2_try'].shape}")
    print(f"drop_image_embed: {dataset[1]['drop_image_embed']}")

    # shapes after using dataloader without collate function
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        # collate_fn=collate_fn,
        batch_size=4,
        num_workers=6,
    )
    print("\n------using data loader without collate function--------")
    print(f"Lenght of dataloader with batchsize 4: {len(train_dataloader)}")
    for batch in train_dataloader:
        print(f"Shape of cloth image: {batch['cloth_image'].shape}")
        print(f"Shape of target image: {batch['target_image'].shape}")
        print(f"Shape of pose: {batch['pose_image'].shape}")
        print(f"Shape of clip cloth image: {batch['clip_cloth_img'].shape}")
        print(f"Shape of masked image: {batch['masked_img'].shape}")
        print(f"Shape of mask: {batch['mask'].shape}")
        print(f"Shape of text_input_ids_ref: {batch['text_input_ids_ref'].shape}")
        print(f"Shape of text_input_ids_2_ref: {batch['text_input_ids_2_ref'].shape}")
        print(f"Shape of text_input_ids_try: {batch['text_input_ids_try'].shape}")
        print(f"Shape of text_input_ids_2_try: {batch['text_input_ids_2_try'].shape}")
        print(f"drop_image_embed: {batch['drop_image_embed']}")
        break

    # shapes after using dataloader with collate function
    train_dataloader = torch.utils.data.DataLoader(
        dataset, shuffle=True, collate_fn=collate_fn, batch_size=4, num_workers=6
    )
    print("\n------using data loader with collate function--------")
    print(f"Lenght of dataloader with batchsize 4: {len(train_dataloader)}")
    for batch in train_dataloader:
        print(f"Shape of cloth image: {batch['cloth_images'].shape}")
        print(f"Shape of target image: {batch['target_images'].shape}")
        print(f"Shape of pose: {batch['pose_images'].shape}")
        print(f"Shape of clip cloth image: {batch['clip_cloth_images'].shape}")
        print(f"Shape of masked image: {batch['masked_images'].shape}")
        print(f"Shape of mask: {batch['masks'].shape}")
        print(f"Shape of text_input_ids_ref: {batch['text_input_ids_ref'].shape}")
        print(f"Shape of text_input_ids_2_ref: {batch['text_input_ids_2_ref'].shape}")
        print(f"Shape of text_input_ids_try: {batch['text_input_ids_try'].shape}")
        print(f"Shape of text_input_ids_2_try: {batch['text_input_ids_2_try'].shape}")
        print(f"drop_image_embed: {batch['drop_image_embeds']}")
        break
