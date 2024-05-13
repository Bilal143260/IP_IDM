import argparse
from pathlib import Path
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from diffusers import AutoencoderKL, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPTextModelWithProjection
import torch
import torch.nn.functional as F
import os
import bitsandbytes as bnb
from data_module import MyDataset, collate_fn
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.unet_hacked_tryon import UNet2DConditionModel as UNet2DConditionModel_try

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="yisol/IDM-VTON",
        # required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--data_json_file",
        type=str,
        default="/home/bilal/datasets/tested_data/data_2.json",
        # required=True,
        help="Training data",
    )
    parser.add_argument(
        "--data_root_path",
        type=str,
        default="/home/bilal/datasets/tested_data",
        # required=True,
        help="Training data root path",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images"
        ),
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--train_batch_size", type=int, default=3, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--noise_offset", type=float, default=None, help="noise offset")
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=6,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=2,
        help=(
            "Save a checkpoint of the training state every X updates"
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=0, help="For distributed training: local_rank")
    
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args

def main():
    args = parse_args()
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    tokenizer_2 = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer_2")
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder_2")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet_try = UNet2DConditionModel_try.from_pretrained(args.pretrained_model_name_or_path,subfolder="unet")
    unet_ref = UNet2DConditionModel_ref.from_pretrained(args.pretrained_model_name_or_path,subfolder="unet_encoder")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.pretrained_model_name_or_path,subfolder="image_encoder")
    
    # freeze parameters of models to save more memory
    # unet.requires_grad_(False)
    unet_ref.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    image_encoder.requires_grad_(False)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    
    unet_ref.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device) # use fp32
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    text_encoder_2.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device, dtype=weight_dtype)

    # optimizer
    params_to_opt = unet_try.parameters()
    # optimizer = torch.optim.AdamW(params_to_opt, lr=args.learning_rate, weight_decay=args.weight_decay)
    optimizer = bnb.optim.Adam8bit(params_to_opt, lr=args.learning_rate, weight_decay=args.weight_decay)


    # dataloader
    train_dataset = MyDataset(args.data_json_file, tokenizer=tokenizer, tokenizer_2=tokenizer_2, size=args.resolution, image_root_path=args.data_root_path)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers
    )

    # Prepare everything with our `accelerator`.
    unet_try, optimizer, train_dataloader = accelerator.prepare(unet_try, optimizer, train_dataloader)

    global_step = 0

    for epoch in range(0, args.num_train_epochs):
        total_loss = 0  # Initialize total loss for the epoch
        num_steps = 0   # Counter for the number of steps
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet_try):

                #convert images to latent space
                with torch.inference_mode():
                    target_latents = vae.encode(batch["target_images"].to(accelerator.device, dtype=torch.float32)).latent_dist.sample()
                    target_latents = target_latents * vae.config.scaling_factor
                    target_latents = target_latents.to(accelerator.device, dtype=weight_dtype)

                    # Convert masked images to latent space
                    masked_latents = vae.encode(
                        batch["masked_images"].reshape(batch["target_images"].shape).to(accelerator.device, dtype=torch.float32)
                    ).latent_dist.sample()
                    masked_latents = masked_latents * vae.config.scaling_factor
                    masked_latents = masked_latents.to(accelerator.device, dtype=weight_dtype)

                    masks = batch["masks"]
                    mask =  F.interpolate(masks, size=(128, 96), mode='bilinear', align_corners=False)
                    # mask =  F.interpolate(masks, size=(64, 48), mode='bilinear', align_corners=False)

                    pose_latents = vae.encode(batch["pose_images"].to(accelerator.device, dtype=torch.float32)).latent_dist.sample()
                    pose_latents = pose_latents * vae.config.scaling_factor
                    pose_latents = pose_latents.to(accelerator.device, dtype=weight_dtype)

                    cloth_latents = vae.encode(batch["cloth_images"].to(accelerator.device, dtype=torch.float32)).latent_dist.sample()
                    cloth_latents = cloth_latents * vae.config.scaling_factor
                    cloth_latents = cloth_latents.to(accelerator.device, dtype=weight_dtype)


                # Sample noise that we'll add to the latents
                noise = torch.randn_like(target_latents)
                # if args.noise_offset:
                #     # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                #     noise += args.noise_offset * torch.randn((target_latents.shape[0], target_latents.shape[1], 1, 1)).to(accelerator.device, dtype=weight_dtype)

                bsz = target_latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=target_latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                target_noisy_latents = noise_scheduler.add_noise(target_latents, noise, timesteps)

                # concatenate the noised latents with the mask and the masked latents
                latent_model_input = torch.cat([target_noisy_latents, mask, masked_latents, pose_latents], dim=1)

                #for reference unet
                noise_ref = torch.randn_like(cloth_latents)
                bsz = cloth_latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=cloth_latents.device)
                timesteps = timesteps.long()
                cloth_noisy_latents = noise_scheduler.add_noise(cloth_latents, noise_ref, timesteps)

                with torch.no_grad():
                    image_embeds = image_encoder(batch["clip_cloth_images"].to(accelerator.device, dtype=weight_dtype),output_hidden_states=True).hidden_states[-2]
                    image_embeds_ = []
                    for image_embed, drop_image_embed in zip(image_embeds, batch["drop_image_embeds"]):
                        if drop_image_embed == 1:
                            image_embeds_.append(torch.zeros_like(image_embed))
                        else:
                            image_embeds_.append(image_embed)
                    image_embeds = torch.stack(image_embeds_)
                    image_embeds = image_embeds.float()
                    image_embeds = unet_try.encoder_hid_proj(image_embeds)
                    # print(f"DIMENSIONS OF IMAGE EMBEDS {image_embeds.shape}")
            
                with torch.no_grad():
                    encoder_output = text_encoder(batch['text_input_ids_try'].to(accelerator.device), output_hidden_states=True)
                    text_embeds = encoder_output.hidden_states[-2]
                    encoder_output_2 = text_encoder_2(batch['text_input_ids_2_try'].to(accelerator.device), output_hidden_states=True)
                    pooled_text_embeds = encoder_output_2[0]
                    text_embeds_2 = encoder_output_2.hidden_states[-2]
                    text_embeds = torch.concat([text_embeds, text_embeds_2], dim=-1)
                    # print(f"DIMENSIONS OF TEXT EMBEDS {text_embeds.shape}")
                
                # add cond
                add_time_ids = [
                    batch["original_size"].to(accelerator.device),
                    batch["crop_coords_top_left"].to(accelerator.device),
                    batch["target_size"].to(accelerator.device),
                ]
                add_time_ids = torch.cat(add_time_ids, dim=1).to(accelerator.device, dtype=weight_dtype)
                unet_added_cond_kwargs = {"text_embeds": pooled_text_embeds,"time_ids":add_time_ids, "image_embeds": image_embeds}

                # print(f"added_cond_kwargs text_embeds shape {unet_added_cond_kwargs['text_embeds'].shape}")
                # print(f"added_cond_kwargs time_ids shape {unet_added_cond_kwargs['time_ids'].shape}")
                # print(f"added_cond_kwargs image_embeds shape {unet_added_cond_kwargs['image_embeds'].shape}")

                with torch.no_grad():
                    encoder_output_ref = text_encoder(batch['text_input_ids_ref'].to(accelerator.device), output_hidden_states=True)
                    text_embeds_ref = encoder_output_ref.hidden_states[-2]
                    encoder_output_2_ref = text_encoder_2(batch['text_input_ids_2_ref'].to(accelerator.device), output_hidden_states=True)
                    pooled_text_embeds_ref = encoder_output_2_ref[0]
                    text_embeds_2_ref = encoder_output_2_ref.hidden_states[-2]
                    text_embeds_ref = torch.concat([text_embeds_ref, text_embeds_2_ref], dim=-1)

                down, garment_features = unet_ref(cloth_noisy_latents, timesteps, text_embeds_ref)

                # print(f"LENGTH OF GARMENT FEATURES: {len(garment_features)}")
                # print(f"SHAPE OF LATENT_MODEL_INPUTS: {latent_model_input.shape}")
                #TODO: check if we need to index the result
                noise_pred = unet_try(latent_model_input, timesteps, text_embeds, 
                                  added_cond_kwargs=unet_added_cond_kwargs, garment_features = garment_features)[0]
                # print(f'SHAPE OF UNET TRY OUTPUT: {noise_pred.shape}')
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean().item()
                
                # Backpropagate
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

                if accelerator.is_main_process:
                    print("Epoch {}, step {}, step_loss: {}".format(
                        epoch, step, avg_loss))
                
                # Update total loss and step count
                total_loss += avg_loss
                num_steps += 1
            
            global_step += 1
            
            if global_step % args.save_steps == 0:
                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                accelerator.save_state(save_path, safe_serialization=False, max_shard_size="15GB")

        # Calculate and log the average loss for the epoch
        average_loss = total_loss / len(train_dataloader)
    unet_try = accelerator.unwrap_model(unet_try)
    accelerator.save_model(unet_try, args.output_dir, safe_serialization=False,  max_shard_size="15GB")

if __name__ == "__main__":
    main()    