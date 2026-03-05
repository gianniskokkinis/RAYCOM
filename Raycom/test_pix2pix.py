import argparse
import os
import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# Import the Generator class from the training file
# Make sure train_pix2pix.py is in the same folder
from train_pix2pix import UNetGenerator

def run_inference(checkpoint_path, input_path, output_path, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Loading Architectural Model
    netG = UNetGenerator().to(device)

    # Weight Loading (Checkpoint)
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    netG.load_state_dict(checkpoint)
    netG.eval() 

    # Preparing for Transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    to_pil = transforms.ToPILImage()

    # Specifying Inputs (File or Folder)
    if os.path.isdir(input_path):
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
        input_files = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.lower().endswith(image_extensions)]
        is_dir = True
        if not os.path.exists(output_path):
            os.makedirs(output_path)
    else:
        input_files = [input_path]
        is_dir = False

    if not input_files:
        print(f"No valid images found at {input_path}")
        return

    print(f"Found {len(input_files)} image(s). Processing...")

    # Execution (Inference Loop)
    with torch.no_grad():
        for img_path in tqdm(input_files, desc="Generating"):
            img = Image.open(img_path).convert("RGB")
            img_tensor = transform(img).unsqueeze(0).to(device)

            output_tensor = netG(img_tensor)

            # Denormalize: [-1, 1] -> [0, 1]
            output_tensor = output_tensor.squeeze(0).cpu()
            output_tensor = output_tensor * 0.5 + 0.5
            output_tensor = torch.clamp(output_tensor, 0, 1)

            result_image = to_pil(output_tensor)

            # Save
            if is_dir:
                filename = os.path.basename(img_path)
                save_path = os.path.join(output_path, f"pred_{filename}")
            else:
                save_path = output_path
                
            result_image.save(save_path)

    print(f"Process complete. Results saved in: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate optical images from RF heatmaps (Single or Batch)")
    parser.add_argument("--checkpoint", required=True, help="Path to generator checkpoint (.pth)")
    parser.add_argument("--input", required=True, help="Path to input image or directory")
    parser.add_argument("--output", required=True, help="Path to output image file or directory")
    
    args = parser.parse_args()

    run_inference(args.checkpoint, args.input, args.output)
