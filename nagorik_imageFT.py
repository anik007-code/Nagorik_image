from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from diffusers import StableDiffusionPipeline
from tqdm import tqdm

class ImageToImageDataset(Dataset):
    def __init__(self, input_images, target_images, transform=None):
        self.input_images = input_images
        self.target_images = target_images
        self.transform = transform

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        input_image = Image.open(self.input_images[idx]).convert("RGB")
        target_image = Image.open(self.target_images[idx]).convert("RGB")

        if self.transform:
            input_image = self.transform(input_image)
            target_image = self.transform(target_image)

        return input_image, target_image


transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

input_images = ["sketch1.jpg", "sketch2.jpg"]
target_images = ["image1.jpg", "image2.jpg"]

# Create dataset and dataloaders
train_dataset = ImageToImageDataset(input_images, target_images, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

# Load Stable Diffusion pipeline . Note: Only for CPU.
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4-original")
pipe.to("cpu")  #

# Set optimizer and learning rate
optimizer = optim.AdamW(pipe.parameters(), lr=5e-6)

# Fine-tuning loop
num_epochs = 5  # You have to increase the number to have a better model or to get the better images.
for epoch in range(num_epochs):
    pipe.train()
    running_loss = 0.0

    for input_image, target_image in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        input_image = input_image.to("cpu")
        target_image = target_image.to("cpu")
        optimizer.zero_grad()
        generated_image = pipe(input_image)
        loss = nn.MSELoss()(generated_image, target_image)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}")

# Save the fine-tuned model
pipe.save_pretrained('./Model/nagorik_diffusion_model')

print("Model training complete and saved.")



