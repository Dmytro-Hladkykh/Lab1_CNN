import torch
from PIL import Image
from torchvision import transforms
from model import CNN
from data import get_data_loaders, get_class_labels
from train import train

def main():
    # Load data
    train_loader, val_loader, test_loader = get_data_loaders(batch_size=64)

    # Initialize model
    model = CNN()

    # Train model
    num_epochs = 100
    lr = 0.001
    train(model, train_loader, val_loader, num_epochs, lr)

    # Load class labels
    class_labels = get_class_labels()

    # Load an example image
    image_path = 'cat.jpg'
    image = Image.open(image_path).convert('RGB')

    # Preprocess the example image
    preprocess = transforms.Compose([
        transforms.Resize((32, 32)),  # Resize the image to match the input size of your model
        transforms.ToTensor(),         # Convert the image to a PyTorch tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize the image
    ])
    input_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension

    # Perform inference
    with torch.no_grad():
        output = model(input_tensor)

    # Interpret the model's output
    _, predicted_class = torch.max(output, 1)
    predicted_label = class_labels[predicted_class.item()]

    print("Predicted class for the example image:", predicted_label)

if __name__ == "__main__":
    main()
