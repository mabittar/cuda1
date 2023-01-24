import torch
import torchvision.models as models


def main():
    print(
        f"Pytorch CUDA Version is  {torch.version.cuda}. CUDA is supported {torch.cuda.is_available()}"
    )
    if torch.cuda.is_available():
        cuda_id = torch.cuda.current_device()
        print(f"CUDA executed by: {torch.cuda.get_device_name(cuda_id)}")

    x = torch.randint(1, 1000, (100, 100))
    print(x)

    # Instantiating a pre-trained model
    model = models.resnet18(pretrained=True)
    # Making the code device-agnostic
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.


if __name__ == "__main__":
    try:
        main()

    except Exception as e:
        print(e)
