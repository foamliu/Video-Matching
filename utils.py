from PIL import Image
from scipy.stats import norm
from torchvision import transforms

from config import device

# model params
threshold = 25.50393648495902
mu_0 = 46.1028
sigma_0 = 6.4981
mu_1 = 9.6851
sigma_1 = 3.060

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
transformer = data_transforms['val']


def get_image(img):
    img = img[..., ::-1]  # RGB
    img = Image.fromarray(img, 'RGB')  # RGB
    img = transformer(img)
    return img.to(device)


def get_prob(theta):
    prob_0 = norm.pdf(theta, mu_0, sigma_0)
    prob_1 = norm.pdf(theta, mu_1, sigma_1)
    total = prob_0 + prob_1
    return prob_1 / total
