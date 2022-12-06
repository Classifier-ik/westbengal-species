from PIL import Image
from torchvision import models, transforms
import torch
import pandas
import pickle
from github import Github

g = Github("Classifier-ik", "whitebeard1999")


def predict(image_path, class_names):
    model = torch.load('Googlenet_50_epochs',
                       map_location=torch.device('cpu'))

    # https://pytorch.org/docs/stable/torchvision/models.html
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )])

    img = Image.open(image_path)
    batch_t = torch.unsqueeze(transform(img), 0)

    model.eval()
    outputs = model(batch_t)
    _, predicted = torch.max(outputs, 1)
    title = [class_names[x] for x in predicted]
    prob = torch.nn.functional.softmax(outputs, dim=1)[0] * 100
    classes = pandas.read_csv('bird_dataset.csv', header=None)
    # print("prob: ", float(max(prob)))
    # print("title: ", title[0])
    # print("name: ", classes[0][int(title[0])-1].split('.')[0])
    return (float(max(prob)), classes[0][int(title[0])-1].split('.')[0])


def upload(name, image):
    repo = g.get_user().get_repo("westbengal-species")
    all_files = []
    contents = repo.get_contents("")
    while contents:
        file_content = contents.pop(0)
        if file_content.type == "dir":
            contents.extend(repo.get_contents(file_content.path))
        else:
            file = file_content
            all_files.append(str(file).replace(
                'ContentFile(path="', '').replace('")', ''))

    with open('/tmp/file.txt', 'r') as file:
        content = file.read()

    # Upload to github
    git_prefix = 'folder1/'
    git_file = git_prefix + 'file.txt'
    if git_file in all_files:
        contents = repo.get_contents(git_file)
        repo.update_file(contents.path, "committing files",
                         content, contents.sha, branch="master")
        print(git_file + ' UPDATED')
    else:
        repo.create_file(git_file, "committing files",
                         content, branch="master")
        print(git_file + ' CREATED')
