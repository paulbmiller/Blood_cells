import torchvision
import torch
import matplotlib.pyplot as plt
from torchvision import transforms


model = torch.hub.load('pytorch/vision', 'resnet50', pretrained=True)
for param in model.parameters():
    param.requires_grad = False

model.fc = torch.nn.Sequential(torch.nn.Linear(2048, 4), torch.nn.Sigmoid())
print(model)

train_data = torchvision.datasets.ImageFolder(
    'dataset2-master/dataset2-master/images/TRAIN',
    transform=torchvision.transforms.Compose(
        [transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor()]))
test_data = torchvision.datasets.ImageFolder(
    'dataset2-master/dataset2-master/images/TEST',
    transform=torchvision.transforms.Compose(
        [transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor()]))

print('Using {} training images'.format(len(train_data)))
print('Using {} test images'.format(len(test_data)))

train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True)

loss_fn = torch.nn.CrossEntropyLoss()
optim = torch.optim.RMSprop(model.fc.parameters(), lr=3e-4)
epochs = 5

model.fc.requires_grad = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = model.to(device)

model.train()
for epoch in range(epochs):
    running_loss = 0.0
    len_loader = len(train_loader)
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optim.zero_grad()
        outputs = model(inputs.to(device))
        loss = loss_fn(outputs.cpu(), labels)
        loss.backward()
        optim.step()

        # print statistics
        running_loss += loss.item()

        if i % 50 == 49:
            print('Epoch {}/{} batch {}/{} loss {:.2f}'.format(epoch + 1, epochs, i + 1, len_loader, running_loss / 50))
            running_loss = 0.0

print('Finished training')

model.eval()

correct = 0
total = 0
with torch.no_grad():
    get_example = True
    for images, labels in test_loader:
        if get_example:
            images_np = images.permute(0, 2, 3, 1).numpy()
        outputs = model(images.to(device))
        predicted = torch.argmax(outputs, dim=1).cpu()
        total += labels.size(0)
        if get_example:
            pred_example = predicted
            label_example = labels
            get_example = False
        correct += (predicted == labels).sum().item()

    print('Accuracy : {:.2f}%'.format(100 * correct / total))

    nrows, ncols = 4, 4
    images_np = images_np[:16]
    pred_example = pred_example[:16]
    label_example = label_example[:16]
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols)
    for i in range(len(images_np)):
        axs[i // ncols, i % ncols].imshow(images_np[i])
        axs[i // ncols, i % ncols].set_title(
            'Predicted: {}, Real: {}'.format(train_data.classes[pred_example[i]], train_data.classes[label_example[i]]))
        axs[i // ncols, i % ncols].axis('off')
    plt.show()


