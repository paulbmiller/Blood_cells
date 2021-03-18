import torch
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
import torchvision


"""
model = ResNet50(include_top=True, weights='imagenet', input_shape=(224, 224, 3))
# plot_model(model, 'resnet50.png')
with open('model_summary_w_top.txt','w') as fh:
    model.summary(print_fn=lambda x: fh.write(x + '\n'))
"""

model = torch.hub.load('pytorch/vision', 'resnet50', pretrained=True)


class ConvBlock(torch.nn.Module):
    def __init__(self, in_feat, fm_sizes, conv_input=False, stride=1):
        super(ConvBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_feat, fm_sizes[0], (1, 1), stride=stride, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(fm_sizes[0])
        self.conv2 = torch.nn.Conv2d(fm_sizes[0], fm_sizes[1], (3, 3), stride=1, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(fm_sizes[1])
        self.conv3 = torch.nn.Conv2d(fm_sizes[1], fm_sizes[2], (1, 1), bias=False)
        self.bn3 = torch.nn.BatchNorm2d(fm_sizes[2])
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv_input = conv_input
        if self.conv_input:
            self.conv0 = torch.nn.Conv2d(in_feat, fm_sizes[2], (1, 1), stride=stride, bias=False)
            self.bn0 = torch.nn.BatchNorm2d(fm_sizes[2])

    def forward(self, x):
        if not self.conv_input:
            prev_out = x
        else:
            prev_out = self.conv0(x)
            prev_out = self.bn0(prev_out)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.bn3(x)

        return self.relu(x + prev_out)


class Res50(torch.nn.Module):
    def __init__(self):
        super(Res50, self).__init__()
        self.conv1_conv = torch.nn.Conv2d(3, 64, (7, 7), 2, 3, bias=False)
        self.conv1_bn = torch.nn.BatchNorm2d(64)
        self.conv1_relu = torch.nn.ReLU(inplace=True)
        self.conv1_pool = torch.nn.MaxPool2d((3, 3), stride=2, padding=1)

        self.conv2_block1 = ConvBlock(64, [64, 64, 256], True)
        self.conv2_block2 = ConvBlock(256, [64, 64, 256])
        self.conv2_block3 = ConvBlock(256, [64, 64, 256])

        self.conv3_block1 = ConvBlock(256, [128, 128, 512], True, 2)
        self.conv3_block2 = ConvBlock(512, [128, 128, 512])
        self.conv3_block3 = ConvBlock(512, [128, 128, 512])
        self.conv3_block4 = ConvBlock(512, [128, 128, 512])

        self.conv4_block1 = ConvBlock(512, [256, 256, 1024], True, 2)
        self.conv4_block2 = ConvBlock(1024, [256, 256, 1024])
        self.conv4_block3 = ConvBlock(1024, [256, 256, 1024])
        self.conv4_block4 = ConvBlock(1024, [256, 256, 1024])
        self.conv4_block5 = ConvBlock(1024, [256, 256, 1024])
        self.conv4_block6 = ConvBlock(1024, [256, 256, 1024])

        self.conv5_block1 = ConvBlock(1024, [512, 512, 2048], True, 2)
        self.conv5_block2 = ConvBlock(2048, [512, 512, 2048])
        self.conv5_block3 = ConvBlock(2048, [512, 512, 2048])

        self.avg_pool = torch.nn.AvgPool2d((7, 7))

        self.fc = torch.nn.Sequential(torch.nn.Linear(2048, 4), torch.nn.Sigmoid())

    def forward(self, x):
        x = self.conv1_conv(x)
        x = self.conv1_bn(x)
        x = self.conv1_relu(x)
        x = self.conv1_pool(x)
        x = self.conv2_block1(x)
        x = self.conv2_block2(x)
        x = self.conv2_block3(x)

        x = self.conv3_block1(x)
        x = self.conv3_block2(x)
        x = self.conv3_block3(x)
        x = self.conv3_block4(x)

        x = self.conv4_block1(x)
        x = self.conv4_block2(x)
        x = self.conv4_block3(x)
        x = self.conv4_block4(x)
        x = self.conv4_block5(x)
        x = self.conv4_block6(x)

        x = self.conv5_block1(x)
        x = self.conv5_block2(x)
        x = self.conv5_block3(x)

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


if __name__ == '__main__':
    res = Res50()

    sum_params = 0
    for name, param in res.named_parameters():
        if param.requires_grad:
            len_params = np.prod(np.array(param.shape))
            # print(name, len_params)
            sum_params += len_params
            # print(name, param.shape)
    print('Total trainable parameters', sum_params)

    model_state_dict = model.state_dict()
    res_state_dict = res.state_dict()

    with torch.no_grad():
        for (n1, param1), (n2, param2) in list(zip(res.named_parameters(), model.named_parameters()))[:-2]:
            res_state_dict = res.state_dict()
            res_state_dict[n1].data.copy_(model_state_dict[n2].data)

    res.load_state_dict(res_state_dict)

    for param in res.parameters():
        param.requires_grad = False
    for param in res.fc.parameters():
        param.requires_grad = True

    train_data = torchvision.datasets.ImageFolder('dataset2-master/dataset2-master/images/TRAIN',
        transform=torchvision.transforms.Compose(
            [transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor()]))
    test_data = torchvision.datasets.ImageFolder('dataset2-master/dataset2-master/images/TEST',
        transform=torchvision.transforms.Compose(
            [transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor()]))

    print('Using {} training images'.format(len(train_data)))
    print('Using {} test images'.format(len(test_data)))

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True)

    loss_fn = torch.nn.CrossEntropyLoss()
    optim = torch.optim.RMSprop(res.fc.parameters(), lr=3e-4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    res.to(device)

    epochs = 1
    res.train()

    for epoch in range(epochs):
        running_loss = 0.0
        len_loader = len(train_loader)
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            optim.zero_grad()
            outputs = res(inputs.to(device))
            loss = loss_fn(outputs.cpu(), labels)
            loss.backward()
            optim.step()

            # print statistics
            running_loss += loss.item()

            if i % 50 == 49:
                print('Epoch {}/{} batch {}/{} loss {:.2f}'.format(epoch + 1, epochs, i + 1, len_loader, running_loss / 50))
                running_loss = 0.0

    print('Finished training')

    res.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        get_example = True
        for images, labels in test_loader:
            if get_example:
                images_np = images.permute(0, 2, 3, 1).numpy()
            outputs = res(images.to(device))
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
            axs[i // ncols, i % ncols].set_title('Predicted: {}, Real: {}'.format(train_data.classes[pred_example[i]],
                                                                                  train_data.classes[label_example[i]]))
            axs[i // ncols, i % ncols].axis('off')
        plt.show()
