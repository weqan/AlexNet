import torch
from alexnet import MyAlexNet
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torchvision.transforms import ToPILImage
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

ROOT_TRAIN = r'C:/Users/Hugo/Desktop/python/data/train'
ROOT_TEST = r'C:/Users/Hugo/Desktop/python/data/val'

# 将图像的像素值归一化到[-1, 1]之间
normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

# 训练集
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 论文要求
    transforms.RandomVerticalFlip(),  # 随机垂直全展 让数据集变多
    transforms.ToTensor(),  # 转化为张量
    normalize
])

# 验证集
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 论文要求
    transforms.ToTensor()  # 转化为张量
])

train_dataset = ImageFolder(ROOT_TRAIN, transform=train_transform)
val_dataset = ImageFolder(ROOT_TEST, transform=val_transform)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)

# 运用GUP训练
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = MyAlexNet().to(device)

# 加载模型
model.load_state_dict(torch.load("C:/Users/Hugo/Desktop/python/save_model/best_model.pth"))

# 获取预测结果
classes = [
    "cat",
    "dog",
]

# 把张量转化为照片模式
show = ToPILImage()

# 进入到验证阶段
model.eval()
t, f = 0, 0
for i in range(100):
    x, y = val_dataset[i][0], val_dataset[i][1]
    # show(x).show()
    x = Variable(torch.unsqueeze(x, dim=0).float(), requires_grad=True).to(device)
    x = torch.tensor(x).to(device)
    with torch.no_grad():
        pred = model(x)
        predicted, actual = classes[torch.argmax(pred[0])], classes[y]
        print(f'{i + 1} predicted:"{predicted}“, actual:"{actual}"')
        if predicted is actual:
            t += 1
        else:
            f += 1
    print(f'正确率：{t / (t + f)}')
    # print(f'正确数：{t}')
    # print(f'错误数：{f}')
