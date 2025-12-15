import os
import sys
import json

import torch
from torchvision import transforms, datasets
from tqdm import tqdm

import torch.nn as nn
from pytorch_lightning import seed_everything
from torch.utils.data import WeightedRandomSampler
from functools import partial
from utils.random_factor import *
from matplotlib import pyplot as plt
import numpy as np
import logging
import datetime
# from model.biformer.object_detection.mmdet.datasets.builder import worker_init_fn
#from model.HIFuse import HiFuse_Tiny
#from model.other_model import HiFuse_Tiny
#from model.new_model import HiFuse_Tiny
#from model.main_model import HiFuse_Tiny
#from model.Resnet import resnet50
#from model.UniFormer import uniformer_base
#from model.Visual_transformer import vit_base_patch16_224_in21k
#from torchvision.models.swin_transformer import swin_b, swin_t, Swin_T_Weights
#from torchvision.models import convnext_base, resnet34, \
 #vit_b_16, vgg19, efficientnet_b3
#from torchvision.models import mobilenet_v2, efficientnet_b0,vit_b_32
#from model.starnet import starnet_s1
#from model.SwinTransformer import SwimTransformer
#from model.Conformer import Conformer
#from model.new_test_model import HiFuse_Tiny
from model.new_test_model2 import HiFuse_Tiny
#from model.cjy_model1 import HiFuse_Tiny
def main():
    """超参数"""
    # 随机种子
    seed = 42
    # 批处理大小
    batch_size = 32
    # 分类数量
    num_classes = 8
    # 学习率
    lr = 1e-4
    # lr = 1e-5
    # 保存学习率
    lr_arr = []
    # 训练次数
    epochs = 300
    # 最佳结果
    best_acc = 0.0
    # 舍弃分支率
    drop_path_rate = 0.2
    # 中断点
    state_epoch = 0
    # 输入图片大小
    image_size = 224
    # 是否从中断点开始训练
    resume_training = False
    # 是否加载预训练模型
    pretrained = False

    # checkpoint 存放的文件夹
    checkpoint_dir = 'checkpoint'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # log 存放的文件夹
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # accuracy,loss 存放的文件夹
    acc_loss_dir = 'acc_loss'
    if not os.path.exists(acc_loss_dir):
        os.makedirs(acc_loss_dir)

    # 数据集路径
    DATASETS_ROOT = r"F:\code\Dataset\Multi-class texture analysis in colorectal cancer histology"
    #kvasir_dataset   Multi-class texture analysis in colorectal cancer histology ISIC2018  F:\code\Dataset\ISIC2018-tiny
    ROOT_TRAIN = os.path.join(DATASETS_ROOT, 'train')
    ROOT_VAL = os.path.join(DATASETS_ROOT, 'val')

    # 输出当前数据集路径与名称
    print(DATASETS_ROOT)

    # 加载模型
    model = HiFuse_Tiny(num_classes=num_classes)

    # 加载GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 保存当前模型名称
    model_name = model.__class__.__name__

    # 配置日志记录
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y_%m_%d_%H_%M_%S")
    if pretrained:
        formatted_time = f"{formatted_time}_pretrained_{pretrained}"
    log_file = os.path.join(log_dir, f'{model_name}_{formatted_time}_log.txt')
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')

    # 创建当前模型acc与loss保存文件夹
    model_acc_loss_dir = f'{acc_loss_dir}/{model_name}_{formatted_time}'
    if not os.path.exists(model_acc_loss_dir):
        os.makedirs(model_acc_loss_dir)

    # 设置随机种子
    seed_everything(seed)

    print(f"Current model: {model_name}")
    print("using {} device.".format(device))
    print(f"Pretrained is {pretrained}")

    """训练预处理"""
    data_transform = {
        "train": transforms.Compose([
            transforms.Resize((image_size, image_size)),
            # 在线数据增强
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(45),

            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),

        "val": transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    train_dataset = datasets.ImageFolder(root=ROOT_TRAIN, transform=data_transform["train"])

    train_num = len(train_dataset)

    data_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in data_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=1)
    with open('../class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    # 创建 DataLoader 时传入 WeightedRandomSampler
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               worker_init_fn=partial(worker_init_fn, rank=0, seed=seed))

    validate_dataset = datasets.ImageFolder(root=ROOT_VAL, transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=batch_size, shuffle=False,
                                                  worker_init_fn=partial(worker_init_fn, rank=0, seed=seed))

    print("using {} images for training, {} images for validate".format(train_num, val_num))

    if resume_training:
        print("The current task is recovery training")

    model.to(device)

    # define loss function
    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    #正则化
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=5e-2)

    # 训练策略
    # scheduler = CosineLRScheduler(optimizer=optimizer,
    #                               t_initial=epochs,
    #                               lr_min=5e-6,
    #                               warmup_t=4,
    #                               warmup_lr_init=1e-4)

    train_steps = len(train_dataset)

    if resume_training:
        checkpoint = torch.load(os.path.join(checkpoint_dir, f'{model_name}_{formatted_time}_checkpoint.pth'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        state_epoch = checkpoint['epoch']

    """训练"""
    train_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(state_epoch, epochs):
        lr_arr.append(optimizer.param_groups[0]['lr'])

        # train
        model.train()

        running_loss = 0.0
        correct_train = 0.0

        train_bar = tqdm(train_loader, file=sys.stdout)

        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = model(images.to(device))

            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()

            predict_y = torch.max(logits, dim=1)[1]
            correct_train += torch.eq(predict_y, labels.to(device)).sum().item()
            # print statistics
            running_loss += loss.item()

            # scheduler.step(epoch)

            train_bar.desc = "state_epoch:{} train epoch[{}/{}] loss:{:.5f} lr:{:.6f}".format(
                state_epoch + 1, epoch + 1, epochs, loss, optimizer.param_groups[0]['lr'])

            # 记录训练日志
            logging.info("state_epoch:{} train epoch[{}/{}] loss:{:.5f} lr:{:.6f}".format(
                state_epoch + 1, epoch + 1, epochs, loss, optimizer.param_groups[0]['lr']))

        train_loss = running_loss / len(train_loader)
        train_accurate = correct_train / train_num
        train_losses.append(train_loss)
        train_accuracies.append(train_accurate)

        # validate
        model.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data, val_labels in val_bar:
                outputs = model(val_data.to(device))
                # loss = loss_function(outputs, test_labels)
                predict_y = torch.max(outputs, dim=1)[1]

                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                           epochs)

        val_accurate = acc / val_num
        val_accuracies.append(val_accurate)
        print('[epoch %d] train_loss: %.5f  val_accuracy: %.5f  train_accuracy: %.5f' %
              (epoch + 1, running_loss / train_steps, val_accurate, train_accurate))
        logging.info('[epoch %d] train_loss: %.5f  val_accuracy: %.5f' %
                     (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            # 保存训练参数
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                # 'scheduler_state_dict': scheduler.state_dict(),
            }, os.path.join(checkpoint_dir, f'{model_name}_{formatted_time}_checkpoint.pth'))

    torch.save(train_losses, f'{model_acc_loss_dir}/{model_name}_train_losses.txt')
    torch.save(train_accuracies, f'{model_acc_loss_dir}/{model_name}_train_accuracies.txt')
    torch.save(val_accuracies, f'{model_acc_loss_dir}/{model_name}_val_accuracies.txt')

    print('Finished Training')
    print('训练精确度', train_accurate)
    print('最好的验证精确度', best_acc)
    plt.plot(np.arange(epochs), lr_arr)
    plt.show()

    # 绘制损失曲线

    # 绘制精度曲线
    plt.plot(train_losses, label='Training Loss')
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
