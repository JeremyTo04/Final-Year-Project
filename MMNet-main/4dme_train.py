import math
import numpy as np
import torchvision.models
import torch.utils.data as data
from torchvision import transforms
import cv2
import torch.nn.functional as F
from torch.autograd import Variable
import pandas as pd
import os, torch
import torch.nn as nn
import time
#import image_utils
import argparse, random
from functools import partial

# from torchvision.models import resnet50

from CA_block import resnet18_pos_attention

from PC_module import VisionTransformer_POS

from torchvision.transforms import Resize
torch.set_printoptions(precision=3, edgeitems=14, linewidth=350)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raf_path', type=str, default='/content/Final-Year-Project/MMNet-main/4dme', help='Raf-DB dataset path.')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Pytorch checkpoint file path')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Pretrained weights')
    parser.add_argument('--beta', type=float, default=0.7, help='Ratio of high importance group in one mini-batch.')
    parser.add_argument('--relabel_epoch', type=int, default=1000,
                        help='Relabeling samples on each mini-batch after 10(Default) epochs.')
    parser.add_argument('--batch_size', type=int, default=34, help='Batch size.')
    parser.add_argument('--optimizer', type=str, default="adam", help='Optimizer, adam or sgd.')
    parser.add_argument('--lr', type=float, default=0.0001, help='Initial learning rate for sgd.')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum for sgd')
    parser.add_argument('--workers', default=0, type=int, help='Number of data loading workers (default: 4)')
    parser.add_argument('--epochs', type=int, default=20, help='Total training epochs.')
    parser.add_argument('--drop_rate', type=float, default=0, help='Drop out rate.')
    return parser.parse_args()


class RafDataSet(data.Dataset):
    def __init__(self, raf_path, phase,num_loso, transform = None, basic_aug = False, transform_norm=None):
        self.phase = phase
        self.transform = transform
        self.raf_path = raf_path
        self.transform_norm = transform_norm
        SUBJECT_COLUMN =0
        NAME_COLUMN = 1
        ONSET_COLUMN = 2
        APEX_COLUMN = 3
        OFF_COLUMN = 4
        LABEL_AU_COLUMN = 5
        LABEL_ALL_COLUMN = 6


        df = pd.read_excel(os.path.join('4dme', 'MMnet_4DME.xlsx'), usecols=[0, 1, 3, 4, 5, 7, 8])
        #print(df)
        df['Subject'] = df['Subject'].apply(str)

        if phase == 'train':
            dataset = df.loc[df['Subject']!=num_loso]
        else:
            dataset = df.loc[df['Subject'] == num_loso]

        Subject = dataset.iloc[:, SUBJECT_COLUMN].values
        File_names = dataset.iloc[:, NAME_COLUMN].values
        Label_all = dataset.iloc[:, LABEL_ALL_COLUMN].values  # 0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral
        #print(Label_all)
        Onset_num = dataset.iloc[:, ONSET_COLUMN].values
        Apex_num = dataset.iloc[:, APEX_COLUMN].values
        Offset_num = dataset.iloc[:, OFF_COLUMN].values
        Label_au = dataset.iloc[:, LABEL_AU_COLUMN].values
        self.file_paths_on = []
        self.file_paths_off = []
        self.file_paths_apex = []
        self.label_all = []
        self.label_au = []
        self.sub= []
        self.file_names =[]
        a=0
        b=0
        c=0
        d=0
        e=0
        # use aligned images for training/testing
        for (f,sub,onset,apex,offset,label_all,label_au) in zip(File_names,Subject,Onset_num,Apex_num,Offset_num,Label_all,Label_au):


            if label_all == 'Positive' or label_all == 'Positive+Repression' or label_all == 'Surprise+Negative' or label_all == 'Surprise+Positive' or label_all == 'Repression' or label_all == 'Surprise+Repression' or label_all == 'surprise' or label_all == 'Negative' or label_all == 'Negative+Repression':

                self.file_paths_on.append(onset)
                self.file_paths_off.append(offset)
                self.file_paths_apex.append(apex)
                self.sub.append(sub)
                self.file_names.append(f)
                if label_all == 'Positive' or label_all == 'Positive+Repression':
                    self.label_all.append(0)
                    a=a+1
                elif label_all == 'surprise' or label_all == 'Surprise+Repression' or label_all == 'Surprise+Negative' or label_all == 'Surprise+Positive':
                    self.label_all.append(1)
                    b=b+1
                else:
                    self.label_all.append(2)
                    c=c+1

            # label_au =label_au.split("+")
                if isinstance(label_au, int):
                    self.label_au.append([label_au])
                else:
                    label_au = label_au.split("+")
                    self.label_au.append(label_au)

            ##label

        self.basic_aug = basic_aug
        #self.aug_func = [image_utils.flip_image,image_utils.add_gaussian_noise]

    def __len__(self):
        return len(self.file_paths_on)

    def __getitem__(self, idx):
        ##sampling strategy for training set
        if self.phase == 'train':
            onset = self.file_paths_on[idx]
            apex = self.file_paths_apex[idx]
            offset =self.file_paths_off[idx]
            on0 = str(random.randint(int(onset), int(onset + int(0.2* (apex - onset) / 4))))
            # on0 = str(int(onset))
            #print(on0)
            on1 = str(
                random.randint(int(onset + int(0.9 * (apex - onset) / 4)), int(onset + int(1.1 * (apex - onset) / 4))))
            on2 = str(
                random.randint(int(onset + int(1.8 * (apex - onset) / 4)), int(onset + int(2.2 * (apex - onset) / 4))))
            on3 = str(random.randint(int(onset + int(2.7 * (apex - onset) / 4)), onset + int(3.3 * (apex - onset) / 4)))
            # apex0 = str(apex)
            apex0 = str(
                random.randint(int(apex - int(0.15* (apex - onset) / 4)), apex + int(0.15 * (offset - apex) / 4)))
            off0 = str(
                random.randint(int(apex + int(0.9 * (offset - apex) / 4)), int(apex + int(1.1 * (offset - apex) / 4))))
            off1 = str(
                random.randint(int(apex + int(1.8 * (offset - apex) / 4)), int(apex + int(2.2 * (offset - apex) / 4))))
            off2 = str(
                random.randint(int(apex + int(2.9 * (offset - apex) / 4)), int(apex + int(3.1 * (offset - apex) / 4))))
            off3 = str(random.randint(int(apex + int(3.8 * (offset - apex) / 4)), offset))



            sub =str(self.sub[idx])
            f = str(self.file_names[idx])
        else:##sampling strategy for testing set
            onset = self.file_paths_on[idx]
            apex = self.file_paths_apex[idx]
            offset = self.file_paths_off[idx]

            on0 = str(onset)
            on1 = str(int(onset + int((apex - onset) / 4)))
            on2 = str(int(onset + int(2 * (apex - onset) / 4)))
            on3 = str(int(onset + int(3 * (apex - onset) / 4)))
            apex0 = str(apex)
            off0 = str(int(apex + int((offset - apex) / 4)))
            off1 = str(int(apex + int(2 * (offset - apex) / 4)))
            off2 = str(int(apex + int(3 * (offset - apex) / 4)))
            off3 = str(offset)

            sub = str(self.sub[idx])
            f = str(self.file_names[idx])


        on0 = 'Frame_' + on0.zfill(9) + '.jpg'
        on1 = 'Frame_' + on1.zfill(9) + '.jpg'
        on2 = 'Frame_' + on2.zfill(9) + '.jpg'
        on3 = 'Frame_' + on3.zfill(9) + '.jpg'
        apex0 = 'Frame_' + apex0.zfill(9) + '.jpg'
        off0 = 'Frame_' + off0.zfill(9) + '.jpg'
        off1 = 'Frame_' + off1.zfill(9) + '.jpg'
        off2 = 'Frame_' + off2.zfill(9) + '.jpg'
        off3 = 'Frame_' + off3.zfill(9) + '.jpg'


        path_on0 = os.path.join(self.raf_path, 'micro_short_gray_video/micro short gray video/micro short gray video', sub, f, on0)
        #print(f"path: {path_on0}")
        path_on0 = path_on0.replace('\\', '/')
        #print(f"path: {path_on0}")   # CHANGED
        path_on1 = os.path.join(self.raf_path, 'micro_short_gray_video/micro short gray video/micro short gray video', sub, f, on1)
        path_on1 = path_on0.replace('\\', '/')
        
        path_on2 = os.path.join(self.raf_path, 'micro_short_gray_video/micro short gray video/micro short gray video', sub, f, on2)
        path_on2 = path_on0.replace('\\', '/')
        
        path_on3 = os.path.join(self.raf_path, 'micro_short_gray_video/micro short gray video/micro short gray video', sub, f, on3)
        path_on3 = path_on0.replace('\\', '/')
        
        path_apex0 = os.path.join(self.raf_path, 'micro_short_gray_video/micro short gray video/micro short gray video', sub, f, apex0)
        path_apex0 = path_on0.replace('\\', '/')
        
        path_off0 = os.path.join(self.raf_path, 'micro short gray video/micro short gray video/micro short gray video', sub, f, off0)
        path_off0 = path_on0.replace('\\', '/')
        
        path_off1 = os.path.join(self.raf_path, 'micro_short_gray_video/micro short gray video/micro short gray video', sub, f, off1)
        path_off1 = path_on0.replace('\\', '/')
        
        path_off2 = os.path.join(self.raf_path, 'micro_short_gray_video/micro short gray video/micro short gray video', sub, f, off2)
        path_off2 = path_on0.replace('\\', '/')
        
        path_off3 = os.path.join(self.raf_path, 'micro_short_gray_video/micro short gray video/micro short gray video', sub, f, off3)
        path_off3 = path_on0.replace('\\', '/')
        image_on0 = cv2.imread(path_on0)
        image_on1= cv2.imread(path_on1)
        image_on2 = cv2.imread(path_on2)
        image_on3 = cv2.imread(path_on3)
        image_apex0 = cv2.imread(path_apex0)
        image_off0 = cv2.imread(path_off0)
        image_off1 = cv2.imread(path_off1)
        image_off2 = cv2.imread(path_off2)
        image_off3 = cv2.imread(path_off3)

        image_on0 = image_on0[:, :, ::-1] # BGR to RGB
        image_on1 = image_on1[:, :, ::-1]
        image_on2 = image_on2[:, :, ::-1]
        image_on3 = image_on3[:, :, ::-1]
        image_off0 = image_off0[:, :, ::-1]
        image_off1 = image_off1[:, :, ::-1]
        image_off2 = image_off2[:, :, ::-1]
        image_off3 = image_off3[:, :, ::-1]
        image_apex0 = image_apex0[:, :, ::-1]

        label_all = self.label_all[idx]
        label_au = self.label_au[idx]

        # normalization for testing and training
        if self.transform is not None:
            image_on0 = self.transform(image_on0)
            image_on1 = self.transform(image_on1)
            image_on2 = self.transform(image_on2)
            image_on3 = self.transform(image_on3)
            image_off0 = self.transform(image_off0)
            image_off1 = self.transform(image_off1)
            image_off2 = self.transform(image_off2)
            image_off3 = self.transform(image_off3)
            image_apex0 = self.transform(image_apex0)
            ALL = torch.cat(
                (image_on0, image_on1, image_on2, image_on3, image_apex0, image_off0, image_off1, image_off2,
                 image_off3), dim=0)
            ## data augmentation for training only
            if self.transform_norm is not None and self.phase == 'train':
                ALL = self.transform_norm(ALL)
            image_on0 = ALL[0:3, :, :]
            image_on1 = ALL[3:6, :, :]
            image_on2 = ALL[6:9, :, :]
            image_on3 = ALL[9:12, :, :]
            image_apex0 = ALL[12:15, :, :]
            image_off0 = ALL[15:18, :, :]
            image_off1 = ALL[18:21, :, :]
            image_off2 = ALL[21:24, :, :]
            image_off3 = ALL[24:27, :, :]


           
            max_len = 45  
            temp = torch.zeros(max_len)

            for i in label_au:
                idx = int(i) - 1
                if 0 <= idx < max_len:
                    temp[idx] = 1

            return image_on0, image_on1, image_on2, image_on3, image_apex0, image_off0, image_off1, image_off2, image_off3, label_all, f, temp



def initialize_weight_goog(m, n=''):
    if isinstance(m, nn.Conv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        fan_out = m.weight.size(0)  # fan-out
        fan_in = 0
        if 'routing_fn' in n:
            fan_in = m.weight.size(1)
        init_range = 1.0 / math.sqrt(fan_in + fan_out)
        m.weight.data.uniform_(-init_range, init_range)
        m.bias.data.zero_()


def criterion2(y_pred, y_true):
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat((y_pred_neg, zeros), dim=-1)
    y_pred_pos = torch.cat((y_pred_pos, zeros), dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    return torch.mean(neg_loss + pos_loss)


class MMNet(nn.Module):
    def __init__(self):
        super(MMNet, self).__init__()

        self.conv_act = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=90*2, kernel_size=3, stride=2,padding=1, bias=False,groups=1),
            nn.BatchNorm2d(180),
            nn.ReLU(inplace=True),
            )
        self.pos =nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=512, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            )
        ##Position Calibration Module(subbranch)
        self.vit_pos=VisionTransformer_POS(img_size=14,
        patch_size=1, embed_dim=512, depth=3, num_heads=4, mlp_ratio=2, qkv_bias=True,norm_layer=partial(nn.LayerNorm, eps=1e-6),drop_path_rate=0.3)
        self.resize=Resize([14,14])
        ##main branch consisting of CA blocks
        self.main_branch =resnet18_pos_attention()
        self.head1 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(1 * 112 *112, 38,bias=False),

        )

        self.timeembed = nn.Parameter(torch.zeros(1, 4, 111, 111))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    def forward(self, x1, x2, x3, x4, x5, x6, x7, x8, x9, if_shuffle):
        ##onset:x1 apex:x5
        B = x1.shape[0]

        #Position Calibration Module (subbranch)
        POS =self.vit_pos(self.resize(x1)).transpose(1,2).view(B,512,14,14)
        act = x5 -x1
        act=self.conv_act(act)
        #main branch and fusion
        out,_=self.main_branch(act,POS)

        return out



def run_training():

    args = parse_args()
    imagenet_pretrained = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    weight_path = r"\content\Final-Year-Project\MMNet-main\4dme\paths"
    if not imagenet_pretrained:
        for m in res18.modules():
            initialize_weight_goog(m)

    if args.pretrained:
        print("Loading pretrained weights...", args.pretrained)
        pretrained = torch.load(args.pretrained)
        pretrained_state_dict = pretrained['state_dict']
        model_state_dict = res18.state_dict()
        loaded_keys = 0
        total_keys = 0
        for key in pretrained_state_dict:
            if ((key == 'module.fc.weight') | (key == 'module.fc.bias')):
                pass
            else:
                model_state_dict[key] = pretrained_state_dict[key]
                total_keys += 1
                if key in model_state_dict:
                    loaded_keys += 1
        print("Loaded params num:", loaded_keys)
        print("Total params num:", total_keys)
        res18.load_state_dict(model_state_dict, strict=False)
    ### data normalization for both training set
    data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),

        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),

    ])
    ### data augmentation for training set only
    data_transforms_norm = transforms.Compose([

        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(4),
        transforms.RandomCrop(224, padding=4),


    ])


    ### data normalization for both teating set
    data_transforms_val = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])



    criterion = torch.nn.CrossEntropyLoss()
    # leave one subject out protocal #, ('S58_2'), , ('S43')]
    LOSO = [ 'S59', 'S60', 'S61', 'S63', 'S64', 'S65', 'S08', 'S09', 'S10', 'S12', 'S13_1', 'S13_2','S14', 'S16','S17','S18', 'S20', 'S21', 'S22', 'S24', 'S25',
    'S27', 'S30_1', 'S30_2', 'S31', 'S33', 'S34','S36', 'S37', 'S38', 'S41', 'S42','S45', 'S46', 'S49', 'S50_1', 'S51','S52_2', 'S53', 'S55', 'S56',
    'S57', 'S58_1']  
    # LOSO = [ 'S59', 'S60'] 

    val_now = 0
    num_sum = 0
    pos_pred_ALL = torch.zeros(3)
    pos_label_ALL = torch.zeros(3)
    TP_ALL = torch.zeros(3)
    
    ##model initialization
    net_all = MMNet()
    model_save_path = f'model_weights.pth' 

    # Check if the model weights file exists
    if os.path.exists(model_save_path):
            # Load the saved weights
            net_all.load_state_dict(torch.load(model_save_path))
            net_all.eval()  # Set the model to evaluation mode
            print(f'Loaded saved model weights from {model_save_path}')
    else:
            # Proceed with training since weights are not available
            print(f'No saved model weights found. Starting training.')
    
    for subj in LOSO:
        all_data = []
        print(subj)
        train_dataset = RafDataSet(args.raf_path, phase='train', num_loso=subj, transform=data_transforms,
                                   basic_aug=True, transform_norm=data_transforms_norm)
        if len(train_dataset) == 0:
          raise ValueError("The training dataset is empty. Please check the dataset paths and filtering logic.")
        val_dataset = RafDataSet(args.raf_path, phase='test', num_loso=subj, transform=data_transforms_val)
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=24,
                                                   num_workers=args.workers,
                                                   shuffle=True,
                                                   pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=24,
                                                 num_workers=args.workers,
                                                 shuffle=False,
                                                 pin_memory=True)
        print('num_sub', subj)
        print('Train set size:', train_dataset.__len__())
        print('Validation set size:', val_dataset.__len__())

        max_corr = 0
        max_f1 = 0
        max_pos_pred = torch.zeros(3)
        max_pos_label = torch.zeros(3)
        max_TP = torch.zeros(3)
        ##model initialization
        net_all = MMNet()

        params_all = net_all.parameters()
        
        # model_save_path = f'model_weights_4dme_subject_{subj}.pth'
        # # model_save_path = 'train_all.pth'   

        # # Check if the model weights file exists
        # if os.path.exists(model_save_path):
        #     # Load the saved weights
        #     net_all.load_state_dict(torch.load(model_save_path))
        #     net_all.eval()  # Set the model to evaluation mode
        #     print(f'Loaded saved model weights for 4dme subject {subj} from {model_save_path}')
        # else:
        #     # Proceed with training since weights are not available
        #     print(f'No saved model weights found for 4dme subject {subj}. Starting training.')


        if args.optimizer == 'adam':
            optimizer_all = torch.optim.AdamW(params_all, lr=0.0008, weight_decay=0.7)
            ##optimizer for MMNet

        elif args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(params, args.lr,
                                        momentum=args.momentum,
                                        weight_decay=1e-4)
        else:
            raise ValueError("Optimizer not supported.")
        ##lr_decay
        scheduler_all = torch.optim.lr_scheduler.ExponentialLR(optimizer_all, gamma=0.987)

        net_all = net_all.cuda()

        # # Define the directory where you want to save the model
        # weight_dir = '/content/Final-Year-Project/MMNet-main/4dme/4dme paths'

        # # Ensure the directory exists
        # if not os.path.exists(weight_dir):
        #     os.makedirs(weight_dir)

        # # Define the full path for saving the model
        # weight_path = os.path.join(weight_dir, 'model_weights.pth')

        # # Now you can save the model without the error
        # torch.save(net_all.state_dict(), weight_path)
    

        for i in range(1, 10): # changed number of epochs to 10 instead of 100
            running_loss = 0.0
            correct_sum = 0
            running_loss_MASK = 0.0
            correct_sum_MASK = 0
            iter_cnt = 0

            net_all.train()

            predictions = []
            dataset_idx = 0 
            for batch_i, (
            image_on0, image_on1, image_on2, image_on3, image_apex0, image_off0, image_off1, image_off2, image_off3,
            label_all, f,
            label_au) in enumerate(train_loader):
                batch_sz = image_on0.size(0)
                b, c, h, w = image_on0.shape
                iter_cnt += 1

                image_on0 = image_on0.cuda()
                image_on1 = image_on1.cuda()
                image_on2 = image_on2.cuda()
                image_on3 = image_on3.cuda()
                image_apex0 = image_apex0.cuda()
                image_off0 = image_off0.cuda()
                image_off1 = image_off1.cuda()
                image_off2 = image_off2.cuda()
                image_off3 = image_off3.cuda()
                label_all = label_all.cuda()
                label_au = label_au.cuda()


                ##train MMNet
                ALL = net_all(image_on0, image_on1, image_on2, image_on3, image_apex0, image_off0, image_off1,
                                   image_off2, image_off3, False)

                loss_all = criterion(ALL, label_all)

                optimizer_all.zero_grad()

                loss_all.backward()

                optimizer_all.step()
                running_loss += loss_all
                _, predicts = torch.max(ALL, 1)
                correct_num = torch.eq(predicts, label_all).sum()
                correct_sum += correct_num


            ## lr decay
            if i <= 50:

                scheduler_all.step()
            if i>=0:
                acc = correct_sum.float() / float(train_dataset.__len__())

                running_loss = running_loss / iter_cnt

                print('[Epoch %d] Training accuracy: %.4f. Loss: %.3f' % (i, acc, running_loss))


            pos_label = torch.zeros(3)
            pos_pred = torch.zeros(3)
            TP = torch.zeros(3)
            ##test
            with torch.no_grad():
                running_loss = 0.0
                iter_cnt = 0
                bingo_cnt = 0
                sample_cnt = 0
                pre_lab_all = []
                Y_test_all = []
                net_all.eval()
                for batch_i, (
                image_on0, image_on1, image_on2, image_on3, image_apex0, image_off0, image_off1, image_off2,
                image_off3, label_all, f,
                label_au) in enumerate(val_loader):
                    batch_sz = image_on0.size(0)
                    b, c, h, w = image_on0.shape

                    image_on0 = image_on0.cuda()
                    image_on1 = image_on1.cuda()
                    image_on2 = image_on2.cuda()
                    image_on3 = image_on3.cuda()
                    image_apex0 = image_apex0.cuda()
                    image_off0 = image_off0.cuda()
                    image_off1 = image_off1.cuda()
                    image_off2 = image_off2.cuda()
                    image_off3 = image_off3.cuda()
                    label_all = label_all.cuda()
                    label_au = label_au.cuda()

                    ##test
                    ALL = net_all(image_on0, image_on1, image_on2, image_on3, image_apex0, image_off0, image_off1, image_off2, image_off3, False)
                    

                    loss = criterion(ALL, label_all)
                    running_loss += loss
                    iter_cnt += 1
                    _, predicts = torch.max(ALL, 1)
                    correct_num = torch.eq(predicts, label_all)
                    bingo_cnt += correct_num.sum().cpu()
                    sample_cnt += ALL.size(0)
                    # Loop through each prediction and label, and store them in the list
                    for j in range(predicts.size(0)):
                      all_data.append([f[j], predicts[j].item(), label_all[j].item()])

                    for cls in range(3):
                        for element in predicts:
                            if element == cls:
                                pos_label[cls] = pos_label[cls] + 1
                        for element in label_all:
                            if element == cls:
                                pos_pred[cls] = pos_pred[cls] + 1
                        for elementp, elementl in zip(predicts, label_all):
                            if elementp == elementl and elementp == cls:
                                TP[cls] = TP[cls] + 1

                    count = 0
                    SUM_F1 = 0
                    for index in range(3):
                        if pos_label[index] != 0 or pos_pred[index] != 0:
                            count = count + 1
                            SUM_F1 = SUM_F1 + 2 * TP[index] / (pos_pred[index] + pos_label[index])

                    AVG_F1 = SUM_F1 / count


                running_loss = running_loss / iter_cnt
                acc = bingo_cnt.float() / float(sample_cnt)
                acc = np.around(acc.numpy(), 4)
                if bingo_cnt > max_corr:
                    max_corr = bingo_cnt
                if AVG_F1 >= max_f1:
                    max_f1 = AVG_F1
                    max_pos_label = pos_label
                    max_pos_pred = pos_pred
                    max_TP = TP
                print("[Epoch %d] Validation accuracy:%.4f. Loss:%.3f, F1-score:%.3f" % (i, acc, running_loss, AVG_F1))
          
        num_sum = num_sum + max_corr
        pos_label_ALL = pos_label_ALL + max_pos_label
        pos_pred_ALL = pos_pred_ALL + max_pos_pred
        TP_ALL = TP_ALL + max_TP
        count = 0
        SUM_F1 = 0
        for index in range(3):
            if pos_label_ALL[index] != 0 or pos_pred_ALL[index] != 0:
                count = count + 1
                SUM_F1 = SUM_F1 + 2 * TP_ALL[index] / (pos_pred_ALL[index] + pos_label_ALL[index])

        F1_ALL = SUM_F1 / count
        val_now = val_now + val_dataset.__len__()
        print("[..........%s] correctnum:%d . zongshu:%d   " % (subj, max_corr, val_dataset.__len__()))
        print("[ALL_corr]: %d [ALL_val]: %d" % (num_sum, val_now))
        print("[F1_now]: %.4f [F1_ALL]: %.4f" % (max_f1, F1_ALL))

        
        # df = pd.DataFrame(all_data, columns=['filename', 'Predicted Label', 'Actual Label'])
        # # Save the DataFrame to an Excel file
        # file_name = f'predictions_and_labels_{subj}.xlsx'
        # print(f"saving predictions for {subj} to {file_name}")
        # results_dir = r"/content/Final-Year-Project/MMNet-main/4dme/micro_short_gray_video/micro short gray video/micro short gray video"
        # if not os.path.exists(results_dir):
        #     os.makedirs(results_dir)
        # file_name = os.path.join(results_dir, f'predictions_and_labels_{subj}.xlsx')
        # df.to_excel(file_name, index=False)
        # # Check if the file has been saved successfully
        # if os.path.exists(file_name):
        #     print(f"File saved successfully: {file_name}")
        # else:
        #     print("Failed to save the file.")
        
    # After processing all subjects, save the model weights
    torch.save(net_all.state_dict(), model_save_path)
    print(f'Model weights saved at {model_save_path}')
    
    return num_sum, pos_label_ALL, pos_pred_ALL, TP_ALL

        
import os

import os
def rename_images_in_subfolders(base_path):
    # Traverse through the directory structure
    for root, dirs, files in os.walk(base_path):
        for file in files:
            # Print the current file being checked
            print(f'Checking file: {file}')
            # Check if the file starts with 'frame_' (case-sensitive)
            if file.startswith('frame_'):
                # Construct the old file path
                old_file_path = os.path.join(root, file)
                # Construct the new file name and path
                new_file_name = file.replace('frame_', 'Frame_', 1)
                new_file_path = os.path.join(root, new_file_name)
                
                # Rename the file
                os.rename(old_file_path, new_file_path)
                print(f'Renamed: {old_file_path} to {new_file_path}')
            else:
                print(f'Not renamed: {file}')





if __name__ == "__main__":
    run_training()
    # base_path = r'C:\Users\alyss\OneDrive\Documents\GitHub\Final-Year-Project\MMNet-main\4dme\micro_short_gray_video\micro short gray video\micro short gray video'
    # rename_images_in_subfolders(base_path)
        
