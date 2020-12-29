import os, time, argparse, shutil
from collections import defaultdict, Counter
from datetime import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch import nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
from tensorboardX import SummaryWriter

from data_load import *
from model import CPM

def train(epoch, model, criterion, optimizer, train_loader, writer, iters):
    
    model.train()
    
    for batch_idx, (img, target) in enumerate(train_loader):
        
        img, target = Variable(img.cuda()), Variable(target.cuda())
        img, target = img.type(torch.cuda.FloatTensor), target.type(torch.cuda.FloatTensor)

        losses = []
        for stage in range(1, args.stages + 1):
            if stage == 1:
                inputs = img
            else:
                inputs = torch.cat([img, outputs], axis=1)
            outputs = model(inputs, stage=stage)
            loss = criterion(outputs, target)
            losses.append(loss)
        
        loss = sum(losses)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch_idx % args.log_interval == 0 and not batch_idx==0 :
            print('Train Epoch:{}/{} [{}/{} ({:.0f}%)]  Loss:{:.4f}'.format(
                epoch + 1, args.epochs,
                batch_idx * len(img), len(train_loader) * len(img),
                100.0 * batch_idx / len(train_loader), loss.item()
            ), end='\r')
            writer.add_scalar('loss', loss.item(), iters) # add to tensorboard
            iters += 1
            count=0
    return loss.item(), iters

def test(epoch, model, criterion, test_loader, writer, iters):
    model.eval()
    
    for batch_idx, (img, target) in enumerate(test_loader):
        
        img, target = Variable(img.cuda()), Variable(target.cuda())
        img, target = img.type(torch.cuda.FloatTensor), target.type(torch.cuda.FloatTensor)
        
        losses = []
        for stage in range(1, args.stages + 1):
            if stage == 1:
                inputs = img
            else:
                inputs = torch.cat([img, outputs], axis=1)
            outputs = model(inputs, stage=stage)
            loss = criterion(outputs, target)
            losses.append(loss)
        
        loss = sum(losses)
        
        if batch_idx % args.log_interval == 0 and not batch_idx==0 :
            print('Val Epoch:{}/{} [{}/{} ({:.0f}%)]  Val loss:{:.4f}'.format(
                epoch + 1, args.epochs,
                batch_idx * len(img), len(test_loader) * len(img),
                100.0 * batch_idx / len(test_loader), loss.item()
            ), end='\r')
            writer.add_scalar('valloss', loss.item(), iters) # add to tensorboard
    return loss.item()


def save_checkpoint(state, is_best, n_keypoints, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'checkpoints/model_best.kp%i.pth.tar' % n_keypoints)

        
def resume(args, ckpt,model):
    if os.path.isfile(ckpt):
        print('==> loading checkpoint {}'.format(ckpt))
        checkpoint = torch.load(ckpt)
        args.start_epoch = checkpoint['epoch']
        best_loss = checkpoint['loss']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer = checkpoint['optimizer']
        iters=checkpoint['iters']
        print("==> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        return model,optimizer,args.start_epoch,best_loss,iters
    else:
        print("==> no checkpoint found at '{}'".format(args.resume))
    
    
def adjust_lr(args, optimizer, epoch, decay=20):
    """
        adjust the learning rate initial lr decayed 10 every 20 epoch
    """
    lr=args.lr*(0.1**(epoch//decay))
    for param in optimizer.param_groups:
        param['lr'] = lr
        

def get_dataset_indices(file, n_keypoints):
    df = pd.read_csv(file)
    notnan = df.apply(lambda x: np.sum(~x[:30].isnull()), axis=1)
    return notnan[notnan == n_keypoints*2].reset_index(drop=True).index


def get_dataloaders(n_keypoints, use_val=True):
    tsfm = transforms.Compose([transforms.ToTensor()])
    dataset = FacialKeyPointsDataset(csv_file=args.train_csv,
                                     n_keypoints=n_keypoints,
                                     size=(96,96),
                                     transform=tsfm)
    indices = get_dataset_indices(file=args.train_csv, n_keypoints=n_keypoints)
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    
    if use_val:
        train_idx, val_idx = train_test_split(indices, test_size=0.15,
                                      shuffle=True, random_state=args.seed)
        train_sampler = SubsetRandomSampler(train_idx)
        val_samper = SubsetRandomSampler(val_idx)
        train_loader = DataLoader(dataset,
                              sampler=train_sampler,
                              batch_size=args.batch_size,
                              drop_last=True,
                              **kwargs)
        val_loader = DataLoader(dataset,
                            sampler=val_samper,
                            batch_size=args.batch_size*2,
                            drop_last=True,
                            **kwargs)
    else:
        train_sampler = SubsetRandomSampler(indices)
        train_loader = DataLoader(dataset,
                                  sampler=train_sampler,
                                  batch_size=args.batch_size,
                                  drop_last=True,
                                  **kwargs)
        val_loader = None
    
    return train_loader, val_loader
      
    
def main(n_keypoints, use_val=True):
    """
    Args:
        n_keypoints(int): number of keypoints (x,y) in dataset
        use_val(bool): False when using all train data for predictions,
                       otherwise, split train for testing
    """
    
    args.cuda = torch.cuda.is_available()
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        
    # SmoothL1Loss/Huber loss is less sensitive to outliers than MSELoss
    # absolute squared term < 1, use L1, else use L2
    criterion = nn.SmoothL1Loss()
    model = CPM(k=n_keypoints)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    writer = SummaryWriter('logs/'+datetime.now().strftime('%B-%d'))
    best_loss = 1e+5
    best_loss_val = 1e+5
    iters = 0
    
    train_loader, val_loader = get_dataloaders(n_keypoints)
    
    assert model is not None
    if args.cuda: 
        model.cuda()
    
    # resume training 
    if args.resume:
        model, optimizer, args.start_epoch, best_loss, iters = resume(args, args.resume, model)

    # train loop
    for epoch in range(args.start_epoch, args.epochs):
        adjust_lr(args, optimizer, epoch, decay=5)
        t1 = time.time()
        loss, iters = train(args, epoch, model,
                            criterion, optimizer,
                            train_loader, writer, iters)
        is_best = loss < best_loss
        best_loss = min(best_loss, loss)
        
        state = {
            'epoch':epoch,
            'state_dict':model.state_dict(),
            'optimizer':optimizer,
            'loss':best_loss,
            'iters': iters,
        }
        
        if use_val:
            loss_val = test(args, epoch, model, criterion,
                            val_loader, writer, iters)
            best_loss_val = min(best_loss_val, loss_val)
            state['loss_val'] = best_loss_val
            
        save_checkpoint(state, is_best, n_keypoints)
    writer.close()
    
    
def show_heatmap(img, heatmap):
    plt.axis('off')
    plt.imshow(img, cmap='gray', alpha=0.5)
    plt.imshow(heatmap, alpha=0.5)

def get_img_and_output(model, df, idx):
    img = np.array(df.iloc[idx,1].split())
    img = img.astype(np.float32).reshape(96,96)
    #img /= 255.0
    img_input = np.expand_dims(img, axis=0)
    img_input = np.expand_dims(img_input, axis=0)
    img_input = torch.from_numpy(img_input)
    outputs_stage1 = model(img_input, stage=1)
    input_stage2 = torch.cat([img_input, outputs_stage1], axis=1)
    outputs_stage2 = model(input_stage2, stage=2)
    outputs_stage2 = outputs_stage2
    
    return img, outputs_stage2
    
def display_heatmap_eachkp(model, df, outfile):
    fig = plt.figure(figsize=(20, 6))
    fig.tight_layout()
    
    idx = random.randint(0, df.shape[0])
    img, heatmaps = get_img_and_output(model, df, idx)
    heatmaps = torch.squeeze(heatmaps, axis=0)
    n = len(heatmaps)
    for i in range(0, n):
        heatmap = heatmaps[i].detach().numpy()
        fig.add_subplot(2, (n + 1)//2, i + 1)
        show_heatmap(img, heatmap)
    fig.savefig(outfile)
    print(f'Saved to {outfile}')
        
def display_heatmap_combined(model, df, outfile):
    fig = plt.figure(figsize=(10, 10))
    fig.tight_layout()
    rows, columns = 3, 3
    for i in range(columns*rows):
        idx = random.randint(0, df.shape[0])
        img, heatmaps = get_img_and_output(model, df, idx)
        heatmaps = heatmaps.sum(axis=1)
        heatmaps = torch.squeeze(heatmaps, axis=0)
        heatmaps = heatmaps.detach().numpy()
        
        fig.add_subplot(rows, columns, i + 1)
        show_heatmap(img, heatmaps)

    if outfile:
        fig.savefig(outfile)
        print(f'Saved to {outfile}')

def display_heatmap_combined_one(model, df, idx, outfile):
    img, output = get_img_and_output(model, df, idx)
    show_heatmap(img, output)
    if outfile:
        plt.savefig(outfile)
        print(f'Saved to {outfile}')


def heatmaps_to_keypoints(heatmaps):
    keypoints = []
    for i in range(len(heatmaps)):
        heatmap = heatmaps[i]
        vals_x, indices_x = torch.max(heatmap, 0)
        vals_y, indices_y = torch.max(heatmap, 1)
        vals_x = vals_x.detach().numpy()
        vals_y = vals_y.detach().numpy()
        x = np.argmax(vals_x)
        y = np.argmax(vals_y)
        keypoints.append(x)
        keypoints.append(y)
    
    return np.array(keypoints)


def create_submission():
    submission = []
    df = pd.read_csv(args.test_csv)
    df_train = pd.read_csv(args.train_csv)
    df_idlookup = pd.read_csv(args.idlookup_table)
    kp_cols15 = df_idlookup[df_idlookup['ImageId'] == 1]['FeatureName'] # sample with 15 keypoints
    kp_cols4 = df_idlookup[df_idlookup['ImageId'] == 1481]['FeatureName'] # sample with 4 keypoints
    total = df.shape[0]
    
    results = []
    for id in df['ImageId']:
        print(id, total, end='\r')
        img = np.array(df[df['ImageId'] == id]['Image'].values[0].split())
        img = img.astype(np.float32).reshape(96,96)
        #img /= 255.0
        img_input = np.expand_dims(img, axis=0)
        img_input = np.expand_dims(img_input, axis=0)
        
        keypoints_include = df_idlookup[df_idlookup['ImageId'] == id]['FeatureName']
        if len(keypoints_include) > 8:
            output = model15(torch.from_numpy(img_input), stage=1)
        else:
            output = model4(torch.from_numpy(img_input), stage=1)

        output = torch.squeeze(output, 0)
        keypoints = heatmaps_to_keypoints(output)
        
        if len(keypoints_include) > 8:
            results.append([img, keypoints])
        
        if len(keypoints_include) > 8:
            keypoints = keypoints[kp_cols15.isin(keypoints_include)]
        else:
            keypoints = keypoints[kp_cols4.isin(keypoints_include)]

        for i in range(len(keypoints)):
            submission.append(keypoints[i])
    
    write_submission(submission)
    return results


def write_submission(submission):
    df_submit = pd.DataFrame(submission)
    df_submit.columns = ['Location']
    df_submit['RowId'] = list(range(1,df_submit.shape[0]+1))
    df_submit = df_submit[['RowId', 'Location']]
    df_submit['Location'] = df_submit['Location'].apply(lambda x: x if x > 0 else 0)
    df_submit.to_csv('submission.csv', index=False)
    print('wrote to submission.csv', df_submit.shape)

    
def display_sample_keypoints(results, outfile):
    fig = plt.figure(figsize=(10, 10))
    fig.tight_layout()
    rows, columns = 3, 3
    for i in range(columns*rows):
        idx = random.randint(0, len(results))
        img, keypoints = results[idx]
        
        fig.add_subplot(rows, columns, i + 1)
        plt.axis('off')
        plt.imshow(img)
        for j in range(0, len(keypoints), 2):
            x, y = keypoints[j], keypoints[j + 1]
            plt.scatter(x, y, c='white', s=20)
    
    if outfile:
        fig.savefig(outfile)
        print(f'Saved to {outfile}')
        

def get_args():
    parser = argparse.ArgumentParser(description='Kaggle facial keypoints adapted from Udacity facial keypoints project')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='input batch size for training (default: 8)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='initial learning rate')
    parser.add_argument('--test-batch-size', type=int, default=16, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--start-epoch', type=int, default=0, 
                        help='start epoch')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--seed', type=int, default=0,
                        metavar='S', help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--resume', type=str, default=None,
                        help='resume training')
    parser.add_argument('--train_csv', type=str, default='data/training.csv')
    parser.add_argument('--test_csv', type=str, default='data/test.csv')
    parser.add_argument('--idlookup_table', type=str, default='data/IdLookupTable.csv')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    args = dotdict(args)
    args.cuda = torch.cuda.is_available()
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    if not os.path.exists('checkpoints'):
        os.makedir(checkpoints)
        
    main(n_keypoints=15, use_val=False)
    main(n_keypoints=4, use_val=False)

    model15 = CPM(n_keypoints=15)
    model15.load_state_dict(torch.load('checkpoints/model_best.kp15.pth.tar')['state_dict'])
    model15.eval()

    model4 = CPM(n_keypoints=4)
    model4.load_state_dict(torch.load('checkpoints/model_best.kp4.pth.tar')['state_dict'])
    model4.eval()
    
    if not os.path.exists('samples'):
        os.makedir(samples)
        
    df = pd.read_csv('data/test.csv')
    display_heatmap_eachkp(model15, df, 'samples/sample.15kp.eachheatmap.png')
    display_heatmap_eachkp(model4, df, 'samples/sample.4kp.eachheatmap.png')
    display_heatmap_combined(model15, df, 'samples/sample.15kp.combinedheatmap.png')
    display_heatmap_combined(model4, df, 'samples/sample.4kp.combinedheatmap.png')

    results = create_submission()
    display_sample_keypoints(results, 'samples/sample.15kp.prediction.png')
                 