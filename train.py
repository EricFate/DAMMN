from data.Data import SeperationDataset,BaseLineDataset,NoiseDataset
from data.img_text_dataset import MyDataset
from model.Model import DAMMN,ClusterLossModel,ResNet,TextNet,CMML,scale_lr
import gzip
from config import opt
from torch.utils import data
from torch.optim.lr_scheduler import StepLR
# from tqdm import tqdm
import time
import torch as t
from utils import meter
from test import metrics
import numpy as np
import datetime


from argparse import ArgumentParser

parser = ArgumentParser('Domain adaption multi-modal net')
# parser.add_argument('--use-gpu', type = bool, default = True)
parser.add_argument('--visible-gpu', type = str, default = '0')
parser.add_argument('--dataset', type = str, default = 'MS-COCO')
parser.add_argument('--modal0', type=int, default=0)
parser.add_argument('--modal1', type=int, default=19)
# parser.add_argument('--textfilename', type = str, default = '/data/yangy/coco/sample_coco_text.npy')#Path of text madality feature data
# parser.add_argument('--imgfilenamerecord', type = str, default = '/data/yangy/coco/sample_coco_imgs.pkl')#Path of name list of img madality data
# parser.add_argument('--imgfilename', type = str, default = '/data/yangy/coco/sample_img/')#Path of img madality data
# parser.add_argument('--labelfilename', type = str, default = '/data/yangy/coco/sample_coco_label.npy')#Path of data label
# parser.add_argument('--savepath', type = str, default = '/home/yangy/wkt/COCO/EXP1/')
# parser.add_argument('--textbatchsize', type = int, default = 32)
# parser.add_argument('--imgbatchsize', type = int, default = 32)
# parser.add_argument('--batchsize', type = int, default = 4)#train and test batchsize
# parser.add_argument('--Textfeaturepara', type = str, default = '2211, 256, 128')#architecture of text feature network
# parser.add_argument('--Imgpredictpara', type = str, default = '128, 20')#architecture of img predict network
# parser.add_argument('--Textpredictpara', type = str, default = '128, 20')#architecture of text predict network
# parser.add_argument('--Predictpara', type = str, default = '128, 20')#architecture of attention predict network
# parser.add_argument('--Attentionparameter', type = str, default = '128, 64, 32, 1')#architecture of attention network
# parser.add_argument('--img-supervise-epochs', type = int, default = 0)
# parser.add_argument('--text-supervise-epochs', type = int, default = 1)
# parser.add_argument('--epochs', type = int, default = 25)# train epochs
# parser.add_argument('--img-lr-supervise', type = float, default = 0.001)
# parser.add_argument('--text-lr-supervise', type = float, default = 0.001)
# parser.add_argument('--lr', type = float, default = 1e-4)#train Learning rate
# parser.add_argument('--lr_decay', type = float, default = 0.5)#train Learning rate
# parser.add_argument('--weight_decay', type = float, default = 0)
# parser.add_argument('--lambda1', type = float, default = 0.01)#ratio of train data to test data
# parser.add_argument('--lambda2', type = float, default = 1)#ratio of train data to test data

def eval(data_set,model):
    print('testing data')
    opt.train = False
    model.eval()
    dataloader = data.DataLoader(data_set, \
                                  batch_size=32, \
                                  shuffle=True, \
                                  # pin_memory=True,
                                  num_workers=opt.num_workers)
    preds = []
    labels = []
    m1_preds = []
    m2_preds = []
    for ii, (unlabel_left_x, unlabel_right_x, unlabel_y) in enumerate(dataloader):
        # train faster rcnn
        unlabel_left_x, unlabel_right_x, unlabel_y = unlabel_left_x.cuda(), unlabel_right_x.cuda(), unlabel_y.cuda()
        m1_pred,m2_pred,pred = model.predict((unlabel_left_x,unlabel_right_x))
        preds.append(pred)
        m1_preds.append(m1_pred)
        m2_preds.append(m2_pred)
        labels.append(unlabel_y)
    m1_preds = t.cat(tuple(m1_preds)).cpu()
    m2_preds = t.cat(tuple(m2_preds)).cpu()
    preds = t.cat(tuple(preds)).cpu()
    labels = t.cat(tuple(labels)).cpu()
    labels_numpy = labels.numpy()
    # one_hot = t.zeros((labels.size(0), opt.n_class)).scatter_(1, labels.view(-1,1), 1)
    pred_numpy = preds.numpy()
    m1_preds = m1_preds.numpy()
    m2_preds = m2_preds.numpy()
    # one_hot_numpy = one_hot.numpy()


    print('modal 1 performance')
    m1_rank = metrics.ranking_loss(m1_preds, labels_numpy)
    print('rank loss: ', m1_rank)
    m1_coverage = metrics.coverage(m1_preds, labels_numpy)
    print('coverage: ', m1_coverage)
    m1_example_auc = metrics.example_auc(m1_preds, labels_numpy)
    print('example auc: ', m1_example_auc)
    m1_macro_auc = metrics.macro_auc(m1_preds, labels_numpy)
    print('macro auc: ', m1_macro_auc)
    m1_micro_auc = metrics.micro_auc(m1_preds, labels_numpy)
    print('micro auc: ', m1_micro_auc)
    m1_acc = metrics.average_precision(m1_preds, labels_numpy)
    print('average precision: ', m1_acc)

    print('modal 2 performance')
    m2_rank = metrics.ranking_loss(m2_preds, labels_numpy)
    print('rank loss: ', m2_rank)
    m2_coverage = metrics.coverage(m2_preds, labels_numpy)
    print('coverage: ', m2_coverage)
    m2_example_auc = metrics.example_auc(m2_preds, labels_numpy)
    print('example auc: ', m2_example_auc)
    m2_macro_auc = metrics.macro_auc(m2_preds, labels_numpy)
    print('macro auc: ', m2_macro_auc)
    m2_micro_auc = metrics.micro_auc(m2_preds, labels_numpy)
    print('micro auc: ', m2_micro_auc)
    m2_acc = metrics.average_precision(m2_preds, labels_numpy)
    print('average precision: ', m2_acc)

    print('final performance')
    f_rank = metrics.ranking_loss(pred_numpy, labels_numpy)
    print('rank loss: ', f_rank)
    f_coverage = metrics.coverage(pred_numpy, labels_numpy)
    print('coverage: ', f_coverage)
    f_example_auc = metrics.example_auc(pred_numpy, labels_numpy)
    print('example auc: ', f_example_auc)
    f_macro_auc = metrics.macro_auc(pred_numpy, labels_numpy)
    print('macro auc: ', f_macro_auc)
    f_micro_auc = metrics.micro_auc(pred_numpy, labels_numpy)
    print('micro auc: ', f_micro_auc)
    f_acc = metrics.average_precision(pred_numpy, labels_numpy)
    print('average precision: ', f_acc)
    opt.train = True
    return np.array([[m1_rank,m1_coverage,m1_example_auc,m1_macro_auc,m1_micro_auc,m1_acc],
                     [m2_rank,m2_coverage,m2_example_auc,m2_macro_auc,m2_micro_auc,m2_acc],
                     [f_rank,f_coverage,f_example_auc,f_macro_auc,f_micro_auc,f_acc]])


def train(dataset,name='dammn',grid_search=False):



    if grid_search:
        r = [
            1e-3,1e-2,1e-1,
             1
            ,10
            ]
        ap = 0
        best_ap = 0
        best_l1 = 0
        best_l2 = 0
        for l1 in r:
            for l2 in r:
                opt.lambda1 = l1
                opt.lambda2 = l2
                test_result = _real_train(dataset,name)
                if np.max(test_result[:,-1,-1]) > ap:
                    best_ap = np.max(test_result[:,-1,-1]).item()
                    best_l1 = l1
                    best_l2 = l2
        print('best lambda1 = %f'%best_l1)
        print('best lambda2 = %f'%best_l2)
        print('best average precision = %f'%best_ap)
    else:
        _real_train(dataset,name)


def _real_train(dataset,name):
    print('---------------------------------------------')
    print('%s training: %s, lambda1 = %f , lambda2 = %f' % (name, opt.dataset, opt.lambda1, opt.lambda2))
    # print('dammn training:')
    # opt.model1_path = 'checkpoint/imgnet_%s_0.pth'%(opt.generation_method)
    # opt.model2_path = 'checkpoint/imgnet_%s_1.pth'%(opt.generation_method)
    begin = datetime.datetime.now()
    # f = gzip.open(opt.mnist_file_path, 'rb')
    opt.train = True
    opt.supervise = False
    # if opt.generation_method == 'noise':
    #     data_set = NoiseDataset(f)
    # else:
    #     data_set = SeperationDataset(f)
    # data_set = MyDataset(superviseunsuperviseproportion=opt.prop)
    data_set = dataset
    dataloader = data.DataLoader(data_set, \
                                 batch_size=4, \
                                 shuffle=True, \
                                 # pin_memory=True,
                                 num_workers=opt.num_workers)
    loss_meter = meter.WindowAverageValueMeter(5)
    previous_loss = 1e10
    if name == 'dammn':
        model = DAMMN().cuda()
    else:
        model = CMML().cuda()
    optimizer = model.optimizer
    # scheduler = StepLR(optimizer, step_size=500, gamma=0.9)
    lr = opt.lr
    test_result = []
    first_epoch_loss = []
    epoch_loss = []
    for epoch in range(opt.epoch):
        print('train epoch', epoch, '.............................')
        loss_meter.reset()
        start = time.time()
        model.train()
        for ii, (train_left_x, train_right_x, train_y, unlabel_left_x, unlabel_right_x) in enumerate(dataloader):
            # train faster rcnn
            train_left_x = t.cat(tuple(train_left_x))
            train_right_x = t.cat(tuple(train_right_x))
            train_y = t.cat(tuple(train_y))
            unlabel_left_x = t.cat(tuple(unlabel_left_x))
            unlabel_right_x = t.cat(tuple(unlabel_right_x))
            # scheduler.step()
            train_left_x, train_right_x, train_y, unlabel_left_x, unlabel_right_x = train_left_x.cuda(), train_right_x.cuda(), train_y.cuda(), unlabel_left_x.cuda(), unlabel_right_x.cuda()
            # loss,class_loss = dammn.train_step((train_left_x,train_right_x),train_y,(unlabel_left_x,unlabel_right_x))
            loss, class_loss, div_loss, ent_loss, img_loss, text_loss = model.train_step((train_left_x, train_right_x),
                                                                                         train_y, (unlabel_left_x,
                                                                                                   unlabel_right_x))
            loss_meter.add(loss.item())
            if epoch == 0:
                first_epoch_loss.append(
                    [class_loss.item(), div_loss.item(), ent_loss.item(), img_loss.item(), text_loss.item()])
        test_result.append(eval(data_set, model))
        print('img_loss', img_loss)
        print('text_loss', text_loss)
        print('class_loss', class_loss)
        print('div_loss', div_loss)
        print('ent_loss', ent_loss)
        epoch_loss.append([class_loss.item(), div_loss.item(), ent_loss.item(), img_loss.item(), text_loss.item()])
        end = time.time()
        print('epoch training duration: %d s' % (end - start))
        print("epoch:{epoch},lr:{lr},loss:{loss}".format(
            epoch=epoch, loss=loss_meter.value()[0], lr=lr))
        # update learning rate
        if loss_meter.value()[0] > previous_loss:
            scale_lr(optimizer,opt.lr_decay)

        previous_loss = loss_meter.value()[0]
    print('total trainning duration:', datetime.datetime.now() - begin)
    print('---------------------------------------------')
    test_result = np.array(test_result)
    np.save('result/%s_%s_%f_%f' % (opt.dataset, name, opt.lambda1, opt.lambda2),
            test_result)
    np.save('result/%s_%s_%f_%f_first_epoch_loss' % (opt.dataset, name, opt.lambda1, opt.lambda2),
            np.array(first_epoch_loss))
    np.save('result/%s_%s_%f_%f_epoch_loss' % (opt.dataset, name, opt.lambda1, opt.lambda2),
            np.array(epoch_loss))
    return test_result


def semi_eval(data_set,model):
    print('testing data')
    opt.train = False
    model.eval()
    dataloader = data.DataLoader(data_set, \
                                  batch_size=32, \
                                  shuffle=True, \
                                  # pin_memory=True,
                                  num_workers=opt.num_workers)
    preds = []
    labels = []
    for ii, (unlabel_left_x, unlabel_right_x, test_y) in enumerate(dataloader):
        # train faster rcnn
        test_x = (unlabel_left_x, unlabel_right_x)[opt.baseline_modal]
        test_x, test_y = test_x.cuda(), test_y.cuda()
        pred = model.predict(test_x)
        preds.append(pred)
        labels.append(test_y)
    preds = t.cat(tuple(preds)).cpu()
    labels = t.cat(tuple(labels)).cpu()
    labels_numpy = labels.numpy()
    pred_numpy = preds.numpy()
    # f_macro_auc = metrics.macro_auc(pred_numpy, one_hot_numpy)
    # print('macro auc: ', f_macro_auc)
    # f_micro_auc = metrics.micro_auc(pred_numpy, one_hot_numpy)
    # print('micro auc: ', f_micro_auc)
    # f_acc = metrics.accuracy(pred_numpy, labels.numpy())
    # print('accuracy: ', f_acc)
    f_rank = metrics.ranking_loss(pred_numpy, labels_numpy)
    print('rank loss: ', f_rank)
    f_coverage = metrics.coverage(pred_numpy, labels_numpy)
    print('coverage: ', f_coverage)
    f_example_auc = metrics.example_auc(pred_numpy, labels_numpy)
    print('example auc: ', f_example_auc)
    f_macro_auc = metrics.macro_auc(pred_numpy, labels_numpy)
    print('macro auc: ', f_macro_auc)
    f_micro_auc = metrics.micro_auc(pred_numpy, labels_numpy)
    print('micro auc: ', f_micro_auc)
    f_acc = metrics.average_precision(pred_numpy, labels_numpy)
    print('average precision: ', f_acc)
    opt.train = True
    return np.array([f_rank,f_coverage,f_example_auc,f_macro_auc,f_micro_auc,f_acc])


def semi_baseline_train(dataset):
    print('---------------------------------------------')
    print('semi baseline training: %s'%(opt.dataset))
    begin = datetime.datetime.now()
    # f = gzip.open(opt.mnist_file_path, 'rb')
    opt.train = True
    opt.supervise = False
    opt.multimodal = False
    # if opt.generation_method == 'noise':
    #     data_set = NoiseDataset(f)
    # else:
    #     data_set = SeperationDataset(f)
    data_set =dataset
    dataloader = data.DataLoader(data_set, \
                                  batch_size=4, \
                                  shuffle=True, \
                                  # pin_memory=True,
                                  num_workers=opt.num_workers)
    loss_meter = meter.AverageValueMeter()
    previous_loss = 1e10
    model = ClusterLossModel(opt.baseline_modal).cuda()
    optimizer = model.optimizer
    lr = opt.lr
    test_result = []
    first_epoch_loss = []
    epoch_loss = []
    max_ap = 0
    max_epoch = 0
    for epoch in range(opt.epoch):
        print('train epoch', epoch, '.............................')
        loss_meter.reset()
        start = time.time()
        model.train()
        for ii, (train_left_x,train_right_x,train_y,unlabel_left_x,unlabel_right_x) in enumerate(dataloader):
            # train faster rcnn
            train_x = (train_left_x,train_right_x)[opt.baseline_modal]
            train_x =t.cat(tuple(train_x))
            train_y =t.cat(tuple(train_y))
            unlabel_x = (unlabel_left_x,unlabel_right_x)[opt.baseline_modal]
            unlabel_x =t.cat(tuple(unlabel_x))
            train_x, train_y, unlabel_x =train_x.cuda(),train_y.cuda(),unlabel_x.cuda()
            # loss,class_loss = dammn.train_step((train_left_x,train_right_x),train_y,(unlabel_left_x,unlabel_right_x))
            loss,class_loss,ent_loss = model.train_step(train_x,train_y,unlabel_x)
            loss_meter.add(loss.item())
            if epoch == 0:
                first_epoch_loss.append([class_loss.item(),ent_loss.item()])
            # semi_eval(data_set,model)
        print('class_loss:',class_loss.item())
        print('ent_loss:',ent_loss.item())
        epoch_loss.append([class_loss.item(), ent_loss.item()])
        end = time.time()
        print('epoch training duration: %d s'%(end-start))
        print("epoch:{epoch},lr:{lr},loss:{loss}".format(
                    epoch = epoch,loss = loss_meter.value()[0],lr=lr))
        result = semi_eval(data_set, model)
        if result[-1] > max_ap:
            max_ap = result[-1]
            max_epoch = epoch
        test_result.append(result)
        # # update learning rate
        if loss_meter.value()[0] > previous_loss:
            scale_lr(optimizer,opt.lr_decay)

        previous_loss = loss_meter.value()[0]
        t.save(model.state_dict(),'checkpoint/semi_baseline_%s_modal%d_epoch%d.pth'%(opt.dataset,opt.baseline_modal,epoch))
    print('total trainning duration:',datetime.datetime.now()-begin)
    print('---------------------------------------------')
    np.save('result/%s_modal%d_cluster_baseline'%(opt.dataset,opt.baseline_modal),
            np.array(test_result))
    np.save('result/%s_cluster_baseline_first_epoch_loss'%(opt.dataset),
            np.array(first_epoch_loss))
    np.save('result/%s_cluster_baseline_epoch_loss'%(opt.dataset),
            np.array(epoch_loss))
    return max_epoch

def baseline_eval(data_set,model):
    print('testing data')
    opt.train = False
    model.eval()
    dataloader = data.DataLoader(data_set, \
                                  batch_size=32, \
                                  shuffle=True, \
                                  # pin_memory=True,
                                  num_workers=opt.num_workers)
    preds = []
    labels = []
    for ii, (unlabel_x_1,unlabel_x_2, unlabel_y) in enumerate(dataloader):
        # train faster rcnn
        unlabel_x = (unlabel_x_1,unlabel_x_2)[opt.baseline_modal]
        unlabel_x = unlabel_x.cuda()
        pred = model.predict(unlabel_x)
        preds.append(pred)
        labels.append(unlabel_y)
    preds = t.cat(tuple(preds)).cpu()
    labels = t.cat(tuple(labels)).cpu()
    labels_numpy = labels.numpy()
    # one_hot = t.zeros((labels.size(0), opt.n_class)).scatter_(1, labels.view(-1,1), 1)
    pred_numpy = preds.numpy()
    # one_hot_numpy = one_hot.numpy()
    f_rank = metrics.ranking_loss(pred_numpy, labels_numpy)
    print('rank loss: ', f_rank)
    f_coverage = metrics.coverage(pred_numpy, labels_numpy)
    print('coverage: ', f_coverage)
    f_example_auc = metrics.example_auc(pred_numpy, labels_numpy)
    print('example auc: ', f_example_auc)
    f_macro_auc = metrics.macro_auc(pred_numpy, labels_numpy)
    print('macro auc: ', f_macro_auc)
    f_micro_auc = metrics.micro_auc(pred_numpy, labels_numpy)
    print('micro auc: ', f_micro_auc)
    f_acc = metrics.average_precision(pred_numpy, labels_numpy)
    print('average precision: ', f_acc)
    opt.train = True
    return np.array([f_rank,f_coverage,f_example_auc,f_macro_auc,f_micro_auc,f_acc])

def baselinetrain(dataset):
    print('---------------------------------------------')
    print('baseline training: %s'%(opt.dataset))
    begin = datetime.datetime.now()
    # f = gzip.open(opt.mnist_file_path, 'rb')
    opt.train = True
    opt.supervise = True
    opt.multimodal =False
    # data_set = BaseLineDataset(f)
    data_set = dataset
    dataloader = data.DataLoader(data_set, \
                                  batch_size=32, \
                                  shuffle=True, \
                                  # pin_memory=True,
                                  num_workers=opt.num_workers)
    loss_meter = meter.AverageValueMeter()
    previous_loss = 1e10
    if opt.baseline_modal == 0:
        net = ResNet().cuda()
    else:
        net = TextNet(opt.text_cfg).cuda()
    optimizer = net.optimizer
    lr = opt.lr
    test_result = []
    max_ap = 0
    max_epoch = 0
    for epoch in range(opt.epoch):
        print('train epoch', epoch, '.............................')
        loss_meter.reset()
        start = time.time()
        net.train()
        for ii, (train_x_1,train_x_2,train_y) in enumerate(dataloader):
            train_x = (train_x_1,train_x_2)[opt.baseline_modal]
            train_x, train_y =train_x.cuda(),train_y.cuda()
            loss = net.train_step(train_x,train_y)
            loss_meter.add(loss.item())
        end = time.time()
        print('epoch training duration: %d s'%(end-start))
        print("epoch:{epoch},lr:{lr},loss:{loss}".format(
                    epoch = epoch,loss = loss_meter.value()[0],lr=lr))
        result = baseline_eval(data_set, net)
        if result[-1] > max_ap:
            max_ap = result[-1]
            max_epoch = epoch
        test_result.append(result)
        # update learning rate
        if loss_meter.value()[0] > previous_loss:
            scale_lr(optimizer,opt.lr_decay)

        previous_loss = loss_meter.value()[0]
        t.save(net.state_dict(),'checkpoint/baseline_%s_modal%d_epoch%d.pth'%(opt.dataset,opt.baseline_modal,epoch))
    print('total trainning duration:',datetime.datetime.now()-begin)
    print('---------------------------------------------')
    np.save('result/%s_modal%d_baseline'%(opt.dataset,opt.baseline_modal),
            np.array(test_result))
    return max_epoch


def baseline_assemble_test(dataset,baseline_epoch,semi=False):
    print('---------------------------------------------')
    prefix = 'semi_baseline' if semi else 'baseline'
    print(prefix+' testing')
    state_dict_0 = t.load('checkpoint/%s_%s_modal%d_epoch%d.pth'%(prefix,opt.dataset,0,baseline_epoch[0]))
    state_dict_1 = t.load('checkpoint/%s_%s_modal%d_epoch%d.pth'%(prefix,opt.dataset,1,baseline_epoch[1]))
    if semi:
        net1 = ClusterLossModel(0).cuda()
        net2 = ClusterLossModel(1).cuda()
    else:
        net1 = ResNet().cuda()
        net2 = TextNet(opt.text_cfg).cuda()
    net1.load_state_dict(state_dict_0)
    net2.load_state_dict(state_dict_1)
    opt.train = False
    net1.eval()
    net2.eval()
    dataloader = data.DataLoader(dataset, \
                                  batch_size=32, \
                                  shuffle=True, \
                                  # pin_memory=True,
                                  num_workers=opt.num_workers)
    preds = []
    labels = []
    m1_preds = []
    m2_preds = []
    for ii, (unlabel_left_x, unlabel_right_x, unlabel_y) in enumerate(dataloader):
        # train faster rcnn
        unlabel_left_x, unlabel_right_x, unlabel_y = unlabel_left_x.cuda(), unlabel_right_x.cuda(), unlabel_y.cuda()
        m1_pred = net1.predict(unlabel_left_x)
        m2_pred = net2.predict(unlabel_right_x)
        preds.append(t.max(t.stack((m1_pred,m2_pred)),dim=0)[0])
        m1_preds.append(m1_pred)
        m2_preds.append(m2_pred)
        labels.append(unlabel_y)
    m1_preds = t.cat(tuple(m1_preds)).cpu()
    m2_preds = t.cat(tuple(m2_preds)).cpu()
    preds = t.cat(tuple(preds)).cpu()
    labels = t.cat(tuple(labels)).cpu()
    labels_numpy = labels.numpy()
    # one_hot = t.zeros((labels.size(0), opt.n_class)).scatter_(1, labels.view(-1,1), 1)
    pred_numpy = preds.numpy()
    m1_preds = m1_preds.numpy()
    m2_preds = m2_preds.numpy()
    # one_hot_numpy = one_hot.numpy()


    print('modal 1 performance')
    m1_rank = metrics.ranking_loss(m1_preds, labels_numpy)
    print('rank loss: ', m1_rank)
    m1_coverage = metrics.coverage(m1_preds, labels_numpy)
    print('coverage: ', m1_coverage)
    m1_example_auc = metrics.example_auc(m1_preds, labels_numpy)
    print('example auc: ', m1_example_auc)
    m1_macro_auc = metrics.macro_auc(m1_preds, labels_numpy)
    print('macro auc: ', m1_macro_auc)
    m1_micro_auc = metrics.micro_auc(m1_preds, labels_numpy)
    print('micro auc: ', m1_micro_auc)
    m1_acc = metrics.average_precision(m1_preds, labels_numpy)
    print('average precision: ', m1_acc)

    print('modal 2 performance')
    m2_rank = metrics.ranking_loss(m2_preds, labels_numpy)
    print('rank loss: ', m2_rank)
    m2_coverage = metrics.coverage(m2_preds, labels_numpy)
    print('coverage: ', m2_coverage)
    m2_example_auc = metrics.example_auc(m2_preds, labels_numpy)
    print('example auc: ', m2_example_auc)
    m2_macro_auc = metrics.macro_auc(m2_preds, labels_numpy)
    print('macro auc: ', m2_macro_auc)
    m2_micro_auc = metrics.micro_auc(m2_preds, labels_numpy)
    print('micro auc: ', m2_micro_auc)
    m2_acc = metrics.average_precision(m2_preds, labels_numpy)
    print('average precision: ', m2_acc)

    print('final performance')
    f_rank = metrics.ranking_loss(pred_numpy, labels_numpy)
    print('rank loss: ', f_rank)
    f_coverage = metrics.coverage(pred_numpy, labels_numpy)
    print('coverage: ', f_coverage)
    f_example_auc = metrics.example_auc(pred_numpy, labels_numpy)
    print('example auc: ', f_example_auc)
    f_macro_auc = metrics.macro_auc(pred_numpy, labels_numpy)
    print('macro auc: ', f_macro_auc)
    f_micro_auc = metrics.micro_auc(pred_numpy, labels_numpy)
    print('micro auc: ', f_micro_auc)
    f_acc = metrics.average_precision(pred_numpy, labels_numpy)
    print('average precision: ', f_acc)
    opt.train = True
    test_result = np.array([[m1_rank,m1_coverage,m1_example_auc,m1_macro_auc,m1_micro_auc,m1_acc],
                     [m2_rank,m2_coverage,m2_example_auc,m2_macro_auc,m2_micro_auc,m2_acc],
                     [f_rank,f_coverage,f_example_auc,f_macro_auc,f_micro_auc,f_acc]])
    np.save('result/%s_%s_assemble' % (opt.dataset, prefix),
            test_result)

if __name__=='__main__':
    args = parser.parse_args()
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpu
    opt.dataset = args.dataset
    dataset = MyDataset(superviseunsuperviseproportion=opt.prop,dataset_name=opt.dataset,pre_set=opt.preset)
    opt.n_class = dataset.label_size
    opt.text_cfg[0] = dataset.text_input_size
    opt.predict_cfg[-1] = opt.n_class
    opt.model1_path = opt.model1_path % (args.dataset, args.modal0)
    opt.model2_path = opt.model2_path % (args.dataset, args.modal1)
    # opt.generation_method = 'sep'
    train(dataset,grid_search=False)
    # opt.baseline_modal = 0
    # baselinetrain()
    # # semi_baseline_train()
    # opt.baseline_modal = 1
    # baselinetrain()
    # semi_baseline_train()
    # opt.generation_method = 'sep_noise'
    # opt.lambda1 = 0.01
    # opt.lambda2 = 1
    # train(dataset,'cmml',grid_search=False)
    # baseline_epoch = []
    # semi_baseline_epoch = []
    # opt.baseline_modal = 0
    # baseline_epoch.append(baselinetrain(dataset))
    # semi_baseline_epoch.append(semi_baseline_train(dataset))
    # opt.baseline_modal = 1
    # baseline_epoch.append(baselinetrain(dataset))
    # semi_baseline_epoch.append(semi_baseline_train(dataset))
    #
    # baseline_assemble_test(dataset,baseline_epoch,semi=False)
    # baseline_assemble_test(dataset,semi_baseline_epoch,semi=True)
