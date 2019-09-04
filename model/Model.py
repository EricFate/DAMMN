# -*- coding: utf-8 -*-
import torchvision.models as Models
from torch import nn
import torch as t
from config import opt
from torchvision.models import AlexNet
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torch import optim


def nograd(f):
    def new_f(*args, **kwargs):
        with t.no_grad():
            return f(*args, **kwargs)

    return new_f


def BatchDot(x, y):
    return t.sum(x * y, dim=1)


def entropy_loss(x, y):
    n = x.size(0)
    loss = 0
    for i in range(n):
        for j in range(i):
            p_1 = t.sigmoid(t.dot(x[i], x[j]))
            p_2 = t.sigmoid(t.dot(y[i], y[j]))
            if t.abs(p_1 - p_2) < opt.delta:
                p = (t.sigmoid(t.dot(x[i], x[j])) + t.sigmoid(t.dot(y[i], y[j]))) / 2
                loss += p * t.log(p) + (1 - p) * t.log(1 - p)
    loss = loss * 2 / (n * (n - 1))
    return loss


def entropy_loss_single(x):
    n = x.size(0)
    loss = 0
    for i in range(n):
        for j in range(i):
            p = t.sigmoid(t.dot(x[i], x[j]))
            loss += p * t.log(p) + (1 - p) * t.log(1 - p)
    loss = loss * 2 / (n * (n - 1))
    return loss


def make_cnn_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 1
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class AttentionNet(nn.Module):
    def __init__(self, neure_num):
        super(AttentionNet, self).__init__()
        self.mlp = make_layers(neure_num[:-1])
        self.attention = nn.Linear(neure_num[-2], neure_num[-1])

    def forward(self, x):
        temp_x = self.mlp(x)
        y = self.attention(temp_x)
        return y


class TextNet(nn.Module):
    def __init__(self, neure_num):
        super(TextNet, self).__init__()
        self.mlp = make_layers(neure_num[:-1])
        self.feature = nn.Linear(neure_num[-2], neure_num[-1])
        self.predictor = PredictNet(opt.predict_cfg)
        _initialize_weights(self)
        # params = []
        # params.append({'params': self.parameters()})
        # for key, value in dict(self.named_parameters()).items():
        #     if value.requires_grad:
        #         if 'bias' in key:
        #             params += [{'params': [value], 'lr': lr * 2, 'weight_decay': 0}]
        #         else:
        #             params += [{'params': [value], 'lr': lr, 'weight_decay': opt.weight_decay}]
        # self.optimizer = optim.Adam(params, lr=opt.lr, weight_decay=opt.weight_decay)
        # self.scheduler = StepLR(self.optimizer, step_size=500, gamma=0.9)
        init_optimizer(self, opt.lr * 10)
        self.ce = nn.BCELoss()

    def extact_feature(self, x):
        temp_x = self.mlp(x)
        x = self.feature(temp_x)
        return x

    def forward(self, x):
        f = self.extact_feature(x)
        pred = self.predictor(f)
        return pred

    def train_step(self, train_x, train_y):
        # self.scheduler.step()
        self.optimizer.zero_grad()
        f = self.extact_feature(train_x)
        pred = self.predictor(f)
        loss = self.ce(pred, train_y)
        loss.backward()
        self.optimizer.step()
        return loss

    @nograd
    def predict(self, x):
        feature = self.extact_feature(x)
        predict = self.predictor(feature)
        return predict


class PredictNet(nn.Module):
    def __init__(self, neure_num):
        super(PredictNet, self).__init__()
        self.mlp = make_predict_layers(neure_num)
        self.sigmoid = nn.Sigmoid()
        _initialize_weights(self)

    def forward(self, x):
        y = self.mlp(x)
        y = self.sigmoid(y)
        return y


class ImageFeatureNet(nn.Module):
    def __init__(self):
        super(ImageFeatureNet, self).__init__()
        self.feature = Models.resnet18(pretrained=True)
        self.feature = nn.Sequential(*list(self.feature.children())[:-1])
        self.fc1 = nn.Sequential(
            nn.Linear(512, 128)
        )

    def forward(self, x):
        N = x.size()[0]
        x = self.feature(x.view(N, 3, 256, 256))
        x = x.view(N, 512)
        x = self.fc1(x)
        return x


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.feature = ImageFeatureNet()

        self.predictor = PredictNet(opt.predict_cfg)
        _initialize_weights(self.predictor)
        # params = []
        # params.append({'params': self.parameters()})
        # for key, value in dict(self.named_parameters()).items():
        #     if value.requires_grad:
        #         if 'bias' in key:
        #             params += [{'params': [value], 'lr': lr * 2, 'weight_decay': 0}]
        #         else:
        #             params += [{'params': [value], 'lr': lr, 'weight_decay': opt.weight_decay}]
        # self.optimizer = optim.Adam(params, lr=opt.lr, weight_decay=opt.weight_decay)
        # self.ce = nn.CrossEntropyLoss()
        # self.scheduler = StepLR(self.optimizer, step_size=500, gamma=0.9)
        init_optimizer(self, opt.lr)
        self.ce = nn.BCELoss()

    def extact_feature(self, x):
        return self.feature(x)

    def forward(self, x):
        f = self.extact_feature(x)
        pred = self.predictor(f)
        return pred

    def train_step(self, train_x, train_y):
        # self.scheduler.step()
        f = self.extact_feature(train_x)
        pred = self.predictor(f)
        loss = self.ce(pred, train_y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    @nograd
    def predict(self, x):
        feature = self.extact_feature(x)
        predict = self.predictor(feature)
        return predict


def make_layers(cfg):
    layers = []
    n = len(cfg)
    input_dim = cfg[0]
    for i in range(1, n):
        output_dim = cfg[i]
        layers += [nn.Linear(input_dim, output_dim), nn.ReLU(inplace=True)]
        input_dim = output_dim
    return nn.Sequential(*layers)


def make_predict_layers(cfg):
    layers = []
    n = len(cfg)
    input_dim = cfg[0]
    for i in range(1, n):
        output_dim = cfg[i]
        layers += [nn.Linear(input_dim, output_dim)]
        input_dim = output_dim
    return nn.Sequential(*layers)


class CMML(nn.Module):
    def __init__(self):
        super(CMML, self).__init__()
        self.net1 = ResNet()
        self.net2 = TextNet(opt.text_cfg)
        if opt.pretrain:
            if opt.model1_path is not None:
                self.net1.load_state_dict(t.load(opt.model1_path))
            if opt.model2_path is not None:
                self.net2.load_state_dict(t.load(opt.model2_path))
        self.domain_descriminator = DescriminatorNet(opt.des_cfg)
        self.attention = AttentionNet(opt.attention_cfg)
        self.predict_net = PredictNet(opt.predict_cfg)
        # params = []
        # params.append({'params': self.parameters()})
        # for key, value in dict(self.named_parameters()).items():
        #     if value.requires_grad:
        #         if 'bias' in key:
        #             params += [{'params': [value], 'lr': lr * 2, 'weight_decay': 0}]
        #         else:
        #             params += [{'params': [value], 'lr': lr, 'weight_decay': opt.weight_decay}]
        # self.optimizer =optim.Adam(params,lr=opt.lr, weight_decay=opt.weight_decay)
        # self.scheduler = StepLR(self.optimizer, step_size=500, gamma=0.9)
        init_optimizer(self, opt.lr)
        self.ce = nn.CrossEntropyLoss()
        self.criterion = nn.BCELoss()
        self.cosine = nn.CosineSimilarity()

    def train_step(self, train_data, train_y, test_data):
        # self.scheduler.step()
        train_left_x, train_right_x = train_data
        test_left_x, test_right_x = test_data
        train_left_feature = self.net1.extact_feature(train_left_x)
        train_right_feature = self.net2.extact_feature(train_right_x)
        test_left_feature = self.net1.extact_feature(test_left_x)
        test_right_feature = self.net2.extact_feature(test_right_x)
        train_left_predict = self.net1.predictor(train_left_feature)
        train_right_predict = self.net2.predictor(train_right_feature)
        test_left_predict = self.net1.predictor(test_left_feature)
        test_right_predict = self.net2.predictor(test_right_feature)
        # attention
        supervise_imgk = self.attention(train_left_feature)
        supervise_textk = self.attention(train_right_feature)

        # print(modality_attention)
        modality_attention = t.cat((supervise_imgk, supervise_textk), 1)
        modality_attention = nn.functional.softmax(modality_attention, dim=1)
        train_total_feature = modality_attention[:, 0:1] * train_left_feature + modality_attention[:,
                                                                                1:] * train_right_feature
        train_total_predict = self.predict_net(train_total_feature)
        # train_total_predict = 0.5 * (train_left_predict + train_right_predict)
        img_loss = self.criterion(train_left_predict, train_y)
        text_loss = self.criterion(train_right_predict, train_y)
        total_loss = self.criterion(train_total_predict, train_y)
        class_loss = img_loss + text_loss + 2 * total_loss
        total_left_predict = t.cat((train_left_predict, test_left_predict))
        total_right_predict = t.cat((train_right_predict, test_right_predict))
        div_loss = t.sum(self.cosine(total_left_predict, total_right_predict))
        # div_loss = t.mean(t.log(self.domain_descriminator(total_left_feature)+opt.eps))+ \
        #            t.mean(t.log(1 - self.domain_descriminator(total_right_feature)+opt.eps))
        dis = 2 - self.cosine(test_left_predict, test_right_predict)

        tensor1 = dis[t.abs(dis) < opt.cita]
        tensor2 = dis[t.abs(dis) >= opt.cita]
        tensor1loss = t.sum(tensor1 * tensor1 / 2)
        tensor2loss = t.sum(opt.cita * (t.abs(tensor2) - opt.cita / 2))
        cons = (tensor1loss + tensor2loss) / dis.size()[0]

        loss = class_loss + opt.lambda1 * div_loss + opt.lambda2 * cons
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss, class_loss, div_loss, cons, img_loss, text_loss
        # return loss,class_loss

    @nograd
    def predict(self, x):
        left_x, right_x = x
        test_left_feature = self.net1.extact_feature(left_x)
        test_right_feature = self.net2.extact_feature(right_x)
        test_left_predict = self.net1.predictor(test_left_feature)
        test_right_predict = self.net2.predictor(test_right_feature)
        supervise_imgk = self.attention(test_left_feature)
        supervise_textk = self.attention(test_right_feature)

        # print(modality_attention)
        modality_attention = t.cat((supervise_imgk, supervise_textk), 1)
        modality_attention = nn.functional.softmax(modality_attention, dim=1)
        train_total_feature = modality_attention[:, 0:1] * test_left_feature + modality_attention[:,
                                                                               1:] * test_right_feature
        train_total_predict = self.predict_net(train_total_feature)
        return test_left_predict, test_right_predict, train_total_predict


class DAMMN(nn.Module):
    def __init__(self):
        super(DAMMN, self).__init__()
        self.net1 = ResNet()
        self.net2 = TextNet(opt.text_cfg)
        if opt.pretrain:
            if opt.model1_path is not None:
                self.net1.load_state_dict(t.load(opt.model1_path))
            if opt.model2_path is not None:
                self.net2.load_state_dict(t.load(opt.model2_path))
        self.domain_descriminator = DescriminatorNet(opt.des_cfg)
        self.attention = AttentionNet(opt.attention_cfg)
        self.predict_net = PredictNet(opt.predict_cfg)
        # params = []
        # params.append({'params': self.parameters()})
        # # for key, value in dict(self.named_parameters()).items():
        # #     if value.requires_grad:
        # #         if 'bias' in key:
        # #             params += [{'params': [value], 'lr': lr * 2, 'weight_decay': 0}]
        # #         else:
        # #             params += [{'params': [value], 'lr': lr, 'weight_decay': opt.weight_decay}]
        # self.optimizer =optim.Adam(params,lr=opt.lr, weight_decay=opt.weight_decay)
        # self.scheduler = StepLR(self.optimizer, stepsize=500, gamma=0.9)
        init_optimizer(self, opt.lr)
        self.ce = nn.CrossEntropyLoss()
        self.criterion = nn.BCELoss()

    def train_step(self, train_data, train_y, test_data):
        # self.scheduler.step()
        train_left_x, train_right_x = train_data
        test_left_x, test_right_x = test_data
        train_left_feature = self.net1.extact_feature(train_left_x)
        train_right_feature = self.net2.extact_feature(train_right_x)
        test_left_feature = self.net1.extact_feature(test_left_x)
        test_right_feature = self.net2.extact_feature(test_right_x)
        train_left_predict = self.net1.predictor(train_left_feature)
        train_right_predict = self.net2.predictor(train_right_feature)
        test_left_predict = self.net1.predictor(test_left_feature)
        test_right_predict = self.net2.predictor(test_right_feature)
        # attention
        supervise_imgk = self.attention(train_left_feature)
        supervise_textk = self.attention(train_right_feature)

        # print(modality_attention)
        modality_attention = t.cat((supervise_imgk, supervise_textk), 1)
        modality_attention = nn.functional.softmax(modality_attention, dim=1)
        train_total_feature = modality_attention[:, 0:1] * train_left_feature + modality_attention[:,
                                                                                1:] * train_right_feature
        train_total_predict = self.predict_net(train_total_feature)
        # train_total_predict = 0.5 * (train_left_predict + train_right_predict)
        img_loss = self.criterion(train_left_predict, train_y)
        text_loss = self.criterion(train_right_predict, train_y)
        total_loss = self.criterion(train_total_predict, train_y)
        class_loss = img_loss + text_loss + 2 * total_loss
        total_left_feature = t.cat((train_left_feature, test_left_feature))
        total_right_feature = t.cat((train_right_feature, test_right_feature))
        valid = Variable(t.ones(total_left_feature.size(0), 1), requires_grad=False).cuda()
        fake = Variable(t.zeros(total_right_feature.size(0), 1), requires_grad=False).cuda()
        d_real_error = self.criterion(self.domain_descriminator(total_left_feature), valid)
        d_fake_error = self.criterion(self.domain_descriminator(total_right_feature), fake)
        div_loss = d_real_error + d_fake_error
        # div_loss = t.mean(t.log(self.domain_descriminator(total_left_feature)+opt.eps))+ \
        #            t.mean(t.log(1 - self.domain_descriminator(total_right_feature)+opt.eps))

        ent_loss = entropy_loss(test_left_predict, test_right_predict)
        loss = class_loss
        # + opt.lambda1 * div_loss + opt.lambda2 * ent_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss, class_loss, div_loss, ent_loss, img_loss, text_loss
        # return loss,class_loss

    @nograd
    def predict(self, x):
        left_x, right_x = x
        test_left_feature = self.net1.extact_feature(left_x)
        test_right_feature = self.net2.extact_feature(right_x)
        test_left_predict = self.net1.predictor(test_left_feature)
        test_right_predict = self.net2.predictor(test_right_feature)
        supervise_imgk = self.attention(test_left_feature)
        supervise_textk = self.attention(test_right_feature)

        # print(modality_attention)
        modality_attention = t.cat((supervise_imgk, supervise_textk), 1)
        modality_attention = nn.functional.softmax(modality_attention, dim=1)
        train_total_feature = modality_attention[:, 0:1] * test_left_feature + modality_attention[:,
                                                                               1:] * test_right_feature
        train_total_predict = self.predict_net(train_total_feature)
        final_predict = t.max(t.stack((test_left_predict, test_right_predict, train_total_predict), dim=2), dim=2)[0]
        # return test_left_predict,test_right_predict,train_total_predict
        return test_left_predict, test_right_predict, final_predict


class DescriminatorNet(nn.Module):
    def __init__(self, cfg):
        super(DescriminatorNet, self).__init__()
        self.mlp = make_predict_layers(cfg)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        d = self.sigmoid(self.mlp(x))
        return d


class ClusterLossModel(nn.Module):
    def __init__(self, modal):
        super(ClusterLossModel, self).__init__()
        if modal == 0:
            self.img_net = ResNet().cuda()
        else:
            self.img_net = TextNet(opt.text_cfg).cuda()
        self.optimizer = self.img_net.optimizer
        # self.ce = nn.CrossEntropyLoss()
        self.ce = nn.BCELoss()
        # self.scheduler = StepLR(self.optimizer, step_size=500, gamma=0.9)

    def train_step(self, train_data, train_y, unlabel_data):
        # self.scheduler.step()
        self.optimizer.zero_grad()
        train_pred = self.img_net(train_data)
        unlabel_pred = self.img_net(unlabel_data)
        class_loss = self.ce(train_pred, train_y)
        ent_loss = entropy_loss_single(unlabel_pred)
        loss = class_loss + opt.lambda2 * ent_loss
        loss.backward()
        self.optimizer.step()
        return loss, class_loss, ent_loss

    @nograd
    def predict(self, x):
        return self.img_net(x)


class ImgNet(nn.Module):
    def __init__(self, cnn_cfg, feature_cfg, predict_cfg):
        super(ImgNet, self).__init__()
        self.convolution = make_cnn_layers(cnn_cfg)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((2, 2))
        self.feature = make_layers(feature_cfg)
        self.predictor = PredictNet(predict_cfg)
        _initialize_weights(self)
        init_optimizer(self, opt.lr)
        self.ce = nn.CrossEntropyLoss()

    def extact_feature(self, x):
        x = self.convolution(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        f = self.feature(x)
        return f

    def forward(self, x):
        f = self.extact_feature(x)
        pred = self.predictor(f)
        return pred

    def train_step(self, train_x, train_y):
        self.optimizer.zero_grad()
        f = self.extact_feature(train_x)
        pred = self.predictor(f)
        loss = self.ce(pred, train_y)
        loss.backward()
        self.optimizer.step()
        return loss

    @nograd
    def predict(self, x):
        feature = self.extact_feature(x)
        predict = self.predictor(feature)
        return predict


def _initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)


# class PredictNet(nn.Module):
#     def __init__(self,predict_cfg):
#         super(PredictNet, self).__init__()
#         self.mlp = make_layers(predict_cfg)
#         self.softmax = nn.Softmax(dim=1)
#         self._initialize_weights()
#
#     def forward(self, x):
#         return self.softmax(self.mlp(x))
#
#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, 0, 0.01)
#                 nn.init.constant_(m.bias, 0)
#
#
# def make_layers(cfg):
#     layers = []
#     n = len(cfg)
#     input_dim = cfg[0]
#     for i in range(1, n):
#         output_dim = cfg[i]
#         layers += [nn.Linear(input_dim, output_dim), nn.ReLU(inplace = True)]
#         input_dim = output_dim
#     return nn.Sequential(*layers)
#
# def make_predict_layers(cfg):
#     layers = []
#     n = len(cfg)
#     input_dim = cfg[0]
#     for i in range(1, n):
#         output_dim = cfg[i]
#         layers += [nn.Linear(input_dim, output_dim)]
#         input_dim = output_dim
#     return nn.Sequential(*layers)


def init_optimizer(self, lr):
    """
    return optimizer, It could be overwriten if you want to specify
    special optimizer
    """
    params = []
    for key, value in dict(self.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                params += [{'params': [value], 'lr': lr * 2, 'weight_decay': 0}]
            # elif 'attention' in key:
            #     params += [{'params': [value], 'lr': lr * 5, 'weight_decay': opt.weight_decay}]
            else:
                params += [{'params': [value], 'lr': lr, 'weight_decay': opt.weight_decay}]
    self.optimizer = t.optim.Adam(params)
    return self.optimizer


def scale_lr(optimizer, decay=0.1):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay
        return optimizer
