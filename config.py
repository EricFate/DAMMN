import torch
class Config:
    # mnist_file_path = 'G:\\data\\mnist\\mnist.pkl.gz'
    mnist_file_path = '/data/yangy/mnist.pkl.gz'

    pretrain = True
    model1_path = 'checkpoint/baseline_%s_modal0_epoch%d.pth'
    model2_path = 'checkpoint/baseline_%s_modal1_epoch%d.pth'


    n_class = 20
    num_workers = 0


    weight_decay = 0
    lr_decay = 0.5  # 1e-3 -> 1e-4
    lr = 1e-4
    delta = 1
    lambda1 = 1e-2
    lambda2 = 0.1
    cita = 1.003

    epoch = 20

    cnn_cfg =  [32, 32, 'M', 64, 64, 'M', 128, 128, 'M']
    # feature_cfg = [512,128]
    text_cfg = [2211, 256, 128]
    predict_cfg = [128,n_class]
    des_cfg = [128,1]
    attention_cfg = [128, 64, 32, 1]

    sep = 13
    dataset = 'coco'
    preset = True
    baseline_modal = 0
    train = True
    supervise = True
    multimodal = True

    perm = None
    prop = (3,6)
    eps = 1e-9
    def _parse(self, kwargs):
        state_dict = self._state_dict()
        for k, v in kwargs.items():
            if k not in state_dict:
                raise ValueError('UnKnown Option: "--%s"' % k)
            setattr(self, k, v)

        print('======user config========')
        print(self._state_dict())
        print('==========end============')

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in Config.__dict__.items() \
                if not k.startswith('_')}

opt = Config()