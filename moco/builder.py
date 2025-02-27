# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn


class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, mlp=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim)

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        # 参数不会因为optim.step而更新
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0) # [N, K]
        # 指针，指示
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # 为label也生成一个bank
        self.register_buffer("label_bank", torch.zeros(self.K, dtype=torch.float)) # [K]
        self.label_bank -= 1 # all init -1

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, label):
        # print(keys.shape) # [32, 128]
        # gather keys before updating queue
        keys = concat_all_gather(keys) # [64, 128] [N, C]

        label = concat_all_gather(label)
        # print(keys.shape)
        # debug

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # queue = torch.Size([128, 65536])
        # keys.T = torch.Size([128, 64])
        # print("queue={}".format(self.queue.shape))
        # print("keys.T={}".format(keys.T.shape))
        # debug
        # [C, K]
        # replace the keys at ptr (dequeue and enqueue)
        # 队首先进，队首先出
        # 等价于0-队尾 end-队首的一个队列
        self.queue[:, ptr:ptr + batch_size] = keys.T # [C, N]
        # label bank
        self.label_bank[ptr:ptr + batch_size] = label
        ptr = (ptr + batch_size) % self.K  # move pointer
        self.queue_ptr[0] = ptr

        # print(self.label_bank.shape) # [1, 65536]
        # debug

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, im_q, im_k, label):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        # torch.Size([32, 128]) [N, C]
        # print("before norm={}".format(q.shape))

        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: [N, 1+K]
        logits = torch.cat([l_pos, l_neg], dim=1)
        # print(logits.shape) # [torch.Size([32, 65537])]
        # debug

        # apply temperature
        logits /= self.T

        gt = self.generate_label(label).cuda()

        # labels: positive key indicators
        # labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k, label)
        # print(label.shape)
        # print(self.label_bank.shape)
        # debug

        # print("logit={}".format(logits.shape))
        # print("gt={}".format(gt.shape))

        logits = torch.sigmoid(logits)

        # print("===================================")
        # print("label={}".format(label))
        # print("gt={}".format(gt))
        # print("===================================")

        return logits, gt

    def generate_label(self, label, neg_idx=23):
        """
        核心
        根据label关系产生一对多的label
        """
        N = label.shape[0]
        # gt = torch.zeros(N, self.K, dtype=torch.long)

        label_cur = torch.unsqueeze(label, dim=1) # [N, 1]
        label_cur = label_cur.repeat(1, self.K) # [N ,K]

        lable_all = torch.unsqueeze(self.label_bank, dim=0) # [1, K]
        label_all = lable_all.repeat(N, 1) # [N, k]

        # print(label_cur.device) # cuda:0
        # print(label_all.device) # cuda:0
        # debug
        a = torch.tensor(1).cuda()
        b = torch.tensor(0).cuda()
        gt = torch.where(label_cur==label_all, a, b) # [N, K]

        # mask neg label
        mask = torch.ones(N, self.K).cuda()
        coord = torch.where(label_cur == neg_idx)
        # print(coord[0].shape[0])
        # if coord[0].shape[0]!=2097152:
        #     debug
        mask[coord] = 0
        gt = gt * mask

        pos = torch.ones(N, 1, dtype=torch.float).cuda()
        gt = torch.cat([pos, gt], dim=1)

        # print("=====================================================")
        # num1 = torch.sum(gt==a)
        # num2 = torch.sum(gt==b)
        # print("num1={}".format(num1))
        # print("num2={}".format(num2))
        # if num1>40:
        #     debug
        # # debug
        # print("=====================================================")

        return gt

# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
