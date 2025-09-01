from .client import ClientBase
import copy
import torch
import torch.nn.functional as functional
import numpy as np
import operator
import gc
from utils.data import load_pkl
import os
from torch.utils.data import DataLoader
from utils.data import MultimodalDataset
import const.constants as const
from fedlog.logbooker import Attender
import utils.data as data
import torch.nn as nn
from tqdm import tqdm
from models.loss.ms import MCSoftContrastiveLoss
from dataset.transform.caption import image_to_caption_collate_fn


class CreamFl:
    def __init__(
        self,
        args,
        id,
        modal,
        modal_id,
        train_samples,
        test_samples,
        serial_id,
        logkey,
        **kwargs
    ) -> None:
        self.id = id
        self.serial_id = serial_id
        # model
        self.modal = modal
        self.modal_id = modal_id
        self.model = copy.deepcopy(args["model"][modal])

        self.algorithm = args["algorithm"]

        self.set_client_log(args, logkey)
        self.dataset = args[self.algorithm][self.modal]
        if "num_classes" in args[self.dataset]:
            self.num_classes = args[self.dataset]["num_classes"]
        self.epochs = args["epochs"]
        self.device = args["device"]
        self.batch_size = args["batch_size"]
        self.learning_rate = args["learning_rate"]
        self.loss = MCSoftContrastiveLoss()
        self.set_optimizer()
        # default set to 4
        self.losses = AverageMeter()
        self.top1, self.test_top1 = AverageMeter(), AverageMeter()
        self.top5, self.test_top5 = AverageMeter(), AverageMeter()
        self.inter_distance = 4
        if self.modal != "multimodal":
            self.center_labels_pos = torch.LongTensor(np.array(range(self.num_classes)))
            self.loss = nn.CrossEntropyLoss()
        # algorithm
        self.algorithm = args["algorithm"]
        if self.algorithm in args["fedAlgorithm"].keys():
            self.fedAlgorithm = args["fedAlgorithm"][self.algorithm]
        self.contrast_local_intra = self.fedAlgorithm["is_intra_constrast"]
        self.contrast_local_inter = self.fedAlgorithm["is_inter_constrast"]
        self.kd_weight = self.fedAlgorithm["kd_weight"]
        self.inter_intra_weight = self.fedAlgorithm["inter_intra_weight"]
        self.disabled_distill = self.fedAlgorithm["disabled_distill"]
        self.not_bert = self.fedAlgorithm["not_bert"]

        # dataset
        self.dataset_dir = const.DIR_DEPOSITORY
        if "dataset_dir" in args.keys():
            self.dataset_dir = args["dataset_dir"]
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.public_test_data = None

    def set_optimizer(self):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )

    def train(
        self, global_img_feature, global_txt_feature, distill_index, public_data_loader
    ):
        self.old_model = copy.deepcopy(self.model)
        self.old_model.eval()
        self.old_model.cuda()
        for i in range(self.epochs):
            if self.modal == "multimodal":
                self.mm_train_with_global_representations(
                    global_img_feature,
                    global_txt_feature,
                    distill_index,
                    public_data_loader,
                )
            else:
                if self.modal == "image":
                    continue
                self.train_with_global_representations(
                    global_img_feature,
                    global_txt_feature,
                    distill_index,
                    public_data_loader,
                )
        # need to modify
        if self.modal == "multimodal":
            self.public_test_data = public_data_loader
        else:
            self.public_test_data = self.load_test_data()
        self.test()
        del self.old_model
        gc.collect()

    def load_test_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        test_data = data.read_client_data(
            self.dataset, self.modal_id, self.dataset_dir, is_train=False
        )
        return DataLoader(
            test_data, batch_size=self.batch_size, drop_last=False, shuffle=True
        )

    def mm_train_with_global_representations(
        self, global_img_feature, global_txt_feature, distill_index, global_train_loader
    ):
        train_loader = self.load_data_by_modal(is_train=True)
        for idx, (
            images,
            captions,
            captions_word,
            caption_lens,
            _,
            _,
            index,
        ) in enumerate(train_loader):
            images = images.to(self.device)
            captions = captions.to(self.device)
            caption_lens = caption_lens.to(self.device)

            output = self.model(images, captions, captions_word, caption_lens)
            # print('img', output['image_features'].shape)
            # print('txt', output['caption_features'].shape)

            loss, loss_dict = self.loss(**output)

            self.optimizer.zero_grad()

            # if self.config.train.get('use_fp16'):
            #     with amp.scale_loss(loss, self.optimizer) as scaled_loss:
            #         scaled_loss.backward()
            # else:
            loss.backward()

            # if self.config.train.grad_clip > 0:
            #     nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(),
            #                                        self.config.train.grad_clip)
            self.optimizer.step()

        criterion = nn.CrossEntropyLoss().cuda()

        if self.contrast_local_intra and self.contrast_local_inter:
            global_img_feature, global_txt_feature = (
                global_img_feature.cuda(),
                global_txt_feature.cuda(),
            )
            distill_dict = {
                b: a for a, b in enumerate(distill_index)
            }  # index in coco to index to list 'distill_index'
            print("Start Intra & Inter Contrasting!")
            for idx, (
                images,
                captions,
                captions_word,
                caption_lens,
                _,
                _,
                index,
            ) in tqdm(enumerate(global_train_loader), total=len(global_train_loader)):
                self.optimizer.zero_grad()
                d_idx = operator.itemgetter(*index)(
                    distill_dict
                )  # idx of current batch

                images = images.to(self.device)
                captions = captions.to(self.device)
                caption_lens = caption_lens.to(self.device)

                output = self.model(images, captions, captions_word, caption_lens)

                out_img = (
                    output["image_features"].sum(axis=1)
                    if len(output["image_features"].shape) == 3
                    else output["image_features"]
                )
                out_txt = (
                    output["caption_features"].sum(axis=1)
                    if len(output["caption_features"].shape) == 3
                    else output["caption_features"]
                )

                target_img_feature = global_img_feature[d_idx, :].type_as(out_img)
                target_txt_feature = global_txt_feature[d_idx, :].type_as(out_txt)

                # pos
                pos_i = torch.sum(out_img * target_img_feature, dim=-1)
                pos_i = pos_i.reshape(-1, 1)
                pos_t = torch.sum(out_txt * target_txt_feature, dim=-1)
                pos_t = pos_t.reshape(-1, 1)
                # neg
                with torch.no_grad():
                    output_o = self.old_model(
                        images, captions, captions_word, caption_lens
                    )
                    out_img_o = (
                        output_o["image_features"].sum(axis=1)
                        if len(output_o["image_features"].shape) == 3
                        else output_o["image_features"]
                    )
                    out_txt_o = (
                        output_o["caption_features"].sum(axis=1)
                        if len(output_o["caption_features"].shape) == 3
                        else output_o["caption_features"]
                    )
                neg_i = torch.sum(out_img * out_img_o, dim=-1)
                neg_t = torch.sum(out_txt * out_txt_o, dim=-1)
                logits_1 = torch.cat((pos_i, neg_i.reshape(-1, 1)), dim=1)
                logits_2 = torch.cat((pos_t, neg_t.reshape(-1, 1)), dim=1)
                logits = torch.cat((logits_1, logits_2), dim=0)

                logits /= 0.5  # temperature
                labels = torch.zeros(images.size(0) * 2).cuda().long()

                loss_intra = criterion(logits, labels)

                # inter contrast
                logits_1_inter = torch.div(
                    torch.matmul(out_img, global_txt_feature.T), 0.5
                )
                logits_2_inter = torch.div(
                    torch.matmul(out_txt, global_img_feature.T), 0.5
                )

                labels_inter = torch.tensor(d_idx).cuda()

                loss_1_inter = criterion(logits_1_inter, labels_inter)
                loss_2_inter = criterion(logits_2_inter, labels_inter)
                loss_inter = loss_1_inter + loss_2_inter

                # if not self.args.loss_scale:
                #     loss = (loss_intra + loss_inter) * self.args.interintra_weight
                # else:
                loss = (
                    loss_intra + loss_inter / (loss_inter / loss_intra).detach()
                ) * self.inter_intra_weight

                self.optimizer.zero_grad()

                # if self.config.train.get('use_fp16'):
                #     with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                #         scaled_loss.backward()
                # else:
                loss.backward()

                # if self.config.train.grad_clip > 0:
                #     nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(),
                #                                        self.config.train.grad_clip)
                self.optimizer.step()

        elif self.contrast_local_intra:
            global_img_feature, global_txt_feature = (
                global_img_feature.cuda(),
                global_txt_feature.cuda(),
            )
            distill_dict = {
                b: a for a, b in enumerate(distill_index)
            }  # index in coco to index to list 'distill_index'
            print("Start Intra Contrasting!")
            for idx, (
                images,
                captions,
                captions_word,
                caption_lens,
                _,
                _,
                index,
            ) in tqdm(enumerate(global_train_loader), total=len(global_train_loader)):
                self.optimizer.zero_grad()
                d_idx = operator.itemgetter(*index)(distill_dict)

                images = images.to(self.device)
                captions = captions.to(self.device)
                caption_lens = caption_lens.to(self.device)

                output = self.model(images, captions, captions_word, caption_lens)

                out_img = (
                    output["image_features"].sum(axis=1)
                    if len(output["image_features"].shape) == 3
                    else output["image_features"]
                )
                out_txt = (
                    output["caption_features"].sum(axis=1)
                    if len(output["caption_features"].shape) == 3
                    else output["caption_features"]
                )

                target_img_feature = global_img_feature[d_idx, :].type_as(out_img)
                target_txt_feature = global_txt_feature[d_idx, :].type_as(out_txt)

                # pos
                pos_i = torch.sum(out_img * target_img_feature, dim=-1)
                pos_i = pos_i.reshape(-1, 1)
                pos_t = torch.sum(out_txt * target_txt_feature, dim=-1)
                pos_t = pos_t.reshape(-1, 1)

                # neg
                with torch.no_grad():
                    output_o = self.old_model(
                        images, captions, captions_word, caption_lens
                    )
                    out_img_o = (
                        output_o["image_features"].sum(axis=1)
                        if len(output_o["image_features"].shape) == 3
                        else output_o["image_features"]
                    )
                    out_txt_o = (
                        output_o["caption_features"].sum(axis=1)
                        if len(output_o["caption_features"].shape) == 3
                        else output_o["caption_features"]
                    )
                neg_i = torch.sum(out_img * out_img_o, dim=-1)
                neg_t = torch.sum(out_txt * out_txt_o, dim=-1)
                logits_1 = torch.cat((pos_i, neg_i.reshape(-1, 1)), dim=1)
                logits_2 = torch.cat((pos_t, neg_t.reshape(-1, 1)), dim=1)
                logits = torch.cat((logits_1, logits_2), dim=0)

                logits /= 0.5  # temperature
                labels = torch.zeros(images.size(0) * 2).cuda().long()

                loss = criterion(logits, labels)

                self.optimizer.zero_grad()

                # if self.config.train.get('use_fp16'):
                #     with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                #         scaled_loss.backward()
                # else:
                loss.backward()

                # if self.config.train.grad_clip > 0:
                #     nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(),
                #                                        self.config.train.grad_clip)
                self.optimizer.step()

        elif self.contrast_local_inter:
            global_img_feature, global_txt_feature = (
                global_img_feature.cuda(),
                global_txt_feature.cuda(),
            )
            distill_dict = {
                b: a for a, b in enumerate(distill_index)
            }  # index in coco to index to list 'distill_index'
            print("Start Inter-modal Contrasting!")
            for idx, (
                images,
                captions,
                captions_word,
                caption_lens,
                _,
                _,
                index,
            ) in tqdm(enumerate(global_train_loader), total=len(global_train_loader)):
                self.optimizer.zero_grad()
                d_idx = operator.itemgetter(*index)(distill_dict)

                images = images.to(self.device)
                captions = captions.to(self.device)
                caption_lens = caption_lens.to(self.device)

                output = self.model(images, captions, captions_word, caption_lens)
                out_img = (
                    output["image_features"].sum(axis=1)
                    if len(output["image_features"].shape) == 3
                    else output["image_features"]
                )
                out_txt = (
                    output["caption_features"].sum(axis=1)
                    if len(output["caption_features"].shape) == 3
                    else output["caption_features"]
                )

                logits_1 = torch.div(torch.matmul(out_img, global_txt_feature.T), 0.5)
                logits_2 = torch.div(torch.matmul(out_txt, global_img_feature.T), 0.5)

                labels = torch.tensor(d_idx).cuda()

                loss_1 = criterion(logits_1, labels)
                loss_2 = criterion(logits_2, labels)
                loss = loss_1 + loss_2

                self.optimizer.zero_grad()
                loss.backward()

                self.optimizer.step()

    def set_client_log(self, args, logkey):
        log_dir = const.DEFAULT_LOG_DIR
        if const.LOG_PATH_KEY in args.keys():
            log_dir = args[const.LOG_PATH_KEY]
        log_path = os.path.join(log_dir, self.algorithm, logkey)
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        self.clog = Attender(
            index=const.CLIENT_ + self.id,
            filePath=os.path.join(log_path, const.CLIENT_ + self.id + const.LOG_SUFFIX),
        )

    def test(self):
        if self.modal == 'multimodal':
            return
        def printnreset(name):
            self.clog.info(
                "TTTEST:  Epoch: [{0}] {1}\t"
                "Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t"
                "Prec@5 {top5.val:.3f} ({top5.avg:.3f})".format(
                    self.epochs, name, top1=self.test_top1, top5=self.test_top5
                )
            )

            self.losses = AverageMeter()
            self.test_top1 = AverageMeter()
            self.test_top5 = AverageMeter()

        self.model.eval()

        with torch.no_grad():
            for i, data in enumerate(self.public_test_data):
                if self.dataset == "cifar100" or self.dataset == "cifar10":
                    inputs_bt, labels_bt = data  # <FloatTensor> <LongTensor>
                    inputs_var = inputs_bt.to(self.device)
                    fvec, _, class_weight, _ = self.model(inputs_var)
                elif (
                    self.dataset == "agnews"
                    or self.dataset == "YelpReviewPolarity"
                    or self.dataset == "agnews"
                ):
                    (inputs_bt, caplens), labels_bt = data
                    inputs_bt = inputs_bt.to(self.device)
                    caplens = caplens.to(self.device)
                    fvec, _, class_weight, _ = self.model(inputs_bt, caplens)

                # # on_hot vector
                # labels_var_one_hot = to_one_hot(labels_var, n_dims=self.classSize)
                # # inter_class_distance
                # fvec = fvec - self.inter_distance * labels_var_one_hot.to(self.gpuid)
                if (
                    self.dataset.lower() == "cifar100".lower()
                    or self.dataset.lower() == "cifar10".lower()
                ):
                    prec1, prec5 = self.accuracy(fvec.data, labels_bt, topk=(1, 5))
                elif (
                    self.dataset.lower() == "AG_NEWS".lower()
                    or self.dataset.lower() == "agnews"
                ):
                    prec1, prec5 = self.accuracy(fvec.data, labels_bt, topk=(1, 4))
                elif self.dataset == "YelpReviewPolarity":
                    prec1, prec5 = self.accuracy(fvec.data, labels_bt, topk=(1, 2))
                self.test_top1.update(prec1[0], inputs_bt.size(0))
                self.test_top5.update(prec5[0], inputs_bt.size(0))

        printnreset(self.dataset)
        self.model.train()

    def printn_and_reset(self, name):
        self.clog.info(
            "Epoch: [{0}] {1}\t"
            "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
            "Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t"
            "Prec@5 {top5.val:.3f} ({top5.avg:.3f})".format(
                self.epochs, name, loss=self.losses, top1=self.top1, top5=self.top5
            )
        )
        self.losses = AverageMeter()
        self.top1 = AverageMeter()
        self.top5 = AverageMeter()

    def load_data_by_modal(self, is_train=True):
        if self.modal == "multimodal":
            return self.read_mm_data(is_train)
        if is_train:
            return self.load_train_data()
        else:
            return self.load_test_data()

    def load_train_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        train_data = data.read_client_data(
            self.dataset, self.modal_id, self.dataset_dir, is_train=True
        )
        collate_fn = None
        # if self.dataset == 'agnews':
        #     collate_fn = caption_collate_fn
        return DataLoader(
            train_data,
            batch_size=self.batch_size,
            drop_last=True,
            shuffle=True,
            collate_fn=collate_fn,
        )

    def load_test_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        test_data = data.read_client_data(
            self.dataset, self.modal_id, self.dataset_dir, is_train=False
        )
        collate_fn = None
        # if self.dataset == 'agnews':
        #     collate_fn = caption_collate_fn
        return DataLoader(
            test_data,
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=True,
            collate_fn=collate_fn,
        )

    def read_mm_data(self, is_train=True):
        image_path = "repository/flickr30k/flickr30k-images"
        total_data_path = "repository/flickr30k/flickr30k_data.pkl"
        total_data = load_pkl(total_data_path)
        tag = "train"
        if not is_train:
            tag = "test"
        client_data = load_pkl(
            os.path.join("repository/flickr30k/client_{}.pkl".format(tag))
        )
        data_indice = client_data[self.modal_id]
        self.train_samples = len(data_indice)
        mm_dataloader = MultimodalDataset(
            indices=data_indice,
            total_data=total_data,
            tag=tag,
            image_path=image_path,
        )
        return DataLoader(
            dataset=mm_dataloader, batch_size=self.batch_size, shuffle=True,collate_fn=image_to_caption_collate_fn,
        )

    def train_in_cifar(self, train_loader):
        for _, x, y in enumerate(train_loader):
            if isinstance(x, list):
                x = x[0].to(self.device)
            else:
                x = x.to(self.device)
            y = y.to(self.device)
            output1, _, class_weight, _ = self.model(x)
            y = y.to(torch.int64)
            # generate one hot encoding for y
            # y1hot = functional.one_hot(y, num_classes=self.num_classes).to(self.device)
            y1hot = to_one_hot(y, n_dims=self.num_classes)
            output1 = output1 - self.inter_distance * y1hot
            loss = self.loss(output1, y)
            center_loss = self.loss(
                torch.mm(class_weight, torch.t(class_weight)),
                self.center_labels_pos.type(torch.int64).to(self.device),
            )
            total_loss = 0.5 * center_loss + loss
            self.cal_differen_dataset_accuracy(x, output1, y)
            self.losses.update(total_loss.item(), x.size(0))
            total_loss.backward()
            self.optimizer.step()

    def train_in_agnews(self, train_loader):
        for _, data in enumerate(train_loader):
            (x, caplens), y = data
            caplens = caplens.to(self.device)
            if isinstance(x, list):
                x = x[0].to(self.device)
            else:
                x = x.to(self.device)
            y = y.to(self.device)
            fvec, _, class_weight, _ = self.model(x, caplens)
            labels_var_one_hot = to_one_hot(y, n_dims=self.num_classes)
            # inter_class_distance
            fvec = fvec - self.inter_distance * labels_var_one_hot.to(self.device)
            # intra_class_distance
            loss = self.loss(fvec, y)
            center_loss = self.loss(
                torch.mm(class_weight, torch.t(class_weight)),
                self.center_labels_pos.to(self.device),
            )
            total_loss = 0.5 * center_loss + loss
            self.cal_differen_dataset_accuracy(x, fvec, y)

            self.losses.update(total_loss.item(), x.size(0))
            total_loss.backward()
            self.optimizer.step()

    def train_with_global_representations(
        self, global_img_feature, global_txt_feature, distill_index, public_data_loader
    ):
        self.model.train()
        train_loader = self.load_data_by_modal()
        if self.dataset == "cifar10":
            self.train_in_cifar(train_loader=train_loader)
        elif self.dataset == "agnews":
            self.train_in_agnews(train_loader=train_loader)
            
        if self.contrast_local_inter and self.contrast_local_intra:
            global_img_feature, global_txt_feature = (
                global_img_feature.cuda(),
                global_txt_feature.cuda(),
            )
            distill_dict = {
                b: a for a, b in enumerate(distill_index)
            }  # index in coco to index to list 'distill_index'
            self.old_model.phase = "extract_conv_feature"
            self.old_model.is_train = False
            self.model.phase = "extract_conv_feature"
            self.model.is_train = False
            self.clog.info("Start Intra & Inter Contrasting!")
            for idx, (
                images,
                captions,
                captions_words,
                caption_lens,
                _,
                _,
                index,
            ) in tqdm(enumerate(public_data_loader), total=len(public_data_loader)):
                self.optimizer.zero_grad()
                d_idx = operator.itemgetter(*index)(distill_dict)
                if self.dataset == "cifar100" or self.dataset == "cifar10":
                    images = images.to(self.device)
                    img_feature = self.model(images)
                    target_feature = global_img_feature[d_idx, :].type_as(
                        img_feature[0]
                    )
                    # negative output
                    with torch.no_grad():
                        old_img_feature = self.old_model(images)
                    logits_inter = torch.div(
                        torch.matmul(img_feature, global_txt_feature.T), 0.5
                    )
                elif self.dataset == "agnews" or self.dataset == "YelpReviewPolarity":
                    captions = captions.to(self.device)
                    caption_lens = caption_lens.to(self.device)
                    img_feature = self.model(captions, caption_lens).squeeze()
                    target_feature = global_txt_feature[d_idx, :].type_as(img_feature)
                    # negative output
                    with torch.no_grad():
                        old_img_feature = self.old_model(
                            captions, caption_lens
                        ).squeeze()

                    logits_inter = torch.div(
                        torch.matmul(img_feature, global_img_feature.T), 0.5
                    )
                labels_inter = torch.tensor(d_idx).cuda()
                loss_inter = self.loss(logits_inter, labels_inter)

                pos = torch.sum(img_feature * target_feature, dim=-1)
                pos = pos.reshape(-1, 1)

                neg = torch.sum(img_feature * old_img_feature, dim=-1)
                logits = torch.cat((pos, neg.reshape(-1, 1)), dim=1)

                logits /= 0.5
                labels = torch.zeros(images.size(0)).cuda().long()

                loss_moon = self.loss(logits, labels)
                # if not self.args.loss_scale:
                #     loss = (loss_moon + loss_inter) * self.inter_intra_weight
                # else:
                loss = (
                    loss_moon + loss_inter / (loss_inter / loss_moon).detach()
                ) * self.inter_intra_weight

                loss.backward()
                self.optimizer.step()
            self.old_model.phase = "None"
            self.old_model.is_train = True
            self.model.phase = "None"
            self.model.is_train = True

        elif self.contrast_local_intra:
            global_img_feature, global_txt_feature = (
                global_img_feature.cuda(),
                global_txt_feature.cuda(),
            )
            distill_dict = {b: a for a, b in enumerate(distill_index)}
            self.old_model.phase = "extract_conv_feature"
            self.old_model.is_train = False
            self.model.phase = "extract_conv_feature"
            self.model.is_train = False
            self.clog.info("Start Intra-modal Contrasting!")
            for idx, (
                images,
                captions,
                captions_word,
                caption_lens,
                _,
                _,
                index,
            ) in tqdm(enumerate(public_data_loader), total=len(public_data_loader)):
                self.optimizer.zero_grad()
                d_idx = operator.itemgetter(*index)(distill_dict)  # batchidx
                if self.dataset == "cifar100" or self.dataset == "cifar10":
                    images = images.to(self.device)
                    im_feature = self.model(images)
                    target_feature = global_img_feature[d_idx, :].type_as(im_feature[0])
                    # neg
                    with torch.no_grad():
                        old_im_feature = self.old_model(images)
                elif self.dataset == "agnews" or self.dataset == "YelpReviewPolarity":
                    captions = captions.to(self.device)
                    caption_lens = caption_lens.to(self.device)
                    im_feature = self.model(captions, caption_lens).squeeze()
                    target_feature = global_txt_feature[d_idx, :].type_as(im_feature)
                    # neg
                    with torch.no_grad():
                        old_im_feature = self.old_model(
                            captions, caption_lens
                        ).squeeze()
                os = torch.sum(im_feature * target_feature, dim=-1)
                pos = pos.reshape(-1, 1)
                # neg
                # neg = cos(im_feature, old_im_feature)
                neg = torch.sum(im_feature * old_im_feature, dim=-1)
                logits = torch.cat((pos, neg.reshape(-1, 1)), dim=1)

                logits /= 0.5  # temperature
                labels = torch.zeros(images.size(0)).cuda().long()

                loss = self.loss(logits, labels)
                loss.backward()
                self.optimizer.step()
            self.old_model.phase = "None"
            self.old_model.is_train = True
            self.model.phase = "None"
            self.model.is_train = True

        elif self.contrast_local_inter:
            global_img_feature, global_txt_feature = (
                global_img_feature.cuda(),
                global_txt_feature.cuda(),
            )
            distill_dict = {
                b: a for a, b in enumerate(distill_index)
            }  # index in coco to index to list 'distill_index'
            self.model.phase = "extract_conv_feature"
            self.model.is_train = False
            # Contrast
            self.clog.info("Start Inter-modal Contrasting!")
            for idx, (
                images,
                captions,
                captions_word,
                caption_lens,
                _,
                _,
                index,
            ) in tqdm(enumerate(public_data_loader), total=len(public_data_loader)):
                self.optimizer.zero_grad()
                d_idx = operator.itemgetter(*index)(distill_dict)  # batchidx
                if self.dataset == "cifar100" or self.dataset == "cifar10":
                    images = images.to(self.device)
                    im_feature = self.model(images)
                    logits = torch.div(
                        torch.matmul(im_feature, global_txt_feature.T), 0.5
                    )
                elif self.dataset == "agnews" or self.dataset == "YelpReviewPolarity":
                    captions = captions.to(self.device)
                    caption_lens = caption_lens.to(self.device)
                    im_feature = self.model(captions, caption_lens).squeeze()
                    logits = torch.div(
                        torch.matmul(im_feature, global_img_feature.T), 0.5
                    )

                labels = torch.tensor(d_idx).cuda()

                loss = self.loss(logits, labels)
                loss.backward()
                self.optimizer.step()
            self.model.phase = "None"
            self.model.is_train = True

    def multimodal_generate_logits(self, dataloader):
        self.model.cuda()
        self.model.eval()
        with torch.no_grad():
            img_vec = []
            txt_vec = []
            distill_index = []
            for idx, (images, captions, captions_word, caption_lens, _, _, index) in tqdm(enumerate(dataloader),
                                                                           total=len(dataloader)):
                images = images.to(self.device)
                captions = captions.to(self.device)
                caption_lens = caption_lens.to(self.device)

                output = self.model(images, captions, captions_word, caption_lens)

                out_img = output['image_features'].sum(axis=1) if len(output['image_features'].shape) == 3 else output[
                    'image_features']
                out_txt = output['caption_features'].sum(axis=1) if len(output['caption_features'].shape) == 3 else \
                    output['caption_features']
                img_vec.extend(out_img)
                txt_vec.extend(out_txt)
                distill_index.extend(index)

        img_vec = torch.cat(img_vec, dim=0).view(-1, self.args.feature_dim)
        txt_vec = torch.cat(txt_vec, dim=0).view(-1, self.args.feature_dim)

        img_vec = img_vec.cpu()
        txt_vec = txt_vec.cpu()
        self.model.cpu()

        return {'img': img_vec, 'txt': txt_vec}, distill_index
    
    def nonmm_generate_logits(self,dataloader):
        vec, idx = self.extract_pub_feature(dataloader)
        if self.dataset == "cifar100" or self.dataset == "cifar10":
            return {"img": vec, "txt": None}, idx
        elif self.dataset == "agnews" or self.dataset == "YelpReviewPolarity":
            return {"img": None, "txt": vec}, idx
        else:
            assert False
    
    def generate_logits(self, dataloader):
        if self.modal == 'multimodal':
            return self.multimodal_generate_logits(dataloader)
        else:
            return self.nonmm_generate_logits(dataloader)

    def extract_pub_feature(self, dataloader):
        self.model.cuda()

        self.model.phase = "extract_conv_feature"
        self.model.is_train = False
        feature = []
        distill_index = []
        # iterate batch
        for idx, (images, captions, captions_word, caption_lens, _, _, index) in tqdm(
            enumerate(dataloader), total=len(dataloader)
        ):
            with torch.no_grad():
                if self.dataset == "cifar100" or self.dataset == "cifar10":
                    images = images.to(self.device)
                    im_feature = self.model(images)

                elif self.dataset == "agnews" or self.dataset == "YelpReviewPolarity":
                    captions = captions.to(self.device)
                    caption_lens = caption_lens.to(self.device)
                    im_feature = self.model(captions, caption_lens).squeeze()

                im_feature = im_feature.cpu().detach()
                feature.append(im_feature)
                distill_index.extend(index)
                # print(f'im_feature {im_feature.shape} labels {labels_var.shape}')
                # if is_test and idx == 1:
                #     break

        feature = torch.cat(feature, dim=0)
        # print(f'feature {feature.shape} labels {labels.shape}')
        self.model.phase = "None"
        self.model.is_train = True

        self.model.cpu()
        return feature, distill_index

    def cal_differen_dataset_accuracy(self, x, output_data, y):
        if self.dataset == "cifar100" or self.dataset == "cifar10":
            prec1, prec5 = self.accuracy(output_data, y, topk=(1, 5))
        elif self.dataset == "agnews":
            prec1, prec5 = self.accuracy(output_data, y, topk=(1, 4))
        self.top1.update(prec1[0], x.size(0))
        self.top5.update(prec5[0], x.size(0))

    def accuracy(self, output, target, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        temp = target.view(1, -1).expand_as(pred)
        temp = temp.to(self.device)
        correct = pred.eq(temp)

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

    def set_client_log(self, args, logkey):
        log_dir = const.DEFAULT_LOG_DIR
        if const.LOG_PATH_KEY in args.keys():
            log_dir = args[const.LOG_PATH_KEY]
        log_path = os.path.join(log_dir, self.algorithm, logkey)
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        self.clog = Attender(
            index=const.CLIENT_ + self.id,
            filePath=os.path.join(log_path, const.CLIENT_ + self.id + const.LOG_SUFFIX),
        )


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def to_one_hot(y, n_dims=None):
    """Take integer y (tensor or variable) with n dims and convert it to 1-hot representation with n+1 dims."""
    y_tensor = y.data if isinstance(y, torch.autograd.Variable) else y
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    # y_one_hot = y_one_hot.view(y.shape, -1)
    return (
        torch.autograd.Variable(y_one_hot)
        if isinstance(y, torch.autograd.Variable)
        else y_one_hot
    )


if __name__ == "__main__":
    import numpy as np

    index = np.array([0, 3, 4])
    nparray = np.array([1, 2, 3, 4, 5])
    # nparray = nparray.tolist()
    print(nparray[index])
