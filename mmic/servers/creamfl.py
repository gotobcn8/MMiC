import const.constants as const
from datetime import datetime
from fedlog.logbooker import Attender
from datetime import datetime

# from logbook.more import ColorizedStderrHandler
from fedlog.logbooker import ColoredConsoleHandler
from clients.creamfl import CreamFl
from utils.data import read_client_data
import os
import torch
import torch.nn as nn
import gc
import operator
import numpy as np
from algorithm.convert import to_numpy

# from ..dataset.coco_generator import prepare_coco_dataloaders
from utils.config import parse_config
from dataset.coco_generator import _get_coco_loader, load_vocab, _get_coco_file_paths
from const.supporter import MultimodalDatasets
from models.loss import ms
from tqdm import tqdm


class CreamFL:
    def __init__(self, args):
        self.algorithm = args["algorithm"]
        if self.algorithm in args["fedAlgorithm"].keys():
            self.fedAlgorithm = args["fedAlgorithm"][self.algorithm]
        # define log
        # 获取当前时间
        self.set_server_log(args)
        self.args = args
        self.global_model = args["model"]["multimodal"]
        self.device = args["device"]
        self.dataset = args["dataset"]
        self.num_clients = args["num_clients"]
        self.join_ratio = args["join_ratio"]
        self.client_drop_rate = args["client_drop_rate"]
        self.learning_rate = args["learning_rate"]

        self.epochs = args["epochs"]
        self.global_rounds = args["global_rounds"]
        self.time_threthold = args["time_threthold"]
        self.rs_test_acc = []
        self.rs_test_auc = []
        self.rs_train_loss = []
        # dataset 6
        self.dataset_dir = const.DIR_DEPOSITORY
        # if 'dataset_dir' in args.keys():
        #     self.dataset_dir = args['dataset_dir']
        # join clients setting

        self.datasets_by_target = self.args[self.args["dataset"]]
        if "server" in self.datasets_by_target.keys():
            self.dataset = self.datasets_by_target["server"]
        self.batch_size = self.args[self.dataset]["batch_size"]
        self.set_local_dataset()
        # client
        self.new_clients_settings = args["new_clients"]
        self.random_clients_selected = args["random_clients_selected"]
        self.new_clients_rate = self.new_clients_settings["rate"]
        self.set_for_new_clients()
        self.num_original_clients = self.num_clients - self.num_new_clients
        self.num_join_clients = self.num_original_clients * self.join_ratio
        self.num_new_clients = int(self.num_clients * self.new_clients_rate)
        self.late_clients = []
        self.all_clients = []
        self.clients = []
        self.set_clients()
        # algorithm
        self.contrast_local_intra = self.fedAlgorithm["is_intra_constrast"]
        self.contrast_local_inter = self.fedAlgorithm["is_inter_constrast"]
        self.kd_weight = self.fedAlgorithm["kd_weight"]
        self.inter_intra_weight = self.fedAlgorithm["inter_intra_weight"]
        self.disabled_distill = self.fedAlgorithm["disabled_distill"]
        # false temporary
        self.eval_new_clients = False
        self.fine_tuning_epoch = 0
        self.algorithm = args["algorithm"]
        self.eval_gap = args["eval_gap"]
        self.budget = []

        self.save_dir = "results"
        self.save_models_dir = os.path.join(os.environ["HOME"], "models", self.logkey)
        if "save_dir" in args.keys() and args["save_dir"] != "":
            self.save_models_dir = args["save_dir"]
            self.save_dir = args["save_dir"]
        # set model
        self.set_model(args)
        # save model
        if "save_dir" in args.keys() and args["save_dir"] != "":
            self.save_models_dir = args["save_dir"]
            self.save_dir = args["save_dir"]
        self.criterion = ms.MCSoftContrastiveLoss()
        # cluster
        self.clusters = []
        self.clients_map_clusters = dict()
        self.cluster_map_clients = []
        self.cluster_ids = []
        # clients info collection
        self.collections = [dict() for _ in range(self.num_clients)]

        # evaluation
        self.best_score = 0

    def set_for_new_clients(self):
        self.num_new_clients = 0

    def set_model(self, args):
        self.global_model = args["model"]["multimodal"]
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.global_model.parameters(), lr=self.learning_rate
        )

    def set_local_dataset(self):
        """
        NEED TO BE SUPPLY
        """
        self.get_server_data()
        self.train_subset = self.dataloader["train_subset_50000"]
        self.train_eval_subset = self.dataloader["train_subset_eval_50000"]
        self.test_dataset = None

    def set_clients(self):
        self.clients = []
        self.clients_by_modal = dict()
        cid_cnt = 0
        for modal, client_num in self.args["client_settings"].items():
            for i in range(client_num):
                train_samples, test_samples = 0, 0
                if (
                    modal != "multimodal"
                    and self.args[self.algorithm][modal] not in MultimodalDatasets
                ):
                    train_data = read_client_data(
                        self.datasets_by_target[modal],
                        i,
                        self.dataset_dir,
                        is_train=True,
                    )
                    test_data = read_client_data(
                        self.datasets_by_target[modal],
                        i,
                        self.dataset_dir,
                        is_train=False,
                    )
                    train_samples, test_samples = len(train_data), len(test_data)

                tmp_client = CreamFl(
                    self.args,
                    id=const.ORIGINAL + str(i),
                    modal=modal,
                    modal_id=i,
                    train_samples=train_samples,
                    test_samples=test_samples,
                    serial_id=cid_cnt,
                    logkey=self.logkey,
                )
                self.clients.append(tmp_client)
                if modal not in self.clients_by_modal.keys():
                    self.clients_by_modal[modal] = []
                self.clients_by_modal[modal].append(tmp_client)
                cid_cnt += 1
        self.all_clients.extend(self.clients)

    def set_server_log(self, args):
        self.logkey = (
            args["dataset"] + "_" + datetime.now().strftime(const.LOG_DIR_TIME_FORMAT)
        )
        log_dir = const.DEFAULT_LOG_DIR
        if const.LOG_PATH_KEY in args.keys():
            log_dir = args[const.LOG_PATH_KEY]
        log_path = os.path.join(log_dir, self.algorithm, self.logkey)
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        self.slog = Attender(
            index=const.SERVER_KEY,
            filePath=os.path.join(log_path, const.SERVER_KEY + const.LOG_SUFFIX),
            handlers=[
                ColoredConsoleHandler(bubble=True, format_string=const.LOG_FILE_FORMAT)
            ],
        )

    # def set_config(self, img='cifa100', txt='AG_NEWS'):
    #     self.config = None
    #     self.config.train.output_file = 'model_noprob'

    #     self.config.model.embed_dim = self.args.feature_dim  # set global model dim

    #     if self.args.not_bert:
    #         self.config.model.not_bert = True
    #         self.config.model.cnn_type = 'resnet50'
    #     else:
    #         self.config.model.not_bert = False
    #         self.config.model.cnn_type = 'resnet101'

    def get_server_data(self):
        """
        datatype is one of [pub_data, train_data, test_data]\n
        Some work need to be done...
        """
        # return DataLoader()
        self.dataloader, self.vocab = self.prepare_coco_dataloaders(
            dataset_root="repository/coco/",
            vocab_path="repository/vocabs/coco_vocab.pkl",
        )

    def train(self):
        for round in range(self.global_rounds):
            # firstly global train representations
            self.global_train()
            client_img_vect, client_img_num = [], []
            client_txt_vect, client_txt_num = [], []
            selected_clients = self.select_clients()
            self.slog.info("Generate Local Representations")
            for client in selected_clients:
                client.train(
                    self.global_img_feature,
                    self.global_txt_feature,
                    self.distill_index,
                    self.train_subset,
                )
                vect, i = client.generate_logits(self.train_eval_subset)
                if self.distill_index is not None:
                    assert i == self.distill_index
                if vect["img"] is not None:
                    client_img_vect.append(vect["img"])
                    client_img_num.append(client.train_samples)
                if vect["txt"] is not None:
                    client_txt_vect.append(vect["txt"])
                    client_txt_num.append(client.train_samples)
            if self.disabled_distill:
                self.distill(
                    client_img_vect,
                    client_txt_vect,
                    client_img_num,
                    client_txt_num,
                    self.distill_index,
                )

            def get_lr(optimizer):
                for param_group in optimizer.param_groups:
                    return param_group["lr"]

            test_scores = self.evaluate({"test": self._dataloaders["test"]})
     
            rsum = (
                test_scores["test"]["n_fold"]["i2t"]["recall_1"]
                + test_scores["test"]["n_fold"]["t2i"]["recall_1"]
                + test_scores["test"]["i2t"]["recall_1"]
                + test_scores["test"]["t2i"]["recall_1"]
            )
            if self.best_score < rsum:
                best_score = rsum
                # default path set in code
                if not os.path.exists(self.save_models_dir):
                    os.makedirs(self.save_models_dir)
                torch.save(
                    {"net": self.global_model.state_dict()},
                    os.path.join(
                        self.save_models_dir, "best_model_" + str(round) + ".pt"
                    ),
                )
                self.slog.info(
                    "In the rounds of {}, we got best score {}".format(
                        round, best_score
                    )
                )
            if round == self.global_rounds - 1:
                torch.save(
                    {"net": self.global_model.state_dict()},
                    self.save_models_dir + "last_model.pt",
                )
                self.slog.info("final score", rsum)

    def select_clients(self, is_late_attended=False):
        self.slog.info("Starting select clients for server")
        if self.random_clients_selected:
            # random number of attend clients
            self.current_num_join_clients = np.random.choice(
                int(self.num_original_clients * self.join_ratio),
                self.num_original_clients + 1,
            )
        else:
            # static number of attend clients
            self.current_num_join_clients = (
                len(self.clients) * (1 - self.client_drop_rate) * self.join_ratio
            )
        selected_clients = list(
            np.random.choice(
                self.clients, int(self.current_num_join_clients), replace=False
            )
        )
        return selected_clients

    def global_train(self):
        # first step: we train global model firstly
        self.global_model.train()
        torch.cuda.empty_cache()
        server_dataloader = self.train_subset
        # for i in range(self.epochs):
        self.slog.info("starting global training")
        for idx, (images, captions, captions_word, caption_lens, a_, b_, index) in tqdm(
            enumerate(server_dataloader), total=len(server_dataloader)
        ):
            images = images.to(self.device)  # [bs, 3, 224, 224]
            captions = captions.to(self.device)  # [bs, seq_len]
            caption_lens = caption_lens.to(self.device)

            output = self.global_model(images, captions, captions_word, caption_lens)
            loss, _ = self.criterion(**output)

            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()

        if self.contrast_local_inter or self.contrast_local_intra:
            img_feature, txt_feature = [], []
            distill_index = []
            self.global_model.eval()
            for idx, (
                images,
                captions,
                captions_word,
                caption_lens,
                _,
                _,
                index,
            ) in tqdm(
                enumerate(self.train_eval_subset), total=len(self.train_eval_subset)
            ):
                with torch.no_grad():
                    images = images.to(self.device)
                    captions = captions.to(self.device)
                    caption_lens = caption_lens.to(self.device)

                    output = self.global_model(
                        images, captions, captions_word, caption_lens
                    )
                    out_img = output["image_features"]
                    out_txt = output["caption_features"]

                    out_img = out_img.cpu().detach()
                    out_txt = out_txt.cpu().detach()

                    img_feature.append(out_img)
                    txt_feature.append(out_txt)
                    distill_index.extend(index)
            self.global_img_feature = torch.concat(img_feature, dim=0)
            self.global_txt_feature = torch.concat(txt_feature, dim=0)
            print(self.global_txt_feature.shape, self.global_img_feature.shape)
            self.distill_index = distill_index
            del img_feature, txt_feature
            gc.collect()

    def distill(
        self, img_vec, txt_vec, img_num, txt_num, distill_index, agg_method="con_w"
    ):
        self.global_model.train()
        client_loss_criterion = nn.MSELoss()
        # aggregation
        img_vec, txt_vec = self.aggregation(
            img_vec, txt_vec, img_num, txt_num, agg_method
        )

        self.img_vec = img_vec
        self.txt_vec = txt_vec

        distill_dict = {
            b: a for a, b in enumerate(distill_index)
        }  # index in coco to index to list 'distill_index'
        # distill
        self.slog.log("start distilling")
        for _, (
            images,
            captions,
            captions_word,
            caption_lens,
            _,
            _,
            index,
        ) in self.get_server_data("pub_data"):
            images = images.to(self.device)  # [bs, 3, 224, 224]
            captions = captions.to(self.device)  # [bs, seq_len]
            caption_lens = caption_lens.to(self.device)

            output = self.global_model(images, captions, captions_word, caption_lens)
            loss = 0

            if (
                "image" in self.clients_by_modal.keys()
                and len(self.clients_by_modal["image "]) > 0
            ):
                out_img = output["image_features"]
                d_idx = operator.itemgetter(*index)(
                    distill_dict
                )  # idx of the current batch
                target_img = self.img_vec[d_idx, :].type_as(out_img)
                loss += self.kd_weight * self.code_sim(
                    out_img, target_img, client_loss_criterion
                )
            if (
                "text" in self.clients_by_modal.keys()
                and len(self.clients_by_modal["text"]) > 0
            ):
                out_txt = output["caption_features"]
                d_idx = operator.itemgetter(*index)(
                    distill_dict
                )  # idx of the current batch
                target_txt = self.txt_vec[d_idx, :].type_as(out_txt)
                loss += self.kd_weight * self.code_sim(
                    out_txt, target_txt, client_loss_criterion
                )
            if (
                "multimodal" in self.clients_by_modal.keys()
                and len(self.clients_by_modal["multimodal"]) > 0
            ):
                out_img = output["image_features"]
                d_idx = operator.itemgetter(*index)(
                    distill_dict
                )  # idx of the current batch
                target_img = self.img_vec[d_idx, :].type_as(out_img)
                out_txt = output["caption_features"]
                target_txt = self.txt_vec[d_idx, :].type_as(out_txt)
                loss += self.kd_weight * self.code_sim(
                    out_img, target_img, client_loss_criterion
                )
                loss += self.kd_weight * self.code_sim(
                    out_txt, target_txt, client_loss_criterion
                )

            self.optimizer.zero_grad()

            loss.backward()

            # if self.config.train.grad_clip > 0:
            #     nn.utils.clip_grad.clip_grad_norm_(self.global_model.parameters(),
            #                                        self.config.train.grad_clip)
            self.optimizer.step()

    def aggregation(self, i_vec, t_vec, i_num, t_num, agg_method="con_w"):
        if agg_method == "con_w":
            contrastive_w = []
            for (
                vec
            ) in (
                i_vec
            ):  # vec: [50000, n_feature], global_txt_feature: [50000, n_feature]
                logits = torch.matmul(vec, self.global_txt_feature.T)  # [50000, 50000]
                exp_logits = torch.exp(logits)
                log_prob = logits - torch.log(
                    torch.sum(exp_logits, dim=1, keepdim=True)
                )
                contrastive_w.append(torch.diagonal(log_prob).reshape(1, -1))
            contrastive_w = torch.softmax(torch.cat(contrastive_w, dim=0), dim=0)
            for i in range(len(i_vec)):
                i_vec[i] = (i_vec[i] * contrastive_w[i].reshape(-1, 1)).unsqueeze(0)
            i_vec = torch.sum(
                torch.cat(i_vec, dim=0), dim=0
            )  # aggregated image vectors

            contrastive_w = []
            for (
                vec
            ) in (
                t_vec
            ):  # vec: [50000, n_feature], global_txt_feature: [50000, n_feature]
                logits = torch.matmul(vec, self.global_img_feature.T)  # [50000, 50000]
                exp_logits = torch.exp(logits)
                log_prob = logits - torch.log(
                    torch.sum(exp_logits, dim=1, keepdim=True)
                )
                contrastive_w.append(torch.diagonal(log_prob).reshape(1, -1))
            contrastive_w = torch.softmax(torch.cat(contrastive_w, dim=0), dim=0)
            for i in range(len(t_vec)):
                t_vec[i] = (t_vec[i] * contrastive_w[i].reshape(-1, 1)).unsqueeze(0)
            t_vec = torch.sum(torch.cat(t_vec, dim=0), dim=0)  # aggregated text vectors
        else:
            raise NotImplementedError

        return i_vec, t_vec

    def code_sim(self, output, target, criterion):
        output = output.sum(axis=1) if len(output.shape) == 3 else output
        target = target.type_as(output)

        return criterion(output, target.type_as(output))

    @torch.no_grad()
    def evaluate(self, val_loaders, n_crossfolds=None, **kwargs):

        # self.model_to_device()
        self.global_model.eval()

        if not isinstance(val_loaders, dict):
            val_loaders = {"te": val_loaders}

        scores = {}
        val_dataloader = self.get_server_data("validate")
        # for key, data_loader in val_loaders.items():
        #     if key == "train" or key == "train_subset" or key == "train_subset_eval" or "train" in key:
        #         continue
        #     self.slog.log('Evaluating validate data')
        _n_crossfolds = n_crossfolds
        scores["validate"] = self.evaluate_metrics(
            val_dataloader, n_crossfolds=_n_crossfolds, key="validate", **kwargs
        )
        return scores

    def evaluate_metrics(
        self,
        dataloader,
        n_crossfolds=None,
        n_images_per_crossfold=1000,
        n_captions_per_crossfold=5000,
        eval_batch_size=1024,
    ):
        """evaluate image-to-caption and caption-to-image retrieval tasks."""
        scores = {}

        self.slog.info("extracting features...")

        extracted_features = self.extract_features(dataloader)

        image_features = extracted_features["image_features"]
        caption_features = extracted_features["caption_features"]
        image_sigmas = extracted_features["image_sigmas"]
        caption_sigmas = extracted_features["caption_sigmas"]
        image_classes = extracted_features["image_classes"]
        caption_classes = extracted_features["caption_classes"]

        scores["mean_log_image_sigma"] = np.mean(image_sigmas)
        scores["mean_log_caption_sigma"] = np.mean(caption_sigmas)

        if n_crossfolds > 0:
            n_fold_scores = self.evaluate_n_fold(
                extracted_features,
                n_crossfolds,
                n_images_per_crossfold,
                n_captions_per_crossfold,
                eval_batch_size,
            )
            scores["n_fold"] = n_fold_scores

        self.slog.info("evaluating i2t...")
        scores["i2t"] = self.evaluate_recall(
            image_features,
            caption_features,
            image_classes,
            caption_classes,
            batch_size=eval_batch_size,
        )
        self.slog.info("evaluating t2i...")
        scores["t2i"] = self.evaluate_recall(
            caption_features,
            image_features,
            caption_classes,
            image_classes,
            batch_size=eval_batch_size,
        )
        for key in ("rsum", "medr", "meanr"):
            scores[key] = scores["i2t"][key] + scores["t2i"][key]
        return scores

    @torch.no_grad()
    def extract_features(self, dataloader):
        """Extract image and caption features using the given model.

        Args:
            model (nn.Module): a model to extract features.
            dataloader (data.Dataloader): the target dataloader to feature extraction.
        """
        self.model.eval()
        self.model.to(self.extract_device)

        num_images = dataloader.dataset.n_images
        num_captions = len(dataloader.dataset)

        image_classes = np.zeros(num_images)
        caption_classes = np.zeros(num_captions)

        image_features = np.zeros((num_images, self.n_embeddings, self.feat_size))
        caption_features = np.zeros((num_captions, self.n_embeddings, self.feat_size))

        image_sigmas = np.zeros((num_images, self.feat_size))
        caption_sigmas = np.zeros((num_captions, self.feat_size))

        image_ids_ = np.zeros(num_images)
        caption_ids = np.zeros(num_captions)

        cur_image_idx = 0
        cur_caption_idx = 0
        seen_image_ids = set()
        iid_to_cls = dataloader.dataset.iid_to_cls

        def get_image_class(image_id):
            if iid_to_cls:
                image_class = iid_to_cls.get(image_id, image_id)
            else:
                image_class = image_id
            return image_class

        for (
            images,
            captions,
            captions_word,
            caption_lens,
            ann_ids,
            image_ids,
            _,
        ) in self.pbar(dataloader):
            images = images.to(self.extract_device)
            captions = captions.to(self.extract_device)
            caption_lens = caption_lens.to(self.extract_device)

            output = self.model(images, captions, captions_word, caption_lens)
            _image_features = output["image_features"]
            _caption_features = output["caption_features"]

            if output.get("image_logsigma") is not None:
                _image_sigmas = output["image_logsigma"]
                _caption_sigmas = output["caption_logsigma"]

            for idx, image_id in enumerate(image_ids):
                image_class = get_image_class(image_id)
                if image_id not in seen_image_ids:
                    image_ids_[cur_image_idx] = image_id
                    seen_image_ids.add(image_id)
                    image_classes[cur_image_idx] = image_class
                    image_features[cur_image_idx] = to_numpy(_image_features[idx])
                    if output.get("image_logsigma") is not None:
                        image_sigmas[cur_image_idx] = to_numpy(_image_sigmas[idx])
                    cur_image_idx += 1
                caption_ids[cur_caption_idx] = ann_ids[idx]
                caption_classes[cur_caption_idx] = image_class
                caption_features[cur_caption_idx] = to_numpy(_caption_features[idx])
                if output.get("image_logsigma") is not None:
                    caption_sigmas[cur_caption_idx] = to_numpy(_caption_sigmas[idx])
                cur_caption_idx += 1

        if iid_to_cls:
            print(
                f"Num images ({num_images}) -> Num classes ({len(set(image_classes))})"
            )
        if cur_image_idx != num_images:
            raise RuntimeError(
                "unexpected error, {} != {}".format(cur_image_idx, num_images)
            )
        if cur_caption_idx != num_captions:
            raise RuntimeError(
                "unexpected error, {}, {}".format(cur_caption_idx, num_captions)
            )
        if set(image_classes) != set(caption_classes):
            raise RuntimeError(
                "unexpected error, I({}) != C({})".format(
                    set(image_classes), set(caption_classes)
                )
            )

        if not iid_to_cls:
            # XXX this code is for aligning image features and caption features
            # but if you use classes as COCO classes, but image_id,
            # the sorted_caption_idx will return multiple instances, and
            # the results will be corrupted.
            sorted_caption_idx = []
            for image_class in image_classes:
                sorted_caption_idx.extend(np.where(caption_classes == image_class)[0])

            sorted_caption_idx = np.array(sorted_caption_idx)
            caption_ids = caption_ids[sorted_caption_idx]
            caption_classes = caption_classes[sorted_caption_idx]
            caption_features = caption_features[sorted_caption_idx]

        image_features = torch.from_numpy(image_features)
        caption_features = torch.from_numpy(caption_features)
        image_classes = torch.from_numpy(image_classes)
        caption_classes = torch.from_numpy(caption_classes)

        return {
            "image_features": image_features,
            "caption_features": caption_features,
            "image_sigmas": image_sigmas,
            "caption_sigmas": caption_sigmas,
            "image_ids": image_ids_,
            "caption_ids": caption_ids,
            "image_classes": image_classes,
            "caption_classes": caption_classes,
        }

    @torch.no_grad()
    def evaluate_recall(
        self,
        q_features,
        g_features,
        q_labels,
        g_labels,
        q_ids=None,
        g_ids=None,
        batch_size=1024,
    ):
        """Evaluate recall

        Args:
            q_features (tensor): N_q x d query features
            g_features (tensor): N_g x d gallery features
            q_labels (tensor): N query labels
            g_labels (tensor): N gallery labels
        """
        if len(q_features) != len(q_labels):
            raise RuntimeError(
                "length mismatch {}, {}".format(q_features.shape, q_labels.shape)
            )
        if len(g_features) != len(g_labels):
            raise RuntimeError(
                "length mismatch {}, {}".format(g_features.shape, g_labels.shape)
            )
        n_queries = len(q_labels)
        n_galleries = len(g_labels)
        best_pred_ranks = np.zeros(n_queries)

        pmm = ParallelMatMulModule()
        g_features = g_features.view(n_galleries * self.n_embeddings, -1).t()
        pmm.set_g_features(g_features)

        q_features = q_features.to(self.eval_device)

        for q_indices in range(n_queries):
            q_indices = np.array(q_indices)

            _q_feature = q_features[q_indices, :]
            _q_feature = _q_feature.view(len(q_indices) * self.n_embeddings, -1)
            _, pred_ranks = pmm(_q_feature, n_embeddings=self.n_embeddings)

            for idx, q_idx in enumerate(q_indices):
                pos_indices = np.where(g_labels == q_labels[q_idx])[0]
                _pred_ranks = [
                    torch.where(pred_ranks[idx] == pos_idx)[0][0].item()
                    for pos_idx in pos_indices
                ]
                best_pred_ranks[q_idx] = min(_pred_ranks)

        recall_1 = self.recall_at_k(best_pred_ranks, 1)
        recall_5 = self.recall_at_k(best_pred_ranks, 5)
        recall_10 = self.recall_at_k(best_pred_ranks, 10)
        medr = np.floor(np.median(best_pred_ranks)) + 1
        meanr = np.mean(best_pred_ranks) + 1

        scores = {
            "recall_1": recall_1,
            "recall_5": recall_5,
            "recall_10": recall_10,
            "rsum": recall_1 + recall_5 + recall_10,
            "medr": medr,
            "meanr": meanr,
        }

        return scores

    def recall_at_k(self, ranks, k):
        """Compute recall at K

        args:
            ranks (list): list of rankings of positive pairs
            k (int): k
        """
        return 100.0 * len(np.where(ranks < k)[0]) / len(ranks)

    def evaluate_n_fold(
        self,
        extracted_features,
        n_crossfolds,
        n_images_per_crossfold,
        n_captions_per_crossfold,
        eval_batch_size,
    ):
        image_features = extracted_features["image_features"]
        caption_features = extracted_features["caption_features"]
        image_classes = extracted_features["image_classes"]
        caption_classes = extracted_features["caption_classes"]

        n_fold_scores = {
            "i2t": {
                "recall_1": [],
                "recall_5": [],
                "recall_10": [],
                "rsum": [],
                "medr": [],
                "meanr": [],
            },
            "t2i": {
                "recall_1": [],
                "recall_5": [],
                "recall_10": [],
                "rsum": [],
                "medr": [],
                "meanr": [],
            },
        }

        for idx in range(n_crossfolds):
            # if self.logger:
            #     self.logger.log('evaluating {}-th fold'.format(idx + 1))

            _image_split = np.arange(
                idx * n_images_per_crossfold, (idx + 1) * n_images_per_crossfold
            )
            _image_features = image_features[_image_split]
            _image_classes = image_classes[_image_split]

            _caption_split = np.arange(
                idx * n_captions_per_crossfold, (idx + 1) * n_captions_per_crossfold
            )
            _caption_features = caption_features[_caption_split]
            _caption_classes = caption_classes[_caption_split]

            _scores = {}
            _scores["i2t"] = self.evaluate_recall(
                _image_features,
                _caption_features,
                _image_classes,
                _caption_classes,
                batch_size=eval_batch_size,
            )
            _scores["t2i"] = self.evaluate_recall(
                _caption_features,
                _image_features,
                _caption_classes,
                _image_classes,
                batch_size=eval_batch_size,
            )
            for _task, _task_scores in _scores.items():
                for key, val in _task_scores.items():
                    n_fold_scores[_task][key].append(val)
        n_fold_scores = {
            _task: {key: np.mean(np.array(val)) for key, val in _task_scores.items()}
            for _task, _task_scores in n_fold_scores.items()
        }
        return n_fold_scores

    #  COCO
    def prepare_coco_dataloaders(
        self,
        # dataloader_config,
        dataset_root,
        vocab_path="./vocabs/coco_vocab.pkl",
        num_workers=6,
        tsne=False,
        client=-1,
    ):
        """Prepare MS-COCO Caption train / val / test dataloaders
        Args:
            dataloader_config (dict): configuration file which should contain "batch_size"
            dataset_root (str): root of your MS-COCO dataset (see README.md for detailed dataset hierarchy)
            vocab_path (str, optional): path for vocab pickle file (default: ./vocabs/coco_vocab.pkl).
            num_workers (int, optional): num_workers for the dataloaders (default: 6)
        Returns:
            dataloaders (dict): keys = ["train", "val", "te"], values are the corresponding dataloaders.
            vocab (Vocabulary object): vocab object
        """
        batch_size = self.batch_size
        # we just initialize by code
        tr_cutout_prob = 0.2
        tr_caption_drop_prob = 0.1
        eval_batch_size = 8
        vocab = load_vocab(vocab_path)
        train_ids, train_extra_ids, val_ids, te_ids, image_root, train_ann, val_ann = (
            _get_coco_file_paths(dataset_root)
        )
        dataloaders = {}
        # dataloaders['train'] = _get_coco_loader(
        #     image_root, train_ann, train_ids, vocab,
        #     num_workers=num_workers, batch_size=batch_size,
        #     train=True,
        #     extra_annotation_path=val_ann,
        #     extra_ids=train_extra_ids,
        #     cutout_prob=tr_cutout_prob,
        #     caption_drop_prob=tr_caption_drop_prob,
        # )
        if tsne:
            pass
        elif client > -1:
            dataloaders["train_client"] = _get_coco_loader(
                image_root,
                train_ann,
                train_ids,
                vocab,
                num_workers=num_workers,
                batch_size=batch_size,
                train=True,
                extra_annotation_path=val_ann,
                extra_ids=train_extra_ids,
                cutout_prob=tr_cutout_prob,
                caption_drop_prob=tr_caption_drop_prob,
                subset=False,
                client=client,
            )
        else:
            dataloaders["train_subset_50000"] = _get_coco_loader(
                image_root,
                train_ann,
                train_ids,
                vocab,
                num_workers=num_workers,
                batch_size=batch_size,
                train=True,
                extra_annotation_path=val_ann,
                extra_ids=train_extra_ids,
                cutout_prob=tr_cutout_prob,
                caption_drop_prob=tr_caption_drop_prob,
                subset=True,
            )
            dataloaders["train_subset_eval_50000"] = _get_coco_loader(
                image_root,
                train_ann,
                train_ids,
                vocab,
                num_workers=num_workers,
                batch_size=batch_size * 2,
                train=False,
                extra_annotation_path=val_ann,
                extra_ids=train_extra_ids,
                cutout_prob=tr_cutout_prob,
                caption_drop_prob=tr_caption_drop_prob,
                subset=True,
            )

        dataloaders["val"] = _get_coco_loader(
            image_root,
            val_ann,
            val_ids,
            vocab,
            num_workers=num_workers,
            batch_size=eval_batch_size,
            train=False,
        )

        dataloaders["test"] = _get_coco_loader(
            image_root,
            val_ann,
            te_ids,
            vocab,
            num_workers=num_workers,
            batch_size=eval_batch_size if not tsne else 200,
            train=False,
        )

        return dataloaders, vocab


class ParallelMatMulModule(nn.Module):
    def set_g_features(self, g_features):
        self._g_features = g_features
        self.g_features = None

    def forward(self, q_features, n_embeddings=1):
        if self.g_features is None:
            self.g_features = self._g_features.to(q_features.device)
        sims = q_features.mm(self.g_features)

        if n_embeddings > 1:
            sims = sims.view(
                int(len(q_features) / n_embeddings),
                n_embeddings,
                int(self.g_features.size()[-1] / n_embeddings),
                n_embeddings,
            )
            sims = sims.permute(0, 1, 3, 2)
            sims = torch.sum(torch.sum(sims, axis=1), axis=1)

        sims, pred_ranks = (-sims).sort()
        return sims, pred_ranks
