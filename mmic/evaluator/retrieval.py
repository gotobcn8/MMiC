"""Evaluation"""
from __future__ import print_function
import logging
import time
import os
import torch
import numpy as np
from utils.path import create
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        """String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)

class RetrievalEvaluator():
    def __init__(self,logger,logkey,save_path = None, device = None) -> None:
        self.logger = logger
        self.logkey = logkey
        self.save_path = save_path
        self.device = device        
        
    def encode_data(self,model, data_loader):
        """Encode all images and captions loadable by `data_loader`
        """
        # np array to keep all the embeddings
        img_embs = None
        cap_embs = None

        max_n_word = 0
        max_lengths = []
        for i, (images, captions, lengths, ids) in enumerate(data_loader):
            max_n_word = max(max_n_word, max(lengths))
            max_lengths.append((max(lengths),lengths[0]))
            # if i == len(data_loader) - 1:
            #     print(max_n_word)
            # if max(lengths) > max_n_word:
            #     print('There is a big error that lengths max is greater thatn max_n_word')

        end = time.time()
        max_lengths2 = []
        model.to(self.device)
        for i, (images, captions, lengths, ids) in enumerate(data_loader):
            # make sure val logger is used
            max_lengths2.append((max(lengths),lengths[0]))
            if max(lengths) > max_n_word:
                print(max_n_word)
            # compute the embeddings
            images,captions,lengths = images.to(self.device),captions.to(self.device),lengths.to(self.device)
            img_emb, cap_emb, cap_len = model.forward_emb(images, captions, lengths)
            if img_embs is None:
                img_embs = np.zeros((len(data_loader.dataset), img_emb.size(1), img_emb.size(2)))
                cap_embs = np.zeros((len(data_loader.dataset), max_n_word, cap_emb.size(2)))
                cap_lens = [0] * len(data_loader.dataset)
            # cache embeddings
            img_embs[list(ids)] = img_emb.data.cpu().numpy().copy()
            cap_embs[list(ids), :max(lengths), :] = cap_emb.data.cpu().numpy().copy()

            for j, nid in enumerate(ids):
                cap_lens[nid] = cap_len[j]

            del images, captions
        self.logger.info("calculate feature time: {}".format(time.time()-end))

        return img_embs, cap_embs, cap_lens


    # def eval_ensemble(self,results_paths, fold5=False):
    #     all_sims = []
    #     all_npts = []
    #     for sim_path in results_paths:
    #         results = np.load(sim_path, allow_pickle=True).tolist()
    #         npts = results['npts']
    #         sims = results['sims']
    #         all_npts.append(npts)
    #         all_sims.append(sims)
    #     all_npts = np.array(all_npts)
    #     all_sims = np.array(all_sims)
    #     assert np.all(all_npts == all_npts[0])
    #     npts = int(all_npts[0])
    #     sims = all_sims.mean(axis=0)

    #     if not fold5:
    #         r, rt = i2t(npts, sims, return_ranks=True)
    #         ri, rti = t2i(npts, sims, return_ranks=True)
    #         ar = (r[0] + r[1] + r[2]) / 3
    #         ari = (ri[0] + ri[1] + ri[2]) / 3
    #         rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
    #         logger.info("rsum: %.1f" % rsum)
    #         logger.info("Average i2t Recall: %.1f" % ar)
    #         logger.info("Image to text: %.1f %.1f %.1f %.1f %.1f" % r)
    #         logger.info("Average t2i Recall: %.1f" % ari)
    #         logger.info("Text to image: %.1f %.1f %.1f %.1f %.1f" % ri)
    #     else:
    #         npts = npts // 5
    #         results = []
    #         all_sims = sims.copy()
    #         for i in range(5):
    #             sims = all_sims[i * npts: (i + 1) * npts, i * npts * 5: (i + 1) * npts * 5]
    #             r, rt0 = i2t(npts, sims, return_ranks=True)
    #             logger.info("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" % r)
    #             ri, rti0 = t2i(npts, sims, return_ranks=True)
    #             logger.info("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" % ri)

    #             if i == 0:
    #                 rt, rti = rt0, rti0
    #             ar = (r[0] + r[1] + r[2]) / 3
    #             ari = (ri[0] + ri[1] + ri[2]) / 3
    #             rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
    #             logger.info("rsum: %.1f ar: %.1f ari: %.1f" % (rsum, ar, ari))
    #             results += [list(r) + list(ri) + [ar, ari, rsum]]
    #         logger.info("-----------------------------------")
    #         logger.info("Mean metrics: ")
    #         mean_metrics = tuple(np.array(results).mean(axis=0).flatten())
    #         logger.info("rsum: %.1f" % (mean_metrics[12]))
    #         logger.info("Average i2t Recall: %.1f" % mean_metrics[10])
    #         logger.info("Image to text: %.1f %.1f %.1f %.1f %.1f" %
    #                     mean_metrics[:5])
    #         logger.info("Average t2i Recall: %.1f" % mean_metrics[11])
    #         logger.info("Text to image: %.1f %.1f %.1f %.1f %.1f" %
    #                     mean_metrics[5:10])


    def evalrank(self,model, data_loader, fold5=False):
        """
        Evaluate a trained model on either dev or test. If `fold5=True`, 5 fold
        cross-validation is done (only for MSCOCO). Otherwise, the full data is
        used for evaluation.
        """
        # load model and options
        self.logger.info('Computing results...')
        model.eval()
        with torch.no_grad():
            img_embs, cap_embs, cap_lens = self.encode_data(model, data_loader)
            self.logger.info('Images: %d, Captions: %d' % (img_embs.shape[0] // 5, cap_embs.shape[0]))

            if not fold5:
                # no cross-validation, full evaluation
                img_embs = np.array([img_embs[i] for i in range(0, len(img_embs), 5)])

                end = time.time()
                sims = self.shard_attn_scores(model, img_embs, cap_embs, cap_lens, shard_size=100)
                self.logger.info("calculate similarity time: {}".format(time.time()-end))

                npts = img_embs.shape[0]
                r, rt = self.i2t(npts, sims, return_ranks=True)
                ri, rti = self.t2i(npts, sims, return_ranks=True)
                ar = (r[0] + r[1] + r[2]) / 3
                ari = (ri[0] + ri[1] + ri[2]) / 3
                rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
                self.logger.info("rsum: %.3f" % rsum)
                self.logger.info("Average i2t Recall: %.3f" % ar)
                self.logger.info("Image to text: %.3f %.3f %.3f %.3f %.3f" % r)
                self.logger.info("Average t2i Recall: %.3f" % ari)
                self.logger.info("Text to image: %.3f %.3f %.3f %.3f %.3f" % ri)
                
                if self.save_path is not None:
                    create.makedirs(self.save_path)
                    save_path = os.path.join(self.save_path,"results_{}.npy".format(self.logkey))
                    np.save(save_path, {'npts': npts, 'sims': sims})
                    self.logger.info('Save the similarity into {}'.format(save_path))
                    # self.logger.info('r1', r[0], step=step)
                    # self.logger.info('r5', r[1], step=step)
                    # self.logger.info('r10', r[2], step=step)
                    # self.logger.info('medr', r[3], step=step)
                    # self.logger.info('meanr', r[4], step=step)
                    # self.logger.info('r1i', ri[0], step=step)
                    # self.logger.info('r5i', ri[1], step=step)
                    # self.logger.info('r10i', ri[2], step=step)
                    # self.logger.info('medri', ri[3], step=step)
                    # self.logger.info('meanr', ri[4], step=step)
                    # self.logger.info('rsum', rsum, step=step)
                self.logger.info('r1:%.3f'%r[0])
                self.logger.info('r5:%.3f'%r[1])
                self.logger.info('r10:%.3f'%r[2])
                self.logger.info('medr:%.3f'%r[3])
                self.logger.info('meanr:%.3f'%r[4])
                self.logger.info('r1i:%.3f'%ri[0])
                self.logger.info('r5i:%.3f'%ri[1])
                self.logger.info('r10i:%.3f'%ri[2])
                self.logger.info('medri:%.3f'%ri[3])
                self.logger.info('meanri:%.3f'%ri[4])
                self.logger.info('rsum:%.3f'%rsum)
                return {
                    'rsum':rsum,
                    'r1':r[0],
                    'r5':r[1],
                    'meanr':r[3],
                    'r1i':ri[0],
                    'r5i':ri[1],
                    'meanr':ri[4]
                }
                
    def shard_attn_scores(self,model, img_embs, cap_embs, cap_lens, shard_size=100):
        n_im_shard = (len(img_embs) - 1) // shard_size + 1
        n_cap_shard = (len(cap_embs) - 1) // shard_size + 1

        sims = np.zeros((len(img_embs), len(cap_embs)))
        for i in range(n_im_shard):
            im_start, im_end = shard_size * i, min(shard_size * (i + 1), len(img_embs))
            for j in range(n_cap_shard):
                ca_start, ca_end = shard_size * j, min(shard_size * (j + 1), len(cap_embs))

                im = torch.from_numpy(img_embs[im_start:im_end]).float().cuda()
                ca = torch.from_numpy(cap_embs[ca_start:ca_end]).float().cuda()
                l = cap_lens[ca_start:ca_end]
                sim = model.forward_sim(im, ca, l)

                sims[im_start:im_end, ca_start:ca_end] = sim.data.cpu().numpy()
        return sims


    def i2t(self, npts, sims, return_ranks=False):
        """
        Images->Text (Image Annotation)
        Images: (N, n_region, d) matrix of images
        Captions: (5N, max_n_word, d) matrix of captions
        CapLens: (5N) array of caption lengths
        sims: (N, 5N) matrix of similarity im-cap
        """
        ranks = np.zeros(npts)
        top1 = np.zeros(npts)
        sim_max_size = sims[0].size
        for index in range(npts):
            inds = np.argsort(sims[index])[::-1]
            # Score
            rank = 1e20
            for i in range(5 * index, min(sim_max_size,5 * index + 5), 1):
                wherevalue = np.where(inds == i)
                tmp = np.where(inds == i)[0][0]
                rank = min(rank,tmp)
            ranks[index] = rank
            top1[index] = inds[0]

        # Compute metrics
        r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
        r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
        r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
        medr = np.floor(np.median(ranks)) + 1
        meanr = ranks.mean() + 1

        if return_ranks:
            return (r1, r5, r10, medr, meanr), (ranks, top1)
        else:
            return (r1, r5, r10, medr, meanr)


    def t2i(self, npts, sims, return_ranks=False):
        """
        Text->Images (Image Search)
        Images: (N, n_region, d) matrix of images
        Captions: (5N, max_n_word, d) matrix of captions
        CapLens: (5N) array of caption lengths
        sims: (N, 5N) matrix of similarity im-cap
        """
        ranks = np.zeros(5 * npts)
        top1 = np.zeros(5 * npts)
        
        # --> (5N(caption), N(image))
        sims = sims.T
        sims_max_size = sims.shape[0]
        for index in range(npts):
            for i in range(5):
                idx = min(5 * index + i,sims_max_size-1)
                inds = np.argsort(sims[idx])[::-1]
                ranks[5 * index + i] = np.where(inds == index)[0][0]
                top1[5 * index + i] = inds[0]

        # Compute metrics
        r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
        r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
        r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
        medr = np.floor(np.median(ranks)) + 1
        meanr = ranks.mean() + 1
        if return_ranks:
            return (r1, r5, r10, medr, meanr), (ranks, top1)
        else:
            return (r1, r5, r10, medr, meanr)
