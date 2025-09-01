import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class GMCLoss():
    def __init__(self,device = None) -> None:
        self.supervised_criterion = nn.CrossEntropyLoss()
        self.device = device
        
    def __call__(self, prediction, target, batch_representations, temperature, batch_size,missing_mask,is_contrast:bool = False):
        return self.gmc_plus(prediction, target, batch_representations, temperature, batch_size,missing_mask,is_contrast)
    
    def super_gmc_loss(self, prediction, target, batch_representations, temperature, batch_size):
        joint_mod_loss_sum = 0
        for mod in range(len(batch_representations) - 1):
            # Negative pairs: everything that is not in the current joint-modality pair
            out_joint_mod = torch.cat(
                [batch_representations[-1], batch_representations[mod]], dim=0
            )
            # [2*B, 2*B]
            tmp = torch.mm(out_joint_mod, out_joint_mod.t().contiguous())
            sim_matrix_joint_mod = torch.exp(
                tmp / temperature
            )
            # Mask for remove diagonal that give trivial similarity, [2*B, 2*B]
            mask_joint_mod = (
                torch.ones_like(sim_matrix_joint_mod)
                - torch.eye(2 * batch_size, device=sim_matrix_joint_mod.device)
            ).bool()
            # Remove 2*B diagonals and reshape to [2*B, 2*B-1]
            sim_matrix_joint_mod = sim_matrix_joint_mod.masked_select(
                mask_joint_mod
            ).view(2 * batch_size, -1)

            # Positive pairs: cosine loss joint-modality
            pos_sim_joint_mod = torch.exp(
                torch.sum(
                    batch_representations[-1] * batch_representations[mod], dim=-1
                )
                / temperature
            )
            # [2*B]
            pos_sim_joint_mod = torch.cat([pos_sim_joint_mod, pos_sim_joint_mod], dim=0)
            loss_joint_mod = -torch.log(
                pos_sim_joint_mod / sim_matrix_joint_mod.sum(dim=-1)
            )
            joint_mod_loss_sum += loss_joint_mod

        supervised_loss = self.supervised_criterion(prediction, target)

        loss = torch.mean(joint_mod_loss_sum * 0.2 + supervised_loss * 0.8)
        # loss = torch.mean(supervised_loss)
        # tqdm_dict = {"loss": loss}
        # return loss, tqdm_dict
        return loss

    
    def super_gmc_loss_latest(self, prediction, target, batch_representations, temperature, batch_size,missing_mask):
        joint_mod_loss_sum = 0
        # Missing modalities: [[0,1],[1,1],[1,0],[1,1]]
        ''' Here using the binary representations
            1,0 = 0010 = 2
            0,1 = 0001 = 1
            1,1 = 0011 = 3
        '''
        loss_sum = 0
        num_valid_pairs = 0
        for i in range(batch_size):
            mask_i = missing_mask[i]
            for j in range(i+1,batch_size):
                mask_j =  missing_mask[j]
                for mod_idx in range(len(batch_representations)-1):
                    compare_bit = 1 << mod_idx
                    if (compare_bit & mask_i) and (compare_bit & mask_j):
                        rep_i = batch_representations[mod_idx][i]
                        rep_j = batch_representations[mod_idx][j]
                        
                        pos_sim = F.cosine_similarity(rep_i.unsqueeze(0), rep_j.unsqueeze(0), dim=-1)
                        neg_sim_sum = 0
                        for neg_mod_idx in range(len(batch_representations)-1):
                            for neg_i in range(batch_size):
                                if neg_i != i and neg_i != j and ((missing_mask[neg_i] >> neg_mod_idx) & 1):
                                    neg_rep_i = batch_representations[neg_mod_idx][neg_i]
                                    neg_sim = F.cosine_similarity(rep_i.unsqueeze(0), neg_rep_i.unsqueeze(0), dim=-1)
                                    neg_sim_sum += torch.exp(neg_sim / temperature)
                                
                        loss = -torch.log(torch.exp(pos_sim / temperature) / (neg_sim_sum + 1e-8))
                        loss_sum += loss
                        num_valid_pairs += 1
                        
        constrastive_mod_loss_sum = loss_sum / (num_valid_pairs + 1e-8)
        supervised_loss = self.supervised_criterion(prediction, target)

        loss = torch.mean(constrastive_mod_loss_sum * 0.2 + supervised_loss * 0.8)
        # loss = torch.mean(supervised_loss)
        # tqdm_dict = {"loss": loss}
        # return loss, tqdm_dict
        return loss

    def convert_missingidx2mask(self,missing_index,num_modalities):
        missing_mask = []
        for v in missing_index:
            tmp = [0] * num_modalities
            for i in range(num_modalities):
                tmp[i] = (v >> i) & 1
            missing_mask.append(tmp)
        return torch.as_tensor(missing_mask,device = self.device)
    
    def cl_gmc_loss(self, prediction, target, batch_representations, temperature, batch_size,missing_index):
        num_modalities = len(batch_representations)-1
        missing_mask = self.convert_missingidx2mask(missing_index,num_modalities)
        
        features_modality1 = batch_representations[0]
        features_modality2 = batch_representations[1]

        # 计算所有样本对之间的相似度
        sim_matrix = torch.matmul(features_modality1, features_modality2.T) / temperature

        # 创建正样本掩码（对角线为1，其他为0）
        positive_mask = torch.eye(batch_size)

        # 创建有效样本掩码（基于modality_mask）
        valid_mask = torch.matmul(missing_mask[:, 0].unsqueeze(1).to(dtype=torch.float32), missing_mask[:, 1].unsqueeze(0).to(dtype=torch.float32))
        valid_mask = valid_mask.to(self.device)
        # 对无效的相似度进行掩码（设为一个很大的负值）
        # sim_matrix = torch.where(valid_mask == 1, sim_matrix, torch.tensor(-1e9).to(self.device))
        sim_matrix = torch.where(valid_mask == 1, sim_matrix, float('-inf'))
        # 计算正样本的损失
        positive_sim = torch.diag(sim_matrix)
        positive_loss = -torch.log(torch.exp(positive_sim) / torch.exp(sim_matrix).sum(dim=1))

        # 只考虑有效的正样本
        valid_positive = (missing_mask[:, 0] * missing_mask[:, 1]).bool()
        positive_loss = positive_loss * valid_positive

        # 计算负样本的损失
        negative_mask = (1 - positive_mask).to(self.device)
        neg_sim = torch.exp(sim_matrix) * negative_mask
        negative_loss = torch.log((1 + neg_sim.sum(dim=1)))

        # 组合正样本和负样本的损失
        # contrastive_loss = (positive_loss + negative_loss) * valid_positive
        contrastive_loss = 0
        pair_num = 0
        for i in range(valid_positive.size(0)):
            if valid_positive[i]:
                pair_num += 1
                contrastive_loss += positive_loss[i] + negative_loss[i]
        contrastive_loss /= (pair_num * 2)
        # 计算监督损失
        supervised_loss = self.supervised_criterion(prediction, target)

        # 根据模态存在情况调整每个样本的损失权重
        sample_weights = missing_mask.sum(dim=1) / 2  # 如果两个模态都存在，权重为1；如果只有一个模态，权重为0.5

        # 组合损失
        total_loss = (0.2 * contrastive_loss + 0.8 * supervised_loss) * sample_weights
        final_loss = total_loss.mean()
        if final_loss is torch.nan:
            print('what happend')
        return final_loss

    def gmc_plus(self, prediction, target, batch_representations, temperature, batch_size,missing_mask,is_contrast:bool):
        # 计算监督损失
        supervised_loss = self.supervised_criterion(prediction, target)
        if not is_contrast:
            return supervised_loss
        joint_mod_loss_sum = 0
        num_mods = len(batch_representations)
        num_modalities = len(batch_representations)-1
        missing_mask = self.convert_missingidx2mask(missing_mask,num_modalities)
        for mod in range(num_mods - 1):
            # 根据缺失的模态掩码，过滤无效样本
            valid_mask_mod = missing_mask[:, mod].unsqueeze(1) * missing_mask[:, -1].unsqueeze(0)
            valid_mask_joint_mod = torch.cat([valid_mask_mod, valid_mask_mod], dim=0)
            valid_mask_joint_mod = torch.cat([valid_mask_joint_mod, valid_mask_joint_mod], dim=1)
            # 提取当前模态和联合模态的表示
            out_joint_mod = torch.cat(
                [batch_representations[-1], batch_representations[mod]], dim=0
            )

            # 计算相似度矩阵 [2*B, 2*B]
            tmp = torch.mm(out_joint_mod, out_joint_mod.t().contiguous())
            sim_matrix_joint_mod = torch.exp(tmp / temperature)

            # 构建掩码，排除无效的相似度
            mask_joint_mod = (
                torch.ones_like(sim_matrix_joint_mod)
                - torch.eye(2 * batch_size, device=sim_matrix_joint_mod.device)
            ).bool()

            # 使用valid_mask过滤无效的相似度
            # valid_mask_joint_mod = torch.cat([valid_mask_mod, valid_mask_mod], dim=0)
            sim_matrix_joint_mod = torch.where(valid_mask_joint_mod == 1, sim_matrix_joint_mod, torch.tensor(0.0).to(sim_matrix_joint_mod.device))

            # 去除对角线，并将矩阵reshape为 [2*B, 2*B-1]
            sim_matrix_joint_mod = sim_matrix_joint_mod.masked_select(mask_joint_mod).view(2 * batch_size, -1)

            # 正样本相似度：基于联合模态和当前模态的余弦相似度
            pos_sim_joint_mod = torch.exp(
                torch.sum(
                    batch_representations[-1] * batch_representations[mod], dim=-1
                )
                / temperature
            )

            # 复制正样本相似度，扩展为 [2*B]
            pos_sim_joint_mod = torch.cat([pos_sim_joint_mod, pos_sim_joint_mod], dim=0)

            # 计算损失
            loss_joint_mod = -torch.log(
                pos_sim_joint_mod / (sim_matrix_joint_mod.sum(dim=-1) + 1e-8)  # 防止除以零
            )

            # 累积当前模态的损失
            joint_mod_loss_sum += loss_joint_mod

        # 组合损失：80%监督损失 + 20%模态间损失
        loss = max(torch.mean(joint_mod_loss_sum),0) * 0.1 + supervised_loss * 0.9

        return loss

class GMCLossWithMissingModalities(torch.nn.Module):
    def __init__(self, temperature=0.07):
        super(GMCLossWithMissingModalities, self).__init__()
        self.temperature = temperature
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
    
    def generate_missing_mask(self,missing_indexs):
        valid_masks = np.ones(shape=(len(missing_indexs),len(missing_indexs)))
        for i,missing_index in enumerate(missing_indexs):
            if missing_index == 1:
                valid_masks[i] = 0
            elif missing_index == 2:
                missing_indexs[:,i] = 0
        return missing_indexs
    
    def forward(self, sim, missing_indexs):
        """
        img_emb: 图像嵌入 (n_image, embed_dim)
        cap_emb: 文本嵌入 (n_caption, embed_dim)
        valid_mask_mod: 随机模态缺失的掩码 (n_image, n_caption)，1 表示该模态有效，0 表示该模态缺失
        """
        valid_mask_mod = self.generate_missing_mask(missing_indexs)
        # 获取每对图像和文本的相似性 (n_image, n_caption)
        sim_matrix = self.cosine_similarity(sim) / self.temperature
        
        # 对角线的正样本对 (图像与其对应描述的相似度)
        positive_pairs = torch.diag(sim_matrix)
        
        # 创建全 1 的掩码，用于标记有效的样本对
        valid_mask = valid_mask_mod.unsqueeze(0) * valid_mask_mod.unsqueeze(1)  # (n_image, n_caption) 的有效掩码
        
        # 将无效模态的位置设置为一个非常小的值，使得它们不会对损失计算产生影响
        sim_matrix = sim_matrix.masked_fill(~valid_mask.bool(), float('-inf'))
        
        # 使用 InfoNCE Loss 进行对比损失计算
        labels = torch.arange(sim.size(0)).long().to(sim.device)  # 正样本索引
        loss_img_to_text = F.cross_entropy(sim_matrix, labels)  # 图像到文本的对比损失
        loss_text_to_img = F.cross_entropy(sim_matrix.t(), labels)  # 文本到图像的对比损失
        
        # 返回损失的平均值
        return (loss_img_to_text + loss_text_to_img) / 2
