import os
import time
import torch
import random
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import time

from utils_led.config import Config
from utils_led.utils import print_log

from torch.utils.data import DataLoader

from models.model_led_initializer import LEDInitializer as InitializationModel
from models.model_diffusion import TransformerDenoisingModel as CoreDenoisingModel
from data.dataloader_ngsim import ngsimDataset

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from torch.utils.tensorboard import SummaryWriter
from data.dataloader_ngsim import maskedMSE_diffusion

import data.loader2 as lo

NUM_Tau = 5


class Trainer:
    def __init__(self, config, local_rank):
        self.world_size = torch.cuda.device_count()
        self.device = self.setup_distributed(local_rank, self.world_size)
        self.cfg = Config(config.cfg, config.info)
        self.pretrain = config.pretrain

        # ------------------------- prepare train/test data loader -------------------------
        train_dset = ngsimDataset('data/files/TrainSet.mat')
        test_dset = ngsimDataset('data/files/TestSet.mat')
        self.train_dset = train_dset
        self.train_sampler = torch.utils.data.distributed.DistributedSampler(train_dset)
        self.train_loader = DataLoader(train_dset, batch_size=self.cfg.train_batch_size, sampler=self.train_sampler,
                                       num_workers=0, collate_fn=train_dset.collate_fn)
        self.test_sampler = torch.utils.data.distributed.DistributedSampler(test_dset)
        self.test_loader = DataLoader(train_dset, batch_size=self.cfg.test_batch_size, sampler=self.test_sampler,
                                      num_workers=0, collate_fn=test_dset.collate_fn)
        self.t2 = lo.ngsimDataset('/home/moresweet/gitCloneZone/HLTP/data/dataset_t_v_t/TestSet.mat')

        # data normalization parameters
        self.traj_mean = torch.FloatTensor(self.cfg.traj_mean).to(self.device).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        self.traj_scale = self.cfg.traj_scale

        # ------------------------- define diffusion parameters -------------------------
        self.n_steps = self.cfg.diffusion.steps  # define total diffusion steps

        # make beta schedule and calculate the parameters used in denoising process.
        self.betas = self.make_beta_schedule(
            schedule=self.cfg.diffusion.beta_schedule, n_timesteps=self.n_steps,
            start=self.cfg.diffusion.beta_start, end=self.cfg.diffusion.beta_end).to(self.device)

        self.alphas = 1 - self.betas
        self.alphas_prod = torch.cumprod(self.alphas, 0)
        self.alphas_bar_sqrt = torch.sqrt(self.alphas_prod)
        self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - self.alphas_prod)

        # ------------------------- define models -------------------------
        self.model = CoreDenoisingModel().to(self.device)
        self.model = DDP(self.model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        # load pretrained models
        if self.pretrain is False:
            model_cp = torch.load(self.cfg.pretrained_core_denoising_model, map_location='cpu')
            self.model.load_state_dict(model_cp['model_dict'])
            self.model_initializer = InitializationModel(t_h=16, d_h=2, t_f=24, d_f=2, k_pred=24).to(self.device)
            self.model_initializer = DDP(self.model_initializer, device_ids=[local_rank], output_device=local_rank,
                                         find_unused_parameters=True)
            self.opt = torch.optim.AdamW(self.model_initializer.parameters(), lr=self.cfg.learning_rate)
        if self.pretrain is True:
            # self.opt = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.learning_rate)
            self.opt = torch.optim.AdamW(self.model.parameters())
        # self.scheduler_model = torch.optim.lr_scheduler.StepLR(self.opt, step_size=self.cfg.decay_step,
        #                                                        gamma=self.cfg.decay_gamma)

        # ------------------------- prepare logs -------------------------
        self.log = open(os.path.join(self.cfg.log_dir, 'log.txt'), 'a+')
        self.print_model_param(self.model, name='Core Denoising Model')
        if self.pretrain is False:
            self.print_model_param(self.model_initializer, name='Initialization Model')

        # temporal reweight in the loss, it is not necessary.
        self.temporal_reweight = torch.FloatTensor([25 - i for i in range(1, 25)]).to(self.device).unsqueeze(
            0).unsqueeze(0) / 10
        # self.temporal_reweight[0][0][0] = 100000
        self.writer = SummaryWriter(log_dir=self.cfg.tensorboard_log_dir)

        # 注册钩子函数来查看线性层权重的梯度
        # def print_grad(grad):
        #     print("Gradient:", grad)
        # self.model.module.enc_lstm.weight_hh_l0.register_hook(print_grad)

    def setup_distributed(self, local_rank, world_size):
        dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
        return device

    def cleanup_distributed(self):
        dist.destroy_process_group()

    def print_model_param(self, model: nn.Module, name: str = 'Model') -> None:
        '''
        Count the trainable/total parameters in `model`.
        '''
        total_num = sum(p.numel() for p in model.parameters())
        trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print_log("[{}] Trainable/Total: {}/{}".format(name, trainable_num, total_num), self.log)
        return None

    def make_beta_schedule(self, schedule: str = 'linear',
                           n_timesteps: int = 1000,
                           start: float = 1e-5, end: float = 1e-2) -> torch.Tensor:
        '''
        Make beta schedule.

        Parameters
        ----
        schedule: str, in ['linear', 'quad', 'sigmoid'],
        n_timesteps: int, diffusion steps,
        start: float, beta start, `start<end`,
        end: float, beta end,

        Returns
        ----
        betas: Tensor with the shape of (n_timesteps)

        '''
        if schedule == 'linear':
            betas = torch.linspace(start, end, n_timesteps)
        elif schedule == "quad":
            betas = torch.linspace(start ** 0.5, end ** 0.5, n_timesteps) ** 2
        elif schedule == "sigmoid":
            betas = torch.linspace(-6, 6, n_timesteps)
            betas = torch.sigmoid(betas) * (end - start) + start
        return betas

    def extract(self, input, t, x):
        shape = x.shape
        out = torch.gather(input, 0, t.to(input.device))
        reshape = [t.shape[0]] + [1] * (len(shape) - 1)
        return out.reshape(*reshape)

    def noise_estimation_loss(self, x, y_0, mask):
        batch_size = x.shape[0]
        # Select a random step for each example
        t = torch.randint(0, self.n_steps, size=(batch_size // 2 + 1,)).to(x.device)
        t = torch.cat([t, self.n_steps - t - 1], dim=0)[:batch_size]
        # x0 multiplier
        a = self.extract(self.alphas_bar_sqrt, t, y_0)
        beta = self.extract(self.betas, t, y_0)
        # eps multiplier
        am1 = self.extract(self.one_minus_alphas_bar_sqrt, t, y_0)
        e = torch.randn_like(y_0)
        # model input
        y = y_0 * a + e * am1
        output = self.model(y, beta, x, mask)
        # batch_size, 20, 2
        return (e - output).square().mean()

    def p_sample(self, x, mask, cur_y, nbrs, t):
        # past,mask,noise_matrix,nbrs,t
        if t == 0:
            z = torch.zeros_like(cur_y).to(x.device)
        else:
            z = torch.randn_like(cur_y).to(x.device)
        t = torch.tensor([t]).cuda()
        # Factor to the model output
        eps_factor = (
                (1 - self.extract(self.alphas, t, cur_y)) / self.extract(self.one_minus_alphas_bar_sqrt, t, cur_y))
        # Model output
        beta = self.extract(self.betas, t.repeat(x.shape[1]), cur_y)
        eps_theta = self.model(cur_y, beta, x, nbrs, mask)
        # eps_theta (B,24,2)
        mean = (1 / self.extract(self.alphas, t, cur_y).sqrt()) * (cur_y - (eps_factor * eps_theta))
        # Generate z
        z = torch.randn_like(cur_y).to(x.device)
        # Fixed sigma
        sigma_t = self.extract(self.betas, t, cur_y).sqrt()
        sample = mean + sigma_t * z
        return (sample)

    def p_sample_accelerate(self, x, mask, cur_y, nbrs, t):
        if t == 0:
            z = torch.zeros_like(cur_y).to(x.device)
        else:
            z = torch.randn_like(cur_y).to(x.device)
        t = torch.tensor([t]).cuda()
        # Factor to the model output
        eps_factor = (
                (1 - self.extract(self.alphas, t, cur_y)) / self.extract(self.one_minus_alphas_bar_sqrt, t, cur_y))
        # Model output
        beta = self.extract(self.betas, t.repeat(x.shape[1]), cur_y)
        eps_theta = self.model.module.generate_accelerate(cur_y, beta, x, nbrs, mask)
        mean = (1 / self.extract(self.alphas, t, cur_y).sqrt()) * (cur_y - (eps_factor * eps_theta))
        # Generate z
        z = torch.randn_like(cur_y).to(x.device)
        # Fixed sigma
        sigma_t = self.extract(self.betas, t, cur_y).sqrt()
        sample = mean + sigma_t * z * 0.00001
        return (sample)

    def p_sample_loop(self, x, nbrs, mask, shape):
        # past_traj, nbrs_traj, traj_mask
        # self.model.eval()
        prediction_total = torch.Tensor().cuda()
        for _ in range(24):
            cur_y = torch.randn(shape).to(x.device)
            # torch.save(cur_y, "./cur_y_5.pt")
            for i in reversed(range(self.n_steps)):
                cur_y = self.p_sample(x, mask, cur_y, nbrs, i)
                # torch.save(cur_y, f"./cur_y_{i}.pt")
            prediction_total = torch.cat((prediction_total, cur_y.unsqueeze(1)), dim=1)
        return prediction_total

    def p_sample_loop_mean(self, x, nbrs, mask, loc):
        prediction_total = torch.Tensor().cuda()
        for loc_i in range(1):
            cur_y = loc
            for i in reversed(range(NUM_Tau)):
                cur_y = self.p_sample(x, mask, cur_y, nbrs, i)
            prediction_total = torch.cat((prediction_total, cur_y.unsqueeze(1)), dim=1)
            # prediction_total = torch.cat((prediction_total, cur_y), dim=1)
        return prediction_total

    def p_sample_loop_accelerate(self, x, nbrs, mask, loc):
        '''
        Batch operation to accelerate the denoising process.

        x: [11, 10, 6]
        mask: [11, 11]
        cur_y: [11, 10, 20, 2]
        '''
        prediction_total = torch.Tensor().cuda()
        cur_y = loc[:, :12]
        for i in reversed(range(NUM_Tau)):
            cur_y = self.p_sample_accelerate(x, mask, cur_y, nbrs, i)
        cur_y_ = loc[:, 12:]
        for i in reversed(range(NUM_Tau)):
            cur_y_ = self.p_sample_accelerate(x, mask, cur_y_, nbrs, i)
        # shape: B=b*n, K=10, T, 2
        prediction_total = torch.cat((cur_y_, cur_y), dim=1)
        return prediction_total

    def fit(self):
        # Training loop
        for epoch in range(0, self.cfg.num_epochs):
            loss_total, loss_distance, loss_uncertainty = self._train_single_epoch(epoch)
            print_log('[{}] Epoch: {}\t\tLoss: {:.6f}\tLoss Dist.: {:.6f}\tLoss Uncertainty: {:.6f}'.format(
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                epoch, loss_total, loss_distance, loss_uncertainty), self.log)

            if (epoch + 1) % self.cfg.test_interval == 0:
                performance, samples = self._test_single_epoch()
                for time_i in range(4):
                    print_log('--ADE({}s): {:.4f}\t--FDE({}s): {:.4f}'.format(
                        time_i + 1, performance['ADE'][time_i] / samples,
                        time_i + 1, performance['FDE'][time_i] / samples), self.log)
                    self.writer.add_scalar('ADE({}s)'.format(time_i + 1),
                                           performance['ADE'][time_i] / samples,
                                           epoch)
                    self.writer.add_scalar('FDE({}s)'.format(time_i + 1),
                                           performance['FDE'][time_i] / samples,
                                           epoch)
                cp_path = self.cfg.model_path % (epoch + 1)
                cp_de_path = self.cfg.pre_model_path % (epoch + 1)
                if self.pretrain is False:
                    model_cp = {'model_initializer_dict': self.model_initializer.state_dict()}
                    torch.save(model_cp, cp_path)
                else:
                    model_cp_de = {'model_dict': self.model.state_dict()}
                    torch.save(model_cp_de, cp_de_path)
            # self.scheduler_model.step()

    def _train_single_epoch(self, epoch):
        total_start_time = time.time()
        # 设置训练模式
        self.model.train()
        if self.pretrain is False:
            self.model_initializer.train()
        loss_total, loss_dt, loss_dc, count = 0, 0, 0, 0
        num_batches = len(self.train_loader)
        with tqdm(total=num_batches, desc="Training Progress", unit="batch") as pbar:
            for data in self.train_loader:
                batch_start_time = time.time()
                past_traj, nbrs_traj, mask, lat_enc, lon_enc, fut_traj, op_mask, _ = data
                past_traj = past_traj.to(self.device)
                nbrs_traj = nbrs_traj.to(self.device)
                fut_traj = fut_traj.to(self.device)
                traj_mask = mask.bool().to(self.device)
                op_mask = op_mask.to(self.device)
                loss = None
                if self.pretrain is False:
                    sample_prediction, mean_estimation, variance_estimation = self.model_initializer(past_traj,
                                                                                                     nbrs_traj,
                                                                                                     traj_mask)
                    sample_prediction = torch.exp(variance_estimation / 2)[
                                            ..., None, None] * sample_prediction / sample_prediction.std(dim=1).mean(
                        dim=(1, 2))[:, None, None, None]
                    loc = sample_prediction + mean_estimation[:, None]

                    generated_y = self.p_sample_loop_accelerate(past_traj, nbrs_traj, traj_mask, loc)
                    loss_dist = ((generated_y - fut_traj.permute(1, 0, 2).unsqueeze(dim=1)).norm(p=2, dim=-1)
                                 *
                                 self.temporal_reweight
                                 ).mean(dim=-1).min(dim=1)[0].mean()
                    loss_uncertainty = (torch.exp(-variance_estimation)
                                        *
                                        (generated_y - fut_traj.permute(1, 0, 2).unsqueeze(dim=1)).norm(p=2,
                                                                                                        dim=-1).mean(
                                            dim=(1, 2))
                                        +
                                        variance_estimation
                                        ).mean()

                    loss = loss_dist * 50 + loss_uncertainty
                    self.writer.add_scalar('loss/step(batch)', loss, epoch * num_batches + count)
                    self.writer.add_scalar('loss_dist/step(batch)', loss_dist, epoch * num_batches + count)
                    self.writer.add_scalar('loss_uncertainty/step(batch)', loss_uncertainty,
                                           epoch * num_batches + count)
                else:
                    generated_y = self.p_sample_loop(past_traj, nbrs_traj, traj_mask,
                                                     [fut_traj.shape[1], fut_traj.shape[0], fut_traj.shape[2]])
                    # generated_y = self.model([fut_traj.shape[1], fut_traj.shape[0], fut_traj.shape[2]], [], past_traj, nbrs_traj, traj_mask)
                    loss_dist = maskedMSE_diffusion(generated_y, fut_traj.permute(1,0,2), op_mask.permute(1,0,2))
                    # generated_y = generated_y * op_mask.permute(1, 0, 2).unsqueeze(1)
                    # loss_dist = ((generated_y - fut_traj.permute(1, 0, 2).unsqueeze(dim=1)).norm(p=2, dim=-1)
                    #              *
                    #              self.temporal_reweight
                    #              ).mean(dim=-1).min(dim=1)[0].mean()

                    # diversity_loss = -( (generated_y.unsqueeze(2) - generated_y.unsqueeze(1)).norm(p=2,
                    # dim=-1).mean(dim=(1, 2))).mean()
                    # gy_range = (generated_y.max(dim=1)[0] - generated_y.min(dim=1)[0]).mean()
                    # fut_range = (fut_traj.max(dim=1)[0] - fut_traj.min(dim=1)[0]).mean()
                    # loss_range = fut_range / gy_range * ((generated_y - fut_traj.permute(1, 0, 2).unsqueeze(dim=1)* self.temporal_reweight).norm(p=2, dim=-1)).sum(dim=-1).max()
                    #                mean: point  min: trajectory mean: Batch
                    # diff_x = (torch.abs(generated_y[:, :, :, 1] - fut_traj.permute(1, 0, 2)[:, :, 1].unsqueeze(1))).sum(dim=2).min(dim=1)[0].mean()
                    # diff_y = (torch.abs(generated_y[:, :, :, 0] - fut_traj.permute(1, 0, 2)[:, :, 0].unsqueeze(1))).sum(dim=2).min(dim=1)[0].mean()
                    # loss_range = diff_x + diff_y
                    loss = loss_dist
                    self.writer.add_scalar('loss/step(batch)', loss, epoch * num_batches + count)
                loss_total += loss.item()
                self.opt.zero_grad()
                loss.backward()

                # def print_graph(grad_fn, indent=0):
                #     if grad_fn is None:
                #         return
                #     print(" " * indent, grad_fn)
                #     for next_fn in grad_fn.next_functions:
                #         print_graph(next_fn[0], indent + 4)
                #
                # # 打印计算图
                # print_graph(loss.grad_fn)
                if self.pretrain is False:
                    torch.nn.utils.clip_grad_norm_(self.model_initializer.parameters(), 1.)
                self.opt.step()
                count += 1
                pbar.update(1)  # 更新进度条

                batch_end_time = time.time()  # 记录每个批次的结束时间
                batch_duration = batch_end_time - batch_start_time  # 计算每个批次的持续时间
                total_duration = batch_end_time - total_start_time  # 计算总的持续时间

                if self.pretrain is False:
                    loss_dt += loss_dist.item() * 50
                    loss_dc += loss_uncertainty.item()
                    pbar.set_postfix({
                        "Batch time": f"{batch_duration:.2f}s",
                        "Total time": f"{total_duration:.2f}s",
                        "Loss": f"{loss_total:.2f}",
                        "Loss_dt": f"{loss_dt:.2f}",
                        "Loss_dc": f"{loss_dc:.2f}"
                    })  # 在进度条后显示批次和总的用时
                else:
                    pbar.set_postfix({
                        "Batch time": f"{batch_duration:.2f}s",
                        "Total time": f"{total_duration:.2f}s",
                        "add loss": f"{loss_dist:.2f}",
                        "Loss": f"{loss_total:.2f}"
                    })  # 在进度条后显示批次和总的用时
                if self.cfg.debug and count == 2:
                    break
        self.writer.add_scalar('Total Loss/Epoch', loss_total, epoch)
        return loss_total / count, loss_dt / count, loss_dc / count

    def _test_single_epoch(self):
        performance = {'FDE': [0, 0, 0, 0],
                       'ADE': [0, 0, 0, 0]}
        samples = 0

        def prepare_seed(rand_seed):
            np.random.seed(rand_seed)
            random.seed(rand_seed)
            torch.manual_seed(rand_seed)
            torch.cuda.manual_seed_all(rand_seed)

        prepare_seed(0)
        count = 0
        with torch.no_grad():
            for data in self.test_loader:
                past_traj, nbrs_traj, mask, lat_enc, lon_enc, fut_traj, op_mask, _ = data
                past_traj = past_traj.to(self.device)
                nbrs_traj = nbrs_traj.to(self.device)
                fut_traj = fut_traj.to(self.device)
                traj_mask = mask.bool().to(self.device)
                pred_traj = None
                if self.pretrain is False:
                    sample_prediction, mean_estimation, variance_estimation = self.model_initializer(past_traj,
                                                                                                     nbrs_traj,
                                                                                                     traj_mask)
                    sample_prediction = torch.exp(variance_estimation / 2)[
                                            ..., None, None] * sample_prediction / sample_prediction.std(dim=1).mean(
                        dim=(1, 2))[:, None, None, None]
                    loc = sample_prediction + mean_estimation[:, None]

                    pred_traj = self.p_sample_loop_accelerate(past_traj, nbrs_traj, traj_mask, loc)
                else:
                    pred_traj = self.p_sample_loop(past_traj, nbrs_traj, traj_mask,
                                                   [fut_traj.shape[1], fut_traj.shape[0], fut_traj.shape[2]])

                fut_traj = fut_traj.permute(1, 0, 2).unsqueeze(1).repeat(1, 24, 1, 1)
                # b*n, K, T, 2
                distances = torch.norm(fut_traj - pred_traj, dim=-1)  # * self.traj_scale
                for time_i in range(1, 5):
                    ade = (distances[:, :, :5 * time_i]).mean(dim=-1).min(dim=-1)[0].sum()
                    fde = (distances[:, :, 5 * time_i - 1]).min(dim=-1)[0].sum()
                    performance['ADE'][time_i - 1] += ade.item()
                    performance['FDE'][time_i - 1] += fde.item()
                samples += distances.shape[0]
                count += 1
                if count == 100:
                    break

        return performance, samples

    def save_data(self):
        '''
        Save the visualization data.
        '''
        model_path = self.cfg.pretrained_core_denoising_model
        model_dict = torch.load(model_path, map_location=torch.device('cpu'))['model_initializer_dict']
        self.model_initializer.load_state_dict(model_dict)

        def prepare_seed(rand_seed):
            np.random.seed(rand_seed)
            random.seed(rand_seed)
            torch.manual_seed(rand_seed)
            torch.cuda.manual_seed_all(rand_seed)

        prepare_seed(0)
        root_path = './visualization/new/'

        with torch.no_grad():
            for data in self.test_loader:
                past_traj, nbrs_traj, mask, lat_enc, lon_enc, fut_traj, op_mask, _ = data
                past_traj = past_traj.to(self.device)
                nbrs_traj = nbrs_traj.to(self.device)
                fut_traj = fut_traj.to(self.device)
                traj_mask = mask.bool().to(self.device)

                sample_prediction, mean_estimation, variance_estimation = self.model_initializer(past_traj, nbrs_traj,
                                                                                                 traj_mask)
                torch.save(sample_prediction, root_path + 'p_var.pt')
                torch.save(mean_estimation, root_path + 'p_mean.pt')
                torch.save(variance_estimation, root_path + 'p_sigma.pt')

                sample_prediction = torch.exp(variance_estimation / 2)[
                                        ..., None, None] * sample_prediction / sample_prediction.std(dim=1).mean(
                    dim=(1, 2))[:, None, None, None]
                loc = sample_prediction + mean_estimation[:, None]

                pred_traj = self.p_sample_loop_accelerate(past_traj, nbrs_traj, traj_mask, loc)
                pred_mean = self.p_sample_loop_mean(past_traj, nbrs_traj, traj_mask, mean_estimation)

                # torch.save(data['pre_motion_3D'], root_path + 'past.pt')
                torch.save(data[0].permute(1, 0, 2), root_path + 'past.pt')
                torch.save(traj_mask, root_path + 'traj_mask.pt')
                # torch.save(data['fut_motion_3D'], root_path + 'future.pt')
                torch.save(data[5].permute(1, 0, 2), root_path + 'future.pt')
                torch.save(nbrs_traj.permute(1, 0, 2), root_path + 'nbrs.pt')
                torch.save(pred_traj, root_path + 'prediction.pt')
                torch.save(pred_mean, root_path + 'p_mean_denoise.pt')

                raise ValueError

    def save_data_new(self):
        '''
        Save the visualization data.
        '''
        model_path = self.cfg.pretrained_core_denoising_model
        model_dict = torch.load(model_path, map_location=torch.device('cpu'))['model_dict']
        self.model.load_state_dict(model_dict)

        def prepare_seed(rand_seed):
            np.random.seed(rand_seed)
            random.seed(rand_seed)
            torch.manual_seed(rand_seed)
            torch.cuda.manual_seed_all(rand_seed)

        prepare_seed(0)
        root_path = './visualization/new/'

        with torch.no_grad():
            for data in self.test_loader:
                past_traj, nbrs_traj, mask, lat_enc, lon_enc, fut_traj, op_mask, _ = data
                past_traj = past_traj.to(self.device)
                nbrs_traj = nbrs_traj.to(self.device)
                fut_traj = fut_traj.to(self.device)
                traj_mask = mask.bool().to(self.device)
                pred_traj = self.p_sample_loop(past_traj, nbrs_traj, traj_mask,
                                               [fut_traj.shape[1], fut_traj.shape[0], fut_traj.shape[2]])
                pred_mean = pred_traj.mean(dim=1)
                torch.save(data[0].permute(1, 0, 2), root_path + 'past.pt')
                torch.save(traj_mask, root_path + 'traj_mask.pt')
                torch.save(data[5].permute(1, 0, 2), root_path + 'future.pt')
                torch.save(nbrs_traj.permute(1, 0, 2), root_path + 'nbrs.pt')
                torch.save(pred_traj, root_path + 'prediction.pt')
                torch.save(pred_mean, root_path + 'p_mean_denoise.pt')

                raise ValueError

    def save_data_multi_path(self):
        '''
        Save the visualization data.
        '''
        model_path = self.cfg.pretrained_core_denoising_model
        model_dict = torch.load(model_path, map_location=torch.device('cpu'))['model_dict']
        self.model.load_state_dict(model_dict)

        def prepare_seed(rand_seed):
            np.random.seed(rand_seed)
            random.seed(rand_seed)
            torch.manual_seed(rand_seed)
            torch.cuda.manual_seed_all(rand_seed)

        prepare_seed(0)
        root_path = './visualization/new/'

        all_nbr_pred_trajs = []
        batch_nbrs_count = []
        batch_nbrs_start = []
        with torch.no_grad():
            for data in self.train_loader:
                ### 指定数据临时验证
                # data = torch.load('/home/moresweet/gitCloneZone/HLTP/test_batch_data', map_location='cpu')
                # past_traj, nbrs_traj, _, _, _, _, _, _, fut_traj, _, _, mask, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, nbrs_id_batch = data
                ### 指定数据临时验证end
                past_traj, nbrs_traj, mask, lat_enc, lon_enc, fut_traj, op_mask, nbrs_id_batch = data
                # 不用把nbrs_id_batch想得太复杂，其就是mask的具备具体id以及拍扁版本。
                past_traj = past_traj.to(self.device)
                nbrs_traj = nbrs_traj.to(self.device)
                fut_traj = fut_traj[0:24,: , :].to(self.device)
                torch.save(fut_traj.permute(1, 0, 2), "./draw_paradigm_fut.pt")
                traj_mask = mask.bool().to(self.device)
                pred_traj = self.p_sample_loop(past_traj, nbrs_traj, traj_mask,
                                               [fut_traj.shape[1], fut_traj.shape[0], fut_traj.shape[2]])
                nbrs_record_count = -1
                # 遍历每个batch中的所有邻车
                with tqdm(total=nbrs_id_batch.shape[0]) as pbar:
                    for ids in nbrs_id_batch:
                        temp_valid_count = 0
                        nbrs_ids = ids[np.where(ids != 0)]
                        # 遍历当前batch中的所有邻车
                        if(len(nbrs_ids) > 0):
                            for id in nbrs_ids:
                                nbrs_record_count += 1
                                temp_list = []
                                # print(id)
                                try:
                                    temp_hist, temp_fut, temp_neighbors, temp_lat_enc, temp_lon_enc, _ = self.train_dset.get_item_by_vId(id)
                                    temp_list.append(tuple([temp_hist, temp_fut, temp_neighbors, temp_lat_enc, temp_lon_enc, _]))
                                    nbr_past_traj, nbr_nbrs_traj, nbr_mask, nbr_lat_enc, nbr_lon_enc, nbr_fut_traj, nbr_op_mask, nbr_nbrs_id_batch = self.train_dset.collate_fn(temp_list)
                                    nbr_past_traj = nbr_past_traj.to(self.device)
                                    nbr_nbrs_traj = nbr_nbrs_traj.to(self.device)
                                    nbr_fut_traj = nbr_fut_traj.to(self.device)
                                    nbr_traj_mask = nbr_mask.bool().to(self.device)
                                    nbr_pred_traj = None
                                    try:
                                        nbr_pred_traj = self.p_sample_loop(nbr_past_traj, nbr_nbrs_traj, nbr_traj_mask,
                                                                   [nbr_fut_traj.shape[1], nbr_fut_traj.shape[0], nbr_fut_traj.shape[2]])
                                    except Exception as e:
                                        print(e)
                                        continue
                                    all_nbr_pred_trajs.append(nbr_pred_traj.cpu())
                                    batch_nbrs_start.append(nbrs_traj.permute(1, 0, 2)[nbrs_record_count][-1])
                                    temp_valid_count += 1
                                except Exception as e:
                                    print(e)
                                    continue
                        # else:
                        #     batch_nbrs_count.append(0)
                        batch_nbrs_count.append(temp_valid_count)
                        if np.array(batch_nbrs_count).sum() != len(all_nbr_pred_trajs):
                            print("dasd")
                            pass
                        pbar.update(1)

                pred_mean = pred_traj.mean(dim=1)
                torch.save(data[0].permute(1, 0, 2), root_path + 'past.pt')
                torch.save(traj_mask, root_path + 'traj_mask.pt')
                torch.save(data[5].permute(1, 0, 2), root_path + 'future.pt')
                torch.save(nbrs_traj.permute(1, 0, 2), root_path + 'nbrs.pt')
                torch.save(pred_traj, root_path + 'prediction.pt')
                torch.save(pred_mean, root_path + 'p_mean_denoise.pt')
                # 保存所有邻居车辆的预测轨迹
                torch.save(all_nbr_pred_trajs, root_path + 'all_nbr_predictions.pt')
                torch.save(batch_nbrs_count, root_path + 'batch_nbrs_count.pt')
                torch.save(batch_nbrs_start, root_path + 'batch_nbrs_start.pt')

                raise ValueError

    def save_data_multi_path_hltp(self):
        '''
        Save the visualization data.
        '''
        model_path = self.cfg.pretrained_core_denoising_model
        model_dict = torch.load(model_path, map_location=torch.device('cpu'))['model_dict']
        self.model.load_state_dict(model_dict)

        def prepare_seed(rand_seed):
            np.random.seed(rand_seed)
            random.seed(rand_seed)
            torch.manual_seed(rand_seed)
            torch.cuda.manual_seed_all(rand_seed)

        prepare_seed(0)
        root_path = './visualization/new/'

        all_nbr_pred_trajs = []
        batch_nbrs_count = []
        batch_nbrs_start = []
        with torch.no_grad():
            for data in self.test_loader:
                ### 指定数据临时验证
                data = torch.load('/home/moresweet/gitCloneZone/HLTP/test_batch_data', map_location='cpu')
                past_traj, nbrs_traj, _, _, _, _, _, _, fut_traj, _, _, mask, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, nbrs_id_batch = data
                ### 指定数据临时验证end
                # past_traj, nbrs_traj, mask, lat_enc, lon_enc, fut_traj, op_mask, nbrs_id_batch = data
                # 不用把nbrs_id_batch想得太复杂，其就是mask的具备具体id以及拍扁版本。
                past_traj = past_traj.to(self.device)
                nbrs_traj = nbrs_traj.to(self.device)
                fut_traj = fut_traj[0:24,: , :].to(self.device)
                torch.save(fut_traj.permute(1, 0, 2), "./draw_paradigm_fut.pt")
                traj_mask = mask.bool().to(self.device)
                pred_traj = self.p_sample_loop(past_traj, nbrs_traj, traj_mask,
                                               [fut_traj.shape[1], fut_traj.shape[0], fut_traj.shape[2]])
                nbrs_record_count = -1
                # 遍历每个batch中的所有邻车
                with tqdm(total=nbrs_id_batch.shape[0]) as pbar:
                    for ids in nbrs_id_batch:
                        temp_valid_count = 0
                        nbrs_ids = ids[np.where(ids != 0)]
                        # 遍历当前batch中的所有邻车
                        if(len(nbrs_ids) > 0):
                            for id in nbrs_ids:
                                nbrs_record_count += 1
                                temp_list = []
                                # print(id)
                                try:
                                    temp_hist, temp_fut, temp_neighbors, lane_stu, neighborslane_stu, cclass_stu, \
                                        neighborsclass_stu, va_stu, neighborsva_stu, hist, fut, neighbors, temp_lat_enc, temp_lon_enc, \
                                        lane, neighborslane, cclass, neighborsclass, va, neighborsva, edge_index, ve_matrix, \
                                        ac_matrix, man_matrix, view_grip, _ = self.t2.get_item_by_id(id)
                                    temp_list.append(tuple(
                                        [temp_hist, temp_fut, temp_neighbors, lane_stu, neighborslane_stu, cclass_stu, \
                                         neighborsclass_stu, va_stu, neighborsva_stu, hist, fut, neighbors, temp_lat_enc,
                                         temp_lon_enc, \
                                         lane, neighborslane, cclass, neighborsclass, va, neighborsva, edge_index,
                                         ve_matrix, \
                                         ac_matrix, man_matrix, view_grip, _]))
                                    nbr_past_traj, nbr_nbrs_traj, lane, nbrslane, cls, nbrscls, va, nbrsva, nbr_fut_traj, hist_batch, nbrs_batch, nbr_mask, \
                                        nbr_lat_enc, nbr_lon_enc, lane_batch, nbrslane_batch, class_batch, nbrsclass_batch, va_batch, \
                                        nbrsva_batch, fut_batch, nbr_op_mask, edge_index_batch, ve_matrix_batch, ac_matrix_batch, \
                                        man_matrix_batch, view_grip_batch, graph_matrix, nbr_nbrs_id_batch = self.t2.collate_fn(
                                        temp_list)
                                    # temp_hist, temp_fut, temp_neighbors, temp_lat_enc, temp_lon_enc, _ = self.t2.get_item_by_vId(id)
                                    # temp_list.append(tuple([temp_hist, temp_fut, temp_neighbors, temp_lat_enc, temp_lon_enc, _]))
                                    # nbr_past_traj, nbr_nbrs_traj, nbr_mask, nbr_lat_enc, nbr_lon_enc, nbr_fut_traj, nbr_op_mask, nbr_nbrs_id_batch = self.train_dset.collate_fn(temp_list)
                                    nbr_past_traj = nbr_past_traj.to(self.device)
                                    nbr_nbrs_traj = nbr_nbrs_traj.to(self.device)
                                    nbr_fut_traj = nbr_fut_traj[0:24,:,:].to(self.device)
                                    nbr_traj_mask = nbr_mask.bool().to(self.device)
                                    nbr_pred_traj = None
                                    try:
                                        nbr_pred_traj = self.p_sample_loop(nbr_past_traj, nbr_nbrs_traj, nbr_traj_mask,
                                                                   [nbr_fut_traj.shape[1], nbr_fut_traj.shape[0], nbr_fut_traj.shape[2]])
                                    except Exception as e:
                                        print(e)
                                        continue
                                    all_nbr_pred_trajs.append(nbr_pred_traj.cpu())
                                    batch_nbrs_start.append(nbrs_traj.permute(1, 0, 2)[nbrs_record_count][-1])
                                    temp_valid_count += 1
                                except Exception as e:
                                    print(e)
                                    continue
                        # else:
                        #     batch_nbrs_count.append(0)
                        batch_nbrs_count.append(temp_valid_count)
                        if np.array(batch_nbrs_count).sum() != len(all_nbr_pred_trajs):
                            print("dasd")
                            pass
                        pbar.update(1)

                pred_mean = pred_traj.mean(dim=1)
                torch.save(past_traj.permute(1, 0, 2), root_path + 'past.pt')
                torch.save(traj_mask, root_path + 'traj_mask.pt')
                torch.save(fut_traj.permute(1, 0, 2), root_path + 'future.pt')
                torch.save(nbrs_traj.permute(1, 0, 2), root_path + 'nbrs.pt')
                torch.save(pred_traj, root_path + 'prediction.pt')
                torch.save(pred_mean, root_path + 'p_mean_denoise.pt')
                # 保存所有邻居车辆的预测轨迹
                torch.save(all_nbr_pred_trajs, root_path + 'all_nbr_predictions.pt')
                torch.save(batch_nbrs_count, root_path + 'batch_nbrs_count.pt')
                torch.save(batch_nbrs_start, root_path + 'batch_nbrs_start.pt')

                raise ValueError

    def test_single_model(self):
        if self.pretrain is False:
            model_path = '/home/moresweet/gitCloneZone/modify_transformer/results/led_augment/new/models/model_0100.p'
            model_dict = torch.load(model_path, map_location=torch.device('cpu'))['model_initializer_dict']
            self.model_initializer.load_state_dict(model_dict)
            print_log(model_path, log=self.log)
        performance = {'FDE': [0, 0, 0, 0],
                       'ADE': [0, 0, 0, 0]}
        samples = 0

        def prepare_seed(rand_seed):
            np.random.seed(rand_seed)
            random.seed(rand_seed)
            torch.manual_seed(rand_seed)
            torch.cuda.manual_seed_all(rand_seed)

        prepare_seed(0)
        count = 0
        with torch.no_grad():
            for data in self.test_loader:
                past_traj, nbrs_traj, mask, lat_enc, lon_enc, fut_traj, op_mask, _ = data
                past_traj = past_traj.to(self.device)
                nbrs_traj = nbrs_traj.to(self.device)
                fut_traj = fut_traj.to(self.device)
                traj_mask = mask.bool().to(self.device)
                pred_traj = None
                if self.pretrain is False:
                    sample_prediction, mean_estimation, variance_estimation = self.model_initializer(past_traj,
                                                                                                     nbrs_traj,
                                                                                                     traj_mask)
                    sample_prediction = torch.exp(variance_estimation / 2)[
                                            ..., None, None] * sample_prediction / sample_prediction.std(dim=1).mean(
                        dim=(1, 2))[:, None, None, None]
                    loc = sample_prediction + mean_estimation[:, None]
                    pred_traj = self.p_sample_loop_accelerate(past_traj, nbrs_traj, traj_mask, loc)
                else:
                    pred_traj = self.p_sample_loop(past_traj, nbrs_traj, traj_mask,
                                                   [fut_traj.shape[1], fut_traj.shape[0], fut_traj.shape[2]])
                fut_traj = fut_traj.permute(1, 0, 2).unsqueeze(1).repeat(1, 24, 1, 1)
                # b*n, K, T, 2
                distances = torch.norm(fut_traj - pred_traj, dim=-1) * self.traj_scale
                for time_i in range(1, 5):
                    ade = (distances[:, :, :5 * time_i]).mean(dim=-1).min(dim=-1)[0].sum()
                    fde = (distances[:, :, 5 * time_i - 1]).min(dim=-1)[0].sum()
                    performance['ADE'][time_i - 1] += ade.item()
                    performance['FDE'][time_i - 1] += fde.item()
                samples += distances.shape[0]
                count += 1
        for time_i in range(4):
            print_log('--ADE({}s): {:.4f}\t--FDE({}s): {:.4f}'.format(time_i + 1, performance['ADE'][time_i] / samples, \
                                                                      time_i + 1, performance['FDE'][time_i] / samples),
                      log=self.log)
