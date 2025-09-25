# coding: utf-8
# @email: enoche.chow@gmail.com

r"""
################################
"""

import os
import itertools
import torch
import torch.optim as optim
from torch.nn.utils.clip_grad import clip_grad_norm_
import numpy as np
import matplotlib.pyplot as plt

from time import time
from logging import getLogger

from utils.utils import get_local_time, early_stopping, dict2str
from utils.topk_evaluator import TopKEvaluator


class AbstractTrainer(object):
    r"""Trainer Class is used to manage the training and evaluation processes of recommender system models.
    AbstractTrainer is an abstract class in which the fit() and evaluate() method should be implemented according
    to different training and evaluation strategies.
    """

    def __init__(self, config, model):
        self.config = config
        self.model = model

    def fit(self, train_data):
        r"""Train the model based on the train data.

        """
        raise NotImplementedError('Method [next] should be implemented.')

    def evaluate(self, eval_data):
        r"""Evaluate the model based on the eval data.

        """

        raise NotImplementedError('Method [next] should be implemented.')


class Trainer(AbstractTrainer):
    r"""The basic Trainer for basic training and evaluation strategies in recommender systems. This class defines common
    functions for training and evaluation processes of most recommender system models, including fit(), evaluate(),
   and some other features helpful for model training and evaluation.

    Generally speaking, this class can serve most recommender system models, If the training process of the model is to
    simply optimize a single loss without involving any complex training strategies, such as adversarial learning,
    pre-training and so on.

    Initializing the Trainer needs two parameters: `config` and `model`. `config` records the parameters information
    for controlling training and evaluation, such as `learning_rate`, `epochs`, `eval_step` and so on.
    More information can be found in [placeholder]. `model` is the instantiated object of a Model Class.

    """

    def __init__(self, config, model, mg=False):
        super(Trainer, self).__init__(config, model)

        self.logger = getLogger()
        self.learner = config['learner']
        self.learning_rate = config['learning_rate']
        self.epochs = config['epochs']
        self.eval_step = min(config['eval_step'], self.epochs)
        self.stopping_step = config['stopping_step']
        self.clip_grad_norm = config['clip_grad_norm']
        self.valid_metric = config['valid_metric'].lower()
        self.valid_metric_bigger = config['valid_metric_bigger']
        self.test_batch_size = config['eval_batch_size']
        self.device = config['device']
        self.weight_decay = 0.0
        if config['weight_decay'] is not None:
            wd = config['weight_decay']
            self.weight_decay = eval(wd) if isinstance(wd, str) else wd

        self.req_training = config['req_training']

        self.start_epoch = 0
        self.cur_step = 0

        tmp_dd = {}
        for j, k in list(itertools.product(config['metrics'], config['topk'])):
            tmp_dd[f'{j.lower()}@{k}'] = 0.0
        self.best_valid_score = -1
        self.best_valid_result = tmp_dd
        self.best_test_upon_valid = tmp_dd
        self.train_loss_dict = dict()
        self.optimizer = self._build_optimizer()

        #fac = lambda epoch: 0.96 ** (epoch / 50)
        lr_scheduler = config['learning_rate_scheduler']        # check zero?
        fac = lambda epoch: lr_scheduler[0] ** (epoch / lr_scheduler[1])
        scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=fac)
        self.lr_scheduler = scheduler

        self.eval_type = config['eval_type']
        self.evaluator = TopKEvaluator(config)

        self.item_tensor = None
        self.tot_item_num = None
        self.mg = mg
        self.alpha1 = config['alpha1']
        self.alpha2 = config['alpha2']
        self.beta = config['beta']

        # Optional TensorBoard
        self.tb_enabled = bool(self.config.get('tensorboard', False))
        self.tb_log_dir = self.config.get('tb_log_dir', None)
        self.tb_writer = None
        if self.tb_enabled:
            try:
                from torch.utils.tensorboard import SummaryWriter  # noqa: F401
            except Exception as e:
                self.logger.warning(f"TensorBoard unavailable ({e}); disabling TB logging.")
                self.tb_enabled = False
        self._last_grad_groups = {}
        self.mg_target_rel_step = float(self.config.get('mg_target_rel_step', 1e-3))
        self.mg_alpha_max_scale = float(self.config.get('mg_alpha_max_scale', 20.0))


    def _build_optimizer(self):
        r"""Init the Optimizer

        Returns:
            torch.optim: the optimizer
        """
        if self.learner.lower() == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'adagrad':
            optimizer = optim.Adagrad(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'rmsprop':
            optimizer = optim.RMSprop(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            self.logger.warning('Received unrecognized optimizer, set default Adam optimizer')
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer

    def _train_epoch(self, train_data, epoch_idx, loss_func=None):
        r"""Train the model in an epoch

        Args:
            train_data (DataLoader): The train data.
            epoch_idx (int): The current epoch id.
            loss_func (function): The loss function of :attr:`model`. If it is ``None``, the loss function will be
                :attr:`self.model.calculate_loss`. Defaults to ``None``.

        Returns:
            float/tuple: The sum of loss returned by all batches in this epoch. If the loss in each batch contains
            multiple parts and the model return these multiple parts loss instead of the sum of loss, It will return a
            tuple which includes the sum of loss in each part.
        """
        if not self.req_training:
            return 0.0, []
        self.model.train()
        loss_func = loss_func or self.model.calculate_loss
        total_loss = None
        loss_batches = []
        # collect gradient norms per module group if TB enabled
        grad_groups = {}

        def _group_name(n):
            prefixes = [
                'image_trs', 'text_trs', 'query_v', 'query_t', 'gate_v', 'gate_t', 'gate_f',
                'gate_image_prefer', 'gate_text_prefer', 'gate_fusion_prefer',
                'user_embedding', 'item_id_embedding', 'image_embedding', 'text_embedding',
                'image_complex_weight', 'text_complex_weight', 'fusion_complex_weight'
            ]
            for p in prefixes:
                if n.startswith(p):
                    return p
            return n.split('.')[0]

        def _zero_grad(opt):
            try:
                opt.zero_grad(set_to_none=True)
            except TypeError:
                opt.zero_grad()

        for batch_idx, interaction in enumerate(train_data):
            _zero_grad(self.optimizer)
            try:
                second_inter = interaction.clone()
            except AttributeError:
                second_inter = interaction
            losses = loss_func(interaction)
            
            if isinstance(losses, tuple):
                loss = sum(losses)
                loss_tuple = tuple(per_loss.item() for per_loss in losses)
                total_loss = loss_tuple if total_loss is None else tuple(map(sum, zip(total_loss, loss_tuple)))
            else:
                loss = losses
                total_loss = losses.item() if total_loss is None else total_loss + losses.item()
            if self._check_nan(loss):
                self.logger.info('Loss is nan at epoch: {}, batch index: {}. Exiting.'.format(epoch_idx, batch_idx))
                return loss, torch.tensor(0.0)

            model_has_mirror = bool(getattr(self.model, 'mg_enable', False))
            if not model_has_mirror:
                if self.mg and batch_idx % self.beta == 0:
                    first_loss = self.alpha1 * loss
                    first_loss.backward()

                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    losses = loss_func(second_inter)
                    if isinstance(losses, tuple):
                        loss = sum(losses)
                    else:
                        loss = losses

                    if self._check_nan(loss):
                        self.logger.info('Loss is nan at epoch: {}, batch index: {}. Exiting.'.format(epoch_idx, batch_idx))
                        return loss, torch.tensor(0.0)
                    second_loss = -1 * self.alpha2 * loss
                    second_loss.backward()
                else:
                    loss.backward()

                if self.tb_enabled:
                    with torch.no_grad():
                        for name, p in self.model.named_parameters():
                            if p.grad is not None:
                                gnorm = torch.norm(p.grad.detach()).item()
                                group = _group_name(name)
                                grad_groups.setdefault(group, []).append(gnorm)

                if self.clip_grad_norm:
                    clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)
                self.optimizer.step()
                loss_batches.append(loss.detach())
                continue

            loss.backward()
            
            # collect grads before step
            if self.tb_enabled:
                with torch.no_grad():
                    for name, p in self.model.named_parameters():
                        if p.grad is not None:
                            gnorm = torch.norm(p.grad.detach()).item()
                            group = _group_name(name)
                            grad_groups.setdefault(group, []).append(gnorm)

            if self.clip_grad_norm:
                clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)
            self.optimizer.step()
            loss_batches.append(loss.detach())

            # optional diagnostics for multi-modal models such as SMORE
            log_interval = int(getattr(self.model, 'mg_log_interval', 50))
            if log_interval > 0:
                step_id = int(getattr(self.model, 'global_step', 0))
                if step_id and step_id % log_interval == 0 and hasattr(self.model, 'log_mm_diagnostics'):
                    try:
                        self.model.log_mm_diagnostics(self.optimizer)
                    except Exception as diag_err:
                        self.logger.warning(f"log_mm_diagnostics failed at step {step_id}: {diag_err}")

            if getattr(self.model, 'mg_enable', False):
                mg_interval = int(getattr(self.model, 'mg_interval', 0))
                if mg_interval > 0:
                    step_id = int(getattr(self.model, 'global_step', 0))
                    if step_id % mg_interval == 0:
                        lr = self.optimizer.param_groups[0].get('lr', 1.0)

                        # ===== 1) 在“原点 θ”重新计算 g(θ)（用 second_inter 减少额外IO）=====
                        _zero_grad(self.optimizer)
                        mirror_input = second_inter
                        loss_curr = loss_func(mirror_input)
                        mirror_scalar_curr = sum(loss_curr) if isinstance(loss_curr, tuple) else loss_curr
                        mirror_scalar_curr.backward()

                        # 缓存参数和梯度
                        params, grads = [], []
                        for p in self.model.parameters():
                            if p.requires_grad and p.grad is not None:
                                params.append(p)
                                grads.append(p.grad.detach().clone())

                        # ===== 2) 计算 grad_rms / param_rms 并得到自适应 alpha_eff =====
                        with torch.no_grad():
                            if len(grads) == 0:
                                alpha_eff = float(getattr(self.model, 'mg_alpha', 0.5))  # 兜底
                            else:
                                g_all = torch.cat([g.view(-1) for g in grads])
                                grad_rms = float(g_all.norm() / (g_all.numel() ** 0.5))
                                p_all = torch.cat([p.detach().view(-1) for p in params])
                                param_rms = float(p_all.norm() / (p_all.numel() ** 0.5) + 1e-12)

                                alpha_base = float(getattr(self.model, 'mg_alpha', 0.5))
                                target_rel = float(getattr(self, 'mg_target_rel_step', 1e-3))
                                target_step = target_rel * param_rms
                                # 关键：alpha_eff = max(alpha_base, target_step / (lr * grad_rms + eps))
                                alpha_eff = max(alpha_base, target_step / (lr * grad_rms + 1e-12))
                                max_scale = float(getattr(self, 'mg_alpha_max_scale', 20.0))
                                alpha_eff = min(alpha_eff, alpha_base * max_scale)

                            # 把 alpha_eff 暴露给 model（便于日志）
                            try:
                                setattr(self.model, '_alpha_eff', float(alpha_eff))
                            except Exception:
                                pass

                        # ===== 3) 去镜像点 θ' = θ - alpha_eff * lr * g(θ) =====
                        with torch.no_grad():
                            for p, g in zip(params, grads):
                                p.add_(- alpha_eff * lr * g)

                        # ===== 4) 在镜像点计算 g(θ')，梯度取反并按 beta 缩放 =====
                        _zero_grad(self.optimizer)
                        loss_mirror = loss_func(mirror_input)
                        mirror_scalar = sum(loss_mirror) if isinstance(loss_mirror, tuple) else loss_mirror
                        mirror_scalar.backward()

                        with torch.no_grad():
                            for p in self.model.parameters():
                                if p.requires_grad and p.grad is not None:
                                    p.grad.mul_(- float(getattr(self.model, 'mg_beta', 0.2)))

                        # ===== 5) 还原回原点 θ 并用“反号镜像梯度”更新 =====
                        with torch.no_grad():
                            for p, g in zip(params, grads):
                                p.add_(+ alpha_eff * lr * g)

                        self.optimizer.step()
                        _zero_grad(self.optimizer)

                        # ===== 6) 日志（可选）=====
                        if getattr(self.model, 'mg_verbose', False):
                            with torch.no_grad():
                                def _safe_norm(x): return float(x.norm().item()) if x is not None else float('nan')
                                g_user = next((p.grad for n, p in self.model.named_parameters()
                                            if n.startswith('user_embedding.weight')), None)
                                g_item = next((p.grad for n, p in self.model.named_parameters()
                                            if n.startswith('item_id_embedding.weight')), None)
                                mv = float(mirror_scalar.item()) if torch.is_tensor(mirror_scalar) else float(mirror_scalar)
                                ae = float(getattr(self.model, '_alpha_eff', float(getattr(self.model, 'mg_alpha', 0.5))))
                                print(f"[MG] step={step_id} mirror_loss={mv:.4f} α_eff={ae:.3g} "
                                    f"||g_user||={_safe_norm(g_user):.4f} ||g_item||={_safe_norm(g_item):.4f}")

            # for test
            #if batch_idx == 0:
            #    break
        # save averaged grad norms for this epoch
        if self.tb_enabled:
            self._last_grad_groups = {k: float(sum(v) / max(len(v), 1)) for k, v in grad_groups.items()}
        return total_loss, loss_batches

    def _valid_epoch(self, valid_data):
        r"""Valid the model with valid data

        Args:
            valid_data (DataLoader): the valid data

        Returns:
            float: valid score
            dict: valid result
        """
        valid_result = self.evaluate(valid_data)
        valid_score = valid_result[self.valid_metric] if self.valid_metric else valid_result['NDCG@20']
        return valid_score, valid_result

    def _check_nan(self, loss):
        if torch.isnan(loss):
            #raise ValueError('Training loss is nan')
            return True

    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, losses):
        train_loss_output = 'epoch %d training [time: %.2fs, ' % (epoch_idx, e_time - s_time)
        if isinstance(losses, tuple):
            train_loss_output = ', '.join('train_loss%d: %.4f' % (idx + 1, loss) for idx, loss in enumerate(losses))
        else:
            train_loss_output += 'train loss: %.4f' % losses
        return train_loss_output + ']'

    def fit(self, train_data, valid_data=None, test_data=None, saved=False, verbose=True):
        r"""Train the model based on the train data and the valid data.

        Args:
            train_data (DataLoader): the train data
            valid_data (DataLoader, optional): the valid data, default: None.
                                               If it's None, the early_stopping is invalid.
            test_data (DataLoader, optional): None
            verbose (bool, optional): whether to write training and evaluation information to logger, default: True
            saved (bool, optional): whether to save the model parameters, default: True

        Returns:
             (float, dict): best valid score and best valid result. If valid_data is None, it returns (-1, None)
        """
        # init TB writer lazily
        if self.tb_enabled and self.tb_writer is None:
            from torch.utils.tensorboard import SummaryWriter
            run_name = f"{self.config['model']}_{self.config['dataset']}_{get_local_time()}"
            base_dir = self.tb_log_dir or os.path.join('log', 'tensorboard')
            full_dir = os.path.join('src', base_dir) if not os.path.isabs(base_dir) else base_dir
            os.makedirs(full_dir, exist_ok=True)
            self.tb_writer = SummaryWriter(log_dir=os.path.join(full_dir, run_name))

        for epoch_idx in range(self.start_epoch, self.epochs):
            # train
            training_start_time = time()
            # expose current epoch to model for scheduling if needed
            try:
                setattr(self.model, 'cur_epoch', epoch_idx)
            except Exception:
                pass
            self.model.pre_epoch_processing()
            train_loss, _ = self._train_epoch(train_data, epoch_idx)
            if torch.is_tensor(train_loss):
                # get nan loss
                break
            #for param_group in self.optimizer.param_groups:
            #    print('======lr: ', param_group['lr'])
            self.lr_scheduler.step()

            self.train_loss_dict[epoch_idx] = sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            training_end_time = time()
            train_loss_output = \
                self._generate_train_loss_output(epoch_idx, training_start_time, training_end_time, train_loss)
            post_info = self.model.post_epoch_processing()
            if verbose:
                self.logger.info(train_loss_output)
                if post_info is not None:
                    self.logger.info(post_info)

            # per-epoch TensorBoard logging
            if self.tb_enabled and self.tb_writer is not None:
                # losses
                if isinstance(train_loss, tuple):
                    total = float(sum(train_loss))
                    self.tb_writer.add_scalar('loss/total', total, epoch_idx)
                    for i, l in enumerate(train_loss):
                        self.tb_writer.add_scalar(f'loss/part_{i+1}', float(l), epoch_idx)
                else:
                    self.tb_writer.add_scalar('loss/total', float(train_loss), epoch_idx)
                # learning rate
                try:
                    self.tb_writer.add_scalar('opt/lr', self.optimizer.param_groups[0]['lr'], epoch_idx)
                except Exception:
                    pass
                # gradient norms per module group (averaged over epoch)
                for gk, gv in self._last_grad_groups.items():
                    self.tb_writer.add_scalar(f'grad_norm/{gk}', gv, epoch_idx)
                # parameter norms quick view (top-level)
                with torch.no_grad():
                    for name, p in self.model.named_parameters():
                        if p.requires_grad and p.data is not None:
                            self.tb_writer.add_scalar(f'param_norm/{name.split(".")[0]}', torch.norm(p.data).item(), epoch_idx)
                # model diagnostics if provided
                if hasattr(self.model, 'tb_diagnostics'):
                    try:
                        diag = self.model.tb_diagnostics()
                        if isinstance(diag, dict):
                            for k, v in diag.items():
                                if isinstance(v, (int, float)):
                                    self.tb_writer.add_scalar(f'model/{k}', float(v), epoch_idx)
                    except Exception as e:
                        self.logger.warning(f"tb_diagnostics failed: {e}")

            # eval: To ensure the test result is the best model under validation data, set self.eval_step == 1
            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                valid_score, valid_result = self._valid_epoch(valid_data)
                self.best_valid_score, self.cur_step, stop_flag, update_flag = early_stopping(
                    valid_score, self.best_valid_score, self.cur_step,
                    max_step=self.stopping_step, bigger=self.valid_metric_bigger)
                valid_end_time = time()
                valid_score_output = "epoch %d evaluating [time: %.2fs, valid_score: %f]" % \
                                     (epoch_idx, valid_end_time - valid_start_time, valid_score)
                valid_result_output = 'valid result: \n' + dict2str(valid_result)
                # test
                _, test_result = self._valid_epoch(test_data)
                if verbose:
                    self.logger.info(valid_score_output)
                    self.logger.info(valid_result_output)
                    self.logger.info('test result: \n' + dict2str(test_result))
                if update_flag:
                    update_output = '██ ' + self.config['model'] + '--Best validation results updated!!!'
                    if verbose:
                        self.logger.info(update_output)
                    self.best_valid_result = valid_result
                    self.best_test_upon_valid = test_result

                if stop_flag:
                    stop_output = '+++++Finished training, best eval result in epoch %d' % \
                                  (epoch_idx - self.cur_step * self.eval_step)
                    if verbose:
                        self.logger.info(stop_output)
                    break
        # close TB writer if open
        if self.tb_writer is not None:
            try:
                self.tb_writer.flush()
                self.tb_writer.close()
            except Exception:
                pass
        return self.best_valid_score, self.best_valid_result, self.best_test_upon_valid


    @torch.no_grad()
    def evaluate(self, eval_data, is_test=False, idx=0):
        r"""Evaluate the model based on the eval data.
        Returns:
            dict: eval result, key is the eval metric and value in the corresponding metric value
        """
        self.model.eval()

        # batch full users
        batch_matrix_list = []
        for batch_idx, batched_data in enumerate(eval_data):
            # predict: interaction without item ids
            scores = self.model.full_sort_predict(batched_data)
            masked_items = batched_data[1]
            # mask out pos items
            scores[masked_items[0], masked_items[1]] = -1e10
            # rank and get top-k
            _, topk_index = torch.topk(scores, max(self.config['topk']), dim=-1)  # nusers x topk
            batch_matrix_list.append(topk_index)
        return self.evaluator.evaluate(batch_matrix_list, eval_data, is_test=is_test, idx=idx)

    def plot_train_loss(self, show=True, save_path=None):
        r"""Plot the train loss in each epoch

        Args:
            show (bool, optional): whether to show this figure, default: True
            save_path (str, optional): the data path to save the figure, default: None.
                                       If it's None, it will not be saved.
        """
        epochs = list(self.train_loss_dict.keys())
        epochs.sort()
        values = [float(self.train_loss_dict[epoch]) for epoch in epochs]
        plt.plot(epochs, values)
        plt.xticks(epochs)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        if show:
            plt.show()
        if save_path:
            plt.savefig(save_path)
