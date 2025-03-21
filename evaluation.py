import torch
import numpy as np

class Evaluation:

    def __init__(self, dataset, dataloader, user_count, h0_strategy, trainer, setting):
        self.dataset = dataset
        self.dataloader = dataloader
        self.user_count = user_count # total number of users
        self.h0_strategy = h0_strategy # strategy for initializing/resetting the hidden state for the model
        self.trainer = trainer # trainer object that has an evaluate method to perform forward passes and produce predictions
        self.setting = setting # hyperparameters, device info, batch size, etc

    def evaluate(self):
        # reinitialize dataset's internal state
        self.dataset.reset()

        # initializes the hidden state for the model.
        # the batch size and computing device (CPU/GPU) are provided from self.setting
        h = self.h0_strategy.on_init(self.setting.batch_size, self.setting.device)

        with torch.no_grad(): # no gradients are computed, saving memory and computation
            iter_cnt = 0 # a counter for the total number of prediction iterations (across all batches/users)

            # these accumulate the total recall at 1, 5, and 10 respectively.
            recall1 = 0
            recall5 = 0
            recall10 = 0

            # mean average precision
            average_precision = 0.

            u_iter_cnt = np.zeros(self.user_count)
            u_recall1 = np.zeros(self.user_count)
            u_recall5 = np.zeros(self.user_count)
            u_recall10 = np.zeros(self.user_count)
            u_average_precision = np.zeros(self.user_count)

            # counter for how many times each user has been resetted.
            reset_count = torch.zeros(self.user_count)

            # x: The input sequences (for example, location IDs).
            # t: Timestamps associated with each input in the sequence.
            # t_slot: Time slot information (derived from the timestamps).
            # s: Spatial features such as coordinates.
            # y: The label sequences, which represent the target values (typically the next location).
            # y_t: Timestamps for the labels.
            # y_t_slot: Time slot information for the labels.
            # y_s: Spatial features for the labels.
            # reset_h: A flag (or list of flags) indicating if a user's sequence pointer should be reset.
            # active_users: The indices of users in the current batch.

            # loop over batches
            for i, (x, t, t_slot, s, y, y_t, y_t_slot, y_s, reset_h, active_users) in enumerate(self.dataloader):
                active_users = active_users.squeeze() # reduce any dimensions of size 1 from the tensor

                # iterate over reset_h which contains flags indicating whether a particular user's sequence pointer needs to be reset
                for j, reset in enumerate(reset_h):
                    # a reset typically means that the current sequence for the user has finished, and we need to reinitialize the hidden state for that user
                    if reset:
                        # LSTM has two hidden state components
                        if self.setting.is_lstm:
                            hc = self.h0_strategy.on_reset_test(active_users[j], self.setting.device)
                            h[0][0, j] = hc[0]
                            h[1][0, j] = hc[1]
                        # otherwise, one hidden state
                        else:
                            h[0, j] = self.h0_strategy.on_reset_test(active_users[j], self.setting.device)
                        reset_count[active_users[j]] += 1

                # squeeze for reasons of "loader-batch-size-is-1"
                x = x.squeeze().to(self.setting.device)
                t = t.squeeze().to(self.setting.device)
                t_slot = t_slot.squeeze().to(self.setting.device)
                s = s.squeeze().to(self.setting.device)

                y = y.squeeze()
                y_t = y_t.squeeze().to(self.setting.device)
                y_t_slot = y_t_slot.squeeze().to(self.setting.device)
                y_s = y_s.squeeze().to(self.setting.device)
                active_users = active_users.to(self.setting.device)

                # evaluate:
                out, h = self.trainer.evaluate(x, t, t_slot, s, y_t, y_t_slot, y_s, h, active_users)

                for j in range(self.setting.batch_size):
                    # o contains a per user list of votes for all locations for each sequence entry
                    o = out[j]

                    # partition elements
                    o_n = o.cpu().detach().numpy()
                    ind = np.argpartition(o_n, -10, axis=1)[:, -10:] # top 10 elements

                    y_j = y[:, j]

                    for k in range(len(y_j)):
                        if (reset_count[active_users[j]] > 1):
                            continue # skip already evaluated users.

                        # resort indices for k:
                        ind_k = ind[k]
                        r = ind_k[np.argsort(-o_n[k, ind_k], axis=0)] # sort top 10 elements descending

                        r = torch.tensor(r)
                        t = y_j[k]

                        # compute MAP:
                        r_kj = o_n[k, :]
                        t_val = r_kj[t]
                        upper = np.where(r_kj > t_val)[0]
                        precision = 1. / (1+len(upper))

                        # store
                        u_iter_cnt[active_users[j]] += 1
                        u_recall1[active_users[j]] += t in r[:1]
                        u_recall5[active_users[j]] += t in r[:5]
                        u_recall10[active_users[j]] += t in r[:10]
                        u_average_precision[active_users[j]] += precision

            formatter = "{0:.8f}"
            for j in range(self.user_count):
                iter_cnt += u_iter_cnt[j]
                recall1 += u_recall1[j]
                recall5 += u_recall5[j]
                recall10 += u_recall10[j]
                average_precision += u_average_precision[j]

                if (self.setting.report_user > 0 and (j+1) % self.setting.report_user == 0):
                    print('Report user', j, 'preds:', u_iter_cnt[j], 'recall@1', formatter.format(u_recall1[j]/u_iter_cnt[j]), 'MAP', formatter.format(u_average_precision[j]/u_iter_cnt[j]), sep='\t')

            print('recall@1:', formatter.format(recall1/iter_cnt))
            print('recall@5:', formatter.format(recall5/iter_cnt))
            print('recall@10:', formatter.format(recall10/iter_cnt))
            print('MAP', formatter.format(average_precision/iter_cnt))
            print('predictions:', iter_cnt)
