import numpy as np
import torch

from models.MLP import MLP

class Ensemble():
    """
    Creates an ensemble of neural network functions with randomized prior
    functions (Osband et. al. 2018).
    """

    def __init__(self, params):
        self.params = params
        self.kappa = self.params['kappa']

        self.dtype = self.params['dtype']
        self.device = self.params['device']

        self.models = []
        self.priors = []
        self.optims = []

        for i in range(self.params['ens_size']):
            model = MLP(self.params['model_params']).to(device=self.device)
            self.models.append(model)
            self.optims.append(torch.optim.Adam(
                model.parameters(),
                lr=self.params['lr'], 
                weight_decay=self.params['reg']
            ))
            
            prior = MLP(self.params['model_params']).to(device=self.device)
            prior.eval()
            self.priors.append(prior)

    def train(self):
        for model in self.models:
            model.train()

    def eval(self):
        for model in self.models:
            model.eval()

    def pforward(self, ind, x):
        prior = self.params['prior_beta'] * self.priors[ind].forward(x)
        return self.models[ind].forward(x) + prior

    def get_preds(self, x):
        x = torch.tensor(x, dtype=self.dtype).to(device=self.device)
        return [self.pforward(i, x) for i in range(len(self.models))]

    def get_preds_np(self, x):
        preds = self.get_preds(x)
        preds = [pred.detach().cpu() for pred in preds]
        return preds

    def forward(self, x):
        x = x.to(device=self.device)
        preds = torch.zeros((x.shape[0], len(self.models)))
        for i in range(len(self.models)):
            preds[:,i] = self.pforward(i, x).squeeze(-1)

        if self.kappa is not None:
            # log n is correction for adding together multiple values
            n = torch.tensor(len(self.models), dtype=self.dtype)
            exp_term = self.kappa * preds - torch.log(n)
            lse = torch.logsumexp(exp_term, dim=1, keepdim=True)
            return (1 / self.kappa) * lse
        else:
            return torch.max(preds)

    def update_batch(self, ind, x, y, batch_size, num_steps):
        total_loss = 0
        for i in range(num_steps):
            bi, ei = i*batch_size, (i+1)*batch_size

            y_hat = self.pforward(ind, x[bi:ei])
            if len(y_hat.shape) > 1:
                y_hat = y_hat[:,0]

            ys = y[bi:ei]
            if len(ys.shape) > 1:
                ys = ys[:,0]

            loss = torch.nn.functional.mse_loss(ys, y_hat)

            self.optims[ind].zero_grad()
            loss.backward()
            self.optims[ind].step()

            total_loss += loss.item()

        return total_loss / num_steps

    def update_from_buf(self, buf, num_steps, batch_size, H, gamma):
        size = min(buf.size, buf.total_in)
        num_inds = batch_size*len(self.models)*num_steps
        inds = np.random.randint(0, size, size=num_inds)
        states = buf.buffer['s'][inds]

        x = np.zeros((len(inds), states[0].shape[0]))
        targets = np.zeros(len(inds))
        comp = {}

        self.eval()
        for i in range(len(inds)):
            x[i] = states[i]
            if inds[i] in comp:
                targets[i] = comp[inds[i]]
                continue
            dis, max_k = 1, min(H, buf.total_in-inds[i])
            for k in range(max_k):
                targets[i] += dis * buf.get('r', inds[i]+k)
                dis *= gamma
                if buf.get('d', inds[i]+k):
                    break

            sz = buf.get('ns', inds[i]+k)

            preds = self.get_preds_np(sz)
            targets[i] += dis * np.mean(preds)
            comp[inds[i]] = targets[i]

        targets += np.random.normal(0, self.params['rpf_noise'], targets.shape)
        x = torch.tensor(x, dtype=self.dtype).to(device=self.device)
        targets = torch.tensor(targets, dtype=self.dtype).to(device=self.device)

        self.train()
        for i in range(len(self.models)):
            bi, ei = i*batch_size*num_steps, (i+1)*batch_size*num_steps
            self.update_batch(i, x[bi:ei], targets[bi:ei], 
                batch_size, num_steps)

        self.eval()
