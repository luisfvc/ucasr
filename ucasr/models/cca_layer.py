
import torch

import torch.nn as nn


class CCALayer(nn.Module):

    def __init__(self, in_dim, alpha=1.0, wl=0.0, normalized=False):
        super(CCALayer, self).__init__()

        self.in_dim = in_dim
        self.r1 = self.r2 = self.rT = 1e-3

        self.alpha = alpha
        self.wl = wl
        self.normalized = normalized

        self.U = nn.Parameter(torch.zeros(in_dim, in_dim), requires_grad=False)
        self.V = nn.Parameter(torch.zeros(in_dim, in_dim), requires_grad=False)

        self.mean1 = nn.Parameter(torch.zeros(in_dim), requires_grad=False)
        self.mean2 = nn.Parameter(torch.zeros(in_dim), requires_grad=False)

        self.S11 = nn.Parameter(torch.zeros(in_dim, in_dim), requires_grad=False)
        self.S22 = nn.Parameter(torch.zeros(in_dim, in_dim), requires_grad=False)
        self.S12 = nn.Parameter(torch.zeros(in_dim, in_dim), requires_grad=False)

        self.eye_matrix = nn.Parameter(torch.eye(in_dim), requires_grad=False)

        self.loss = 0
        self.corr = 0

    def forward(self, H1, H2):

        if self.training:
            m = H1.shape[0]

            # compute batch mean
            mean1 = H1.mean(dim=0)
            mean2 = H2.mean(dim=0)

            # running average updates of means
            self.mean1.data = ((1 - self.alpha)*self.mean1 + self.alpha * mean1)
            self.mean2.data = ((1 - self.alpha)*self.mean2 + self.alpha * mean2)

            # hidden representations and transpose to correlation format
            H1bar = (H1 - self.mean1).t()
            H2bar = (H2 - self.mean2).t()

            S11 = (1/(m-1)) * torch.matmul(H1bar, H1bar.t())
            S11 += self.r1 * self.eye_matrix

            S22 = (1/(m-1)) * torch.matmul(H2bar, H2bar.t())
            S22 += self.r2 * self.eye_matrix

            S12 = (1/(m-1)) * torch.matmul(H1bar, H2bar.t())

            # running average updates of statistics
            self.S11.data = (1 - self.alpha) * self.S11 + self.alpha * S11
            self.S22.data = (1 - self.alpha) * self.S22 + self.alpha * S22
            self.S12.data = (1 - self.alpha) * self.S12 + self.alpha * S12

            # computing the inverse square root matrices
            d, A = torch.linalg.eigh(S11)
            S11si = (A * torch.reciprocal(d.sqrt())).matmul(A.t())      # = S11^-.5

            d, A = torch.linalg.eigh(S22)
            S22si = (A * torch.reciprocal(d.sqrt())).matmul(A.t())      # = S22^-.5

            # compute TT' and T'T (regularized)
            Tnp = S11si.matmul(S12).matmul(S22si)

            # U, Values, V = torch.svd(Tnp)
            #
            # Coeffs = Values
            #
            # U = S11.matmul(U)
            # V = S22.matmul(V.t())

            M1 = Tnp.matmul(Tnp.t())
            # M1 += self.rT * torch.eye(M1.shape[0])
            M1 += self.rT * self.eye_matrix

            M2 = Tnp.t().matmul(Tnp)
            # M2 += self.rT * torch.eye(M2.shape[0])
            M2 += self.rT * self.eye_matrix

            # compute eigen decomposition
            E1, E = torch.linalg.eigh(M1)
            _, F = torch.linalg.eigh(M2)

            # maximize correlation
            E1 = torch.clamp(E1, 1e-7, 1.0).sqrt()
            self.loss = - E1.mean() * self.wl
            self.corr = E1

            # compute projection matrices
            U = S11si.matmul(E)
            V = S22si.matmul(F)

            s = torch.sign(U.t().matmul(S12).matmul(V).diagonal())
            U = s * U

            self.U.data = U
            self.V.data = V

        else:

            # hidden representations
            H1bar = H1 - self.mean1
            H2bar = H2 - self.mean2

            # transpose to formulas in paper
            H1bar = H1bar.t()
            H2bar = H2bar.t()

            U, V = self.U, self.V

        # re-project data
        lv1_cca = H1bar.t().matmul(U)
        lv2_cca = H2bar.t().matmul(V)

        return lv1_cca, lv2_cca


# todo: merge CCARefine into the CCALayer module
class CCARefine(nn.Module):

    def __init__(self, in_dim, alpha=1.0, wl=0.0):
        super(CCARefine, self).__init__()

        self.in_dim = in_dim
        self.r1 = self.r2 = self.rT = 1e-3

        self.alpha = alpha
        self.wl = wl

        self.U = nn.Parameter(torch.zeros(in_dim, in_dim), requires_grad=False)
        self.V = nn.Parameter(torch.zeros(in_dim, in_dim), requires_grad=False)

        self.mean1 = nn.Parameter(torch.zeros(in_dim), requires_grad=False)
        self.mean2 = nn.Parameter(torch.zeros(in_dim), requires_grad=False)

        self.eye_matrix = nn.Parameter(torch.eye(in_dim), requires_grad=False)

    def forward(self, H1, H2):

        # number of observations
        m = H1.shape[0]

        # compute batch mean
        self.mean1.data = H1.mean(dim=0)
        self.mean2.data = H2.mean(dim=0)

        # hidden representations and transpose to correlation format
        H1bar = (H1 - self.mean1).t()
        H2bar = (H2 - self.mean2).t()

        # covariance 1
        S11 = (1.0 / (m - 1)) * torch.matmul(H1bar, H1bar.t())
        S11 += self.r1 * self.eye_matrix

        # covariance 2
        S22 = (1.0 / (m - 1)) * torch.matmul(H2bar, H2bar.t())
        S22 += self.r2 * self.eye_matrix

        # cross-covariance
        S12 = (1.0 / (m - 1)) * torch.matmul(H1bar, H2bar.t())

        # computing the inverse squared root of S11 and S22
        d, A = torch.linalg.eigh(S11)
        S11si = (A * torch.reciprocal(d.sqrt())).matmul(A.t())      # = S11^-.5

        d, A = torch.linalg.eigh(S22)
        S22si = (A * torch.reciprocal(d.sqrt())).matmul(A.t())      # = S22^-.5

        # compute TT' and T'T (regularized)
        Tnp = S11si.matmul(S12).matmul(S22si)

        # # singular value decomposition
        # todo: find out why V is negative using the pytorch.linalg.svd()
        # U, Values, V = torch.linalg.svd(Tnp)
        #
        # Coeffs = Values
        #
        # U = S11.matmul(U)
        # V = S22.matmul(V.t())
        #
        # s = torch.sign(U.t().matmul(S12).matmul(V).diagonal())
        # U = s * U

        M1 = Tnp.matmul(Tnp.t())
        # M1 += self.rT * torch.eye(M1.shape[0])
        M1 += self.rT * self.eye_matrix

        M2 = Tnp.t().matmul(Tnp)
        # M2 += self.rT * torch.eye(M2.shape[0])
        M2 += self.rT * self.eye_matrix

        # compute eigen decomposition
        E1, E = torch.linalg.eigh(M1)
        _, F = torch.linalg.eigh(M2)

        # maximize correlation
        E1 = torch.clamp(E1, 1e-7, 1.0).sqrt()
        self.loss = - E1.mean() * self.wl
        self.corr = E1

        # compute projection matrices
        U = S11si.matmul(E)
        V = S22si.matmul(F)

        s = torch.sign(U.t().matmul(S12).matmul(V).diagonal())
        U = s * U

        self.U.data = U
        self.V.data = V

        Coeffs = self.corr

        return Coeffs


if __name__ == "__main__":
    pass
