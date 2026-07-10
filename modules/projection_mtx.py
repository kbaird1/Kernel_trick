import torch
from pykeops.torch import LazyTensor
from linear_operator.operators import KernelLinearOperator


class ProjectionMatrix():
    '''A KeOps implementation of the projection matrix that maps a fitness vector to a set of gauge-fixed weights'''    
    def __init__(self, alphabet, seq_length):
        self.alphabet = alphabet
        self.alphabet_set = set(alphabet)
        self.n_alleles = len(self.alphabet_set)
        self.seq_length = seq_length
        
    def select_site(self, x, site):
        idx = torch.arange(site * self.n_alleles, (site + 1) * self.n_alleles)
        return x.index_select(-1, idx.to(device=x.device))
    
    # site-specific values
    def get_A(self, pis, eta):
        A = pis[:,:,None]*torch.ones(self.n_alleles+1)
        A = torch.einsum('ijk->ikj', A)
        A*=-eta
        indices = torch.arange(self.n_alleles)
        A[...,indices, indices] += 1
        A[...,-1,:]*=-1
        return A

    def validate_sequences(self, sequences, alphabet):
        alphabet_set = set(alphabet)
        for seq in sequences:
            assert len(seq) == self.seq_length, f"Sequence {seq} does not have length {self.seq_length}"
            assert set(seq).issubset(alphabet_set), f"Invalid characters in sequence {seq}"

    def all_combinations(self, alphabet, L):
        combos = ['']
        for _ in range(L):
            combos = [prefix + char for prefix in combos for char in alphabet]
        return combos
    
        
    # what Carlos calls _covar_func
    def _sitewise_prod_covar_func(self, x1, x2, **kwargs):
        x1_ = LazyTensor(self.select_site(x1, site=0)[:, None, :])
        x2_ = LazyTensor(self.select_site(x2, site=0)[None, :, :])
        K = (x1_ * x2_).sum(-1)

        for i in range(1, self.seq_length):
            x1_ = LazyTensor(self.select_site(x1, site=i)[:, None, :])
            x2_ = LazyTensor(self.select_site(x2, site=i)[None, :, :])
            K *= (x1_ * x2_).sum(-1)

        return K

    # subseqs is the list of subseqs we want the weight of; use '-' for positions not involved
    # ordered seqs will order the columns of the projection matrix by given order. Default is given by all_combinations
    def compute_P(self, pis, eta, subseqs, ordered_seqs = None):
        assert(len(pis)==self.seq_length), f"need one pi per each of the {self.seq_length} positions"
        for pi in pis:
            assert(pi.shape[0]==self.n_alleles), f"each pi should be same length as the alphabet {self.n_alleles} "

        # make sure subseqs uses the alphabet and are all length L
        self.validate_sequences(subseqs, self.alphabet+'-')
        # Lazy-import from the selected epik checkout (fork vs dev).
        import epik_loader

        epik_loader.ensure_epik_loaded()
        from epik.epik.utils import encode_seqs

        # one-hot weights
        x  = encode_seqs(subseqs, self.alphabet+'-')
        # one-hot all sequence on the alphabet
        if ordered_seqs == None:
            y = encode_seqs(self.all_combinations(self.alphabet,self.seq_length),self.alphabet)
        else: 
            y = encode_seqs(ordered_seqs)
        
        M = torch.block_diag(torch.block_diag(*self.get_A(pis,eta))) # blocks are num_alleles + 1 X num_alleles
        P = KernelLinearOperator(
            x @ M, y, covar_func=self._sitewise_prod_covar_func)
        return P
        