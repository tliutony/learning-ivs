import torch


class SamplingRelevanceLoss:
    """
    Bound the relevance between two variables and separate this sample from other samples
    """
    def __init__(self, temperature: float = 0.1, sample_size: int = 100):
        """
        Initialize the sampling relevance loss

        Args:
            temperature: temperature for the softmax
            sample_size: number of negative samples to sample
        """
        self.loss = torch.nn.CrossEntropyLoss()
        self.temperature = temperature
        self.sample_size = sample_size

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate the InfoNCE loss on the given two variables, sample the negative samples from the given variable

        Args:
            x: a batch of samples' logits
        """
        # sample fixed number of positive samples and randomly sample a batch of y as negative samples
        N = x.shape[0]
        pos_logits = torch.bmm(x.view(N, 1, -1), x.view(N, -1, 1)).squeeze() / self.temperature
        neg_logits = []
        # iterate through all x
        for i in range(N):
            mask = torch.ones(N, dtype=torch.bool)
            mask[i] = 0
            neg_indices = torch.arange(N)[mask]
            neg_idx = torch.multinomial(mask.float(), self.sample_size, replacement=True)
            # only sample limited number of negative samples
            neg_sample = x[neg_idx].view(self.sample_size, -1)
            neg_logits_i = torch.matmul(x[i], neg_sample.T).squeeze() / self.temperature
            neg_logits.append(neg_logits_i)
        neg_logits = torch.cat(neg_logits)
        logits = torch.cat([pos_logits, neg_logits])
        labels = torch.cat([torch.zeros(N), torch.ones(self.sample_size * N)])
        loss = self.loss(logits, labels.float().to(x.device))
        return loss

