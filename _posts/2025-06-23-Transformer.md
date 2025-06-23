---
layout: post
title: "Transformer-Attention is all you need"
date: 2025-06-23 12:00:00 +0900
categories: jekyll githubpages
---

{% highlight ruby %} 
class Linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, device='cpu', dtype=None):
        super().__init__(in_features, out_features, bias=bias)
        self.register_buffer('u', nn.functional.normalize(torch.randn(in_features), dim=0))
        with torch.no_grad():
            sigma = self.get_sigma()
        self.register_buffer('spectral_norm', sigma)

        self.sigma = nn.Parameter(torch.ones(1))
        self.to(device)

    def get_sigma(self):
        with torch.no_grad():
            u = self.u
            v = self.weight.mv(u)
            v = nn.functional.normalize(v, dim=0)
            u = self.weight.T.mv(v)
            u = nn.functional.normalize(u, dim=0)
            self.u.data.copy_(u)
        return torch.einsum('c,cd,d->', v, self.weight, u)

    def get_weight(self):
        sigma = self.get_sigma()
        if self.training:
            self.spectral_norm.data.copy_(sigma)
        weight = (self.sigma / sigma) * self.weight
        return weight

    def forward(self, x):
        return nn.functional.linear(x, self.get_weight(), self.bias) 
{% endhighlight %}

