---
layout: post
title: "Transformer-Attention is all you need"
date: 2025-06-23 12:00:00 +0900
categories: jekyll githubpages
---

드디어 CellPLM/layer/transformer.py를 공부한다. 모르는 함수가 많아서 머리가 너무 아프다^^
CellPLM의 Transformer에는 Spectral normalization을 사용한다는데 이게 뭔지 모르겠다.

✅ 2. Spectral Normalization이 뭐고 왜 중요해?
🎯 정의:
Weight 행렬의 spectral norm (σ) = 가장 큰 singular value (특이값)
즉, linear transformation이 얼마나 "출력을 키우는지"를 나타냄.

🎯 목적:
네트워크의 출력 폭발/수축 방지
Lipschitz 상수 제한 → gradient 폭발/소실 방지
특히 GANs에서 안정된 학습을 위해 매우 중요.

🎯 작동 방식:
Spectral Normalization은 아래처럼 weight를 조정해요:

🎯 왜 중요하냐?
일반적인 weight norm은 L2 전체 norm만 조절하지만,

Spectral norm은 단일 방향에서의 최댓값만 제어해요.

이게 더 보수적이고 강력한 방식임.

{% highlight ruby %} self.register_buffer('u', nn.functional.normalize(torch.randn(in_features), dim=0)) {% endhighlight %}

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

