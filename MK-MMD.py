from dalib.modules.kernels import GaussianKernel
import torch
import dalib.adaptation.dan
feature_dim = 1024
batch_size = 2
kernels = (GaussianKernel(alpha=0.5), GaussianKernel(alpha=1.), GaussianKernel(alpha=2.))
loss = dalib.adaptation.dan.MultipleKernelMaximumMeanDiscrepancy(kernels)
 # features from source domain and target domain
z_s, z_t = torch.randn(batch_size, feature_dim), torch.randn(batch_size, feature_dim)
output = loss(z_s, z_t)
print(output)