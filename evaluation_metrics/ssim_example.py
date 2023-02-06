import pytorch_ssim
import torch
from torch.autograd import Variable
import numpy as np

l = []
for i in range(5):

    pred = np.load('./test_data/control_group/predicted_shape%02d.npz'%i)
    true = np.load('./test_data/control_group/real_shape%02d.npz'%i).astype(np.float32)

    pred = np.swapaxes(pred, 1, 4)
    true = np.swapaxes(true, 1, 4)

    img1 = torch.from_numpy(pred)
    img2 = torch.from_numpy(true)

    # img1 = Variable(torch.rand(1, 1, 128, 128, 128))
    # img2 = Variable(torch.rand(1, 1, 128, 128, 128))

    if torch.cuda.is_available():
        img1 = img1.cuda()
        img2 = img2.cuda()

    result = pytorch_ssim.ssim3D(img1, img2)
    l.append(result)
    print(result)
print(sum(l)/len(l))

# ssim_loss = pytorch_ssim.SSIM3D(window_size = 11)
#
# print(ssim_loss(img1, img2))