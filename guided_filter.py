import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from .box_filter import BoxFilter

from PIL import Image
import cv2
import numpy as np


class sFastGuidedFilter(nn.Module):
    def __init__(self, radius=1, eps=2):
        super(sFastGuidedFilter, self).__init__()

        self.r = radius
        self.eps = eps

        self.pad = nn.ReplicationPad2d(radius)
        self.box  = nn.Conv2d(3, 3, kernel_size=2*radius+1, padding=0, dilation=1, bias=False, groups=3)
        self.box.weight.data[...] = 1.0/( 2*radius + 1)**2
        
        self.boxfilter = nn.Sequential(
            self.pad,
            self.box,
        )
        # self.padding = nn.ReplicationPad2d()


    def forward(self, lr_x, lr_y, hr_x):
        n_lrx, c_lrx, h_lrx, w_lrx = lr_x.size()
        n_lry, c_lry, h_lry, w_lry = lr_y.size()
        n_hrx, c_hrx, h_hrx, w_hrx = hr_x.size()

        assert n_lrx == n_lry and n_lry == n_hrx
        assert c_lrx == c_hrx and (c_lrx == 1 or c_lrx == c_lry)
        assert h_lrx == h_lry and w_lrx == w_lry
        assert h_lrx > 2*self.r+1 and w_lrx > 2*self.r+1

        ## mean_x
        mean_x = self.boxfilter(lr_x) 
        ## mean_y
        mean_y = self.boxfilter(lr_y) 
        ## cov_xy
        cov_xy = self.boxfilter(lr_x * lr_y) - mean_x * mean_y
        ## var_x
        var_x = self.boxfilter(lr_x * lr_x) - mean_x * mean_x

        ## A
        A = cov_xy / (var_x + self.eps)
        ## b
        b = mean_y - A * mean_x
        
        # mean_A; mean_b
        # A = self.boxfilter(A) 
        # b = self.boxfilter(b) 

        A = F.interpolate(A, (h_hrx, w_hrx), mode='bilinear', align_corners=True)
        b = F.interpolate(b, (h_hrx, w_hrx), mode='bilinear', align_corners=True)
        # print("A.shpae = ", A.size())
        b_,ch_,h, w = A.size()
        d = torch.zeros(1, 8, h, w, dtype=torch.float32)
        
        # self.r = self.r * 2 ###########################################################################################
        filter = torch.ones(1, 1, 2*self.r+1, 2*self.r+1).cuda()
        L, R, U, D = [filter.clone() for _ in range(4)]
        L[:, :, :, self.r + 1:] = 0
        R[:, :, :, 0: self.r] = 0
        U[:, :, self.r + 1:, :] = 0
        D[:, :, 0: self.r, :] = 0

        NW, NE, SW, SE = U.clone(), U.clone(), D.clone(), D.clone()

        L, R, U, D = L / ((self.r + 1) * (self.r*2 +1)), R / ((self.r + 1) * (self.r*2 +1)), \
                    U / ((self.r + 1) * (self.r*2 +1)), D / ((self.r + 1) * (self.r*2 +1))

        NW[:, :, :, self.r + 1:] = 0
        NE[:, :, :, 0: self.r] = 0
        SW[:, :, :, self.r + 1:] = 0
        SE[:, :, :, 0: self.r] = 0

        NW, NE, SW, SE = NW / ((self.r + 1) ** 2), NE / ((self.r + 1) ** 2), \
                        SW / ((self.r + 1) ** 2), SE / ((self.r + 1) ** 2)

        res = hr_x.clone()
        # print(res.size())
        for ch in range(res.size()[1]):
            A_ = A[:,ch].view(1,1,h,w)
            A_ = self.pad(A_)
            

            b_ = b[:,ch].view(1,1,h,w)
            b_ = self.pad(b_)
            

            im_ch = res[0, ch, ::].clone().view(1, 1, h, w)

            d[:, 0, ::] = F.conv2d(input=A_, weight=L, padding=(0, 0)) * im_ch + F.conv2d(input=b_, weight=L, padding=(0, 0)) - im_ch
            d[:, 1, ::] = F.conv2d(input=A_, weight=R, padding=(0, 0)) * im_ch + F.conv2d(input=b_, weight=R, padding=(0, 0)) - im_ch
            d[:, 2, ::] = F.conv2d(input=A_, weight=U, padding=(0, 0)) * im_ch + F.conv2d(input=b_, weight=U, padding=(0, 0)) - im_ch
            d[:, 3, ::] = F.conv2d(input=A_, weight=D, padding=(0, 0)) * im_ch + F.conv2d(input=b_, weight=D, padding=(0, 0)) - im_ch
            d[:, 4, ::] = F.conv2d(input=A_, weight=NW, padding=(0, 0)) * im_ch + F.conv2d(input=b_, weight=NW, padding=(0, 0)) - im_ch
            d[:, 5, ::] = F.conv2d(input=A_, weight=NE, padding=(0, 0)) * im_ch + F.conv2d(input=b_, weight=NE, padding=(0, 0)) - im_ch
            d[:, 6, ::] = F.conv2d(input=A_, weight=SW, padding=(0, 0)) * im_ch + F.conv2d(input=b_, weight=SW, padding=(0, 0)) - im_ch
            d[:, 7, ::] = F.conv2d(input=A_, weight=SE, padding=(0, 0)) * im_ch + F.conv2d(input=b_, weight=SE, padding=(0, 0)) - im_ch

            d_abs = torch.abs(d)
            #print('im_ch', im_ch)
            #print('dm = ', d_abs.shape, d_abs)
            mask_min = torch.argmin(d_abs, dim=1, keepdim=True)
            #print('mask min = ', mask_min.shape, mask_min)
            dm = torch.gather(input=d, dim=1, index=mask_min).cuda()
            im_ch = dm + im_ch

            res[:, ch, ::] = im_ch
        res = res.int()
        res = torch.clamp(res,0,255)
        res = res.float()
        return res


        
class FastGuidedFilter(nn.Module):
    def __init__(self, r, eps=1e-8):
        super(FastGuidedFilter, self).__init__()

        self.r = r
        self.eps = eps
        self.boxfilter = BoxFilter(r)


    def forward(self, lr_x, lr_y, hr_x):
        n_lrx, c_lrx, h_lrx, w_lrx = lr_x.size()
        n_lry, c_lry, h_lry, w_lry = lr_y.size()
        n_hrx, c_hrx, h_hrx, w_hrx = hr_x.size()

        assert n_lrx == n_lry and n_lry == n_hrx
        assert c_lrx == c_hrx and (c_lrx == 1 or c_lrx == c_lry)
        assert h_lrx == h_lry and w_lrx == w_lry
        assert h_lrx > 2*self.r+1 and w_lrx > 2*self.r+1

        ## N
        N = self.boxfilter(Variable(lr_x.data.new().resize_((1, 1, h_lrx, w_lrx)).fill_(1.0)))

        ## mean_x
        mean_x = self.boxfilter(lr_x) / N
        ## mean_y
        mean_y = self.boxfilter(lr_y) / N
        ## cov_xy
        cov_xy = self.boxfilter(lr_x * lr_y) / N - mean_x * mean_y
        ## var_x
        var_x = self.boxfilter(lr_x * lr_x) / N - mean_x * mean_x

        ## A
        A = cov_xy / (var_x + self.eps)
        ## b
        b = mean_y - A * mean_x

        ## mean_A; mean_b
        mean_A = F.interpolate(A, (h_hrx, w_hrx), mode='bilinear', align_corners=True)
        mean_b = F.interpolate(b, (h_hrx, w_hrx), mode='bilinear', align_corners=True)

        return mean_A*hr_x+mean_b


class GuidedFilter(nn.Module):
    def __init__(self, r, eps=1e-8):
        super(GuidedFilter, self).__init__()

        self.r = r
        self.eps = eps
        self.boxfilter = BoxFilter(r)


    def forward(self, x, y):
        n_x, c_x, h_x, w_x = x.size()
        n_y, c_y, h_y, w_y = y.size()

        assert n_x == n_y
        assert c_x == 1 or c_x == c_y
        assert h_x == h_y and w_x == w_y
        assert h_x > 2 * self.r + 1 and w_x > 2 * self.r + 1

        # N
        N = self.boxfilter(Variable(x.data.new().resize_((1, 1, h_x, w_x)).fill_(1.0)))

        # mean_x
        mean_x = self.boxfilter(x) / N
        # mean_y
        mean_y = self.boxfilter(y) / N
        # cov_xy
        cov_xy = self.boxfilter(x * y) / N - mean_x * mean_y
        # var_x
        var_x = self.boxfilter(x * x) / N - mean_x * mean_x

        # A
        A = cov_xy / (var_x + self.eps)
        # b
        b = mean_y - A * mean_x

        # mean_A; mean_b
        mean_A = self.boxfilter(A) / N
        mean_b = self.boxfilter(b) / N

        return mean_A * x + mean_b

class ConvGuidedFilter(nn.Module):
    def __init__(self, radius=1, norm=nn.BatchNorm2d):
        super(ConvGuidedFilter, self).__init__()

        self.box_filter = nn.Conv2d(3, 3, kernel_size=3, padding=radius, dilation=radius, bias=False, groups=3)
        self.conv_a = nn.Sequential(nn.Conv2d(6, 32, kernel_size=1, bias=False),
                                    norm(32),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(32, 32, kernel_size=1, bias=False),
                                    norm(32),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(32, 3, kernel_size=1, bias=False))
        self.box_filter.weight.data[...] = 1.0

    def forward(self, x_lr, y_lr, x_hr):
        _, _, h_lrx, w_lrx = x_lr.size()
        _, _, h_hrx, w_hrx = x_hr.size()

        N = self.box_filter(x_lr.data.new().resize_((1, 3, h_lrx, w_lrx)).fill_(1.0))
        ## mean_x
        mean_x = self.box_filter(x_lr)/N
        ## mean_y
        mean_y = self.box_filter(y_lr)/N
        ## cov_xy
        cov_xy = self.box_filter(x_lr * y_lr)/N - mean_x * mean_y
        ## var_x
        var_x  = self.box_filter(x_lr * x_lr)/N - mean_x * mean_x

        ## A
        A = self.conv_a(torch.cat([cov_xy, var_x], dim=1))
        ## b
        b = mean_y - A * mean_x

        ## mean_A; mean_b
        mean_A = F.interpolate(A, (h_hrx, w_hrx), mode='bilinear', align_corners=True)
        mean_b = F.interpolate(b, (h_hrx, w_hrx), mode='bilinear', align_corners=True)

        return mean_A * x_hr + mean_b
