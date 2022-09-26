2022.9.24 

Low light Raw2Video

思想：动态 + HDR Plus + Motion Deblur 实现 低光Raw域图像去噪

动态：FIFO-->self.bucket[idx:idx+self.bucket_size]

HDR Plus: HDR Plus Alignment + HDR Plus Merge 多帧融合去噪

综上，目前有了整体框架，FIFO + HDR Plus Alignment + HDR Plus Merge + ISP

**通过目前的结果看出，Fixed Pattern noise未去掉，这部分需要进行噪声估计，将其去掉**

**Motion Deblur部分还未做**

------

后续的工作

code：(基于目前Pipeline: FIFO + HDR Plus Alignment + HDR Plus Merge + ISP)

（1）去除motion blur: 可考虑使用维纳滤波对其进行去除

思路：利用视频帧图像图块，估计出运动轨迹的角度和长度，使用维纳滤波

Step 1: 估计出当前帧和bucket中其余参考帧的运动（目前代码中的HDR Plus Alignment中计算出的运动）

Step 2: 根据估计的运动计算相邻帧之间的运动角度 (e.g: 0->1, 1->2,……)，计算相应的PSF

Step 3: 使用维纳滤波对bucket中的所有图像进行运动模糊去除 (图像是灰度图)

------

（2）噪声估计

**Step 1: 噪声模型** 

Photon noise + dark noise + read noise + ADC noise

Photon noise, dark noise 满足泊松分布 L = Photon noise + dark noise

read noise, ADC noise 加性噪声 满足均值为0的高斯分布

L经过放大器后 G = L · g + Read noise · g

经过ADC后 I = G + ADC noise

I的方差和均值可表达为如下：

均值：E(I)=g⋅E(L)=g⋅t⋅(α⋅Φ+D)

方差：σ(I)2=g2 ⋅ t⋅ (α⋅ Φ + D) + σ(Add noise)2

**Step 2: 噪声估计**

需要估计的参数：暗电流D, 放大系数gain, read noise, ADC noise

暗电流估计：把相机镜头盖住，拍摄多帧图像，进行平均。得到的均值可近似为暗电流大小 (需要考虑不同的gain值影响)

去掉暗电流之后的均值：E(I)=g⋅t⋅(α⋅Φ)

去掉暗电流之后的方差：σ(I) 2 =g⋅E(I)+σ(Add noise)2

信号和方差满足线性关系，放大倍数：g，截距：加性噪声

g值和加性噪声的估计：拍摄多帧灰度图像，计算每个像素点的均值和方差，通过线性拟合来估计

以上来源：https://blog.csdn.net/matrix_space/article/details/105745560?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522166417251916782388025428%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=166417251916782388025428&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_ecpm_v1~pc_rank_v39-2-105745560-null-null.142%5Ev50%5Epc_rank_34_2,201%5Ev3%5Econtrol_1&utm_term=计算摄影：噪声模型&spm=1018.2226.3001.4187

------

dataset：

（1）用相机拍摄自己的数据集

（2）噪声估计

------

experiment：

（1）Starlight和自己数据集的图像结果

（2）L2SID数据集的定量结果：psnr和ssim

