
## 工作进程：

### 2023.2.4

+ 尝试了Meshflow的方法，测试了几个rgb视频，效果不错，这个代码可用。
+ Meshflow分为两步，把这两步的结果都做了一次输出。然后将raw图像应用到这个代码中，第一步中有些图像帧可以，有些不行，但是能够有显示正确的图像通过第二步平滑后后不能实现了。
    + 第一步的结果
|      frame6      |   frame7  |   frame8   | frame9 |
| :--: | --------- | --------- | ----------- |
|![](../../Docs/Images/Meshflow_result/stabilized_frame_6.png)|![](../../Docs/Images/Meshflow_result/stabilized_frame_7.png)|![](../../Docs/Images/Meshflow_result/stabilized_frame_8.png)|(![](../../Docs/Images/Meshflow_result/stabilized_frame_9.png)|
+ 继续调研了几个方法，PatchMatch: A Randomized Correspondence Algorithm for Structural Image Editing，The Generalized PatchMatch Correspondence Algorithm，Collaborative Filtering of Correlated Noise: Exact Transform-Domain Variance for Improved Shrinkage and Patch Matching，目前在理这些方法的代码


### 2023.1.3

+ pipeline

![](../..//Docs/Images/Pipeline0103.png)

+ 算法流程

首先Raw视频序列被输入，然后通过FIFO Bucket保持一定数量的图像帧，当一帧处理完，下一帧作为当前帧时，
又会从视频序列中补充一帧维持使用数量，从而形成一个动态的过程。接着通过运动估计来选择融合的图像帧，运动
估计部分使用handheld中的流估计方法，通过强度和梯度进行估计，再通过中心加权和GMM来预测出最小的运动，
以此用最小运动和可接受的模糊像素值计算出融合的帧数。随后利用帧间信息使用三维的去模糊方法去模糊。最后通
过对齐合并的方法去噪，经过ISP形成去噪后的视频序列

+ 实验结果：将融合帧数变成动态后，有改善效果，但在某些位置还存在匹配错误。因此，通过对齐方式找相似块进行匹配可能不准，需要改进。

### 12.11-17

+ 重新拍摄了灰阶卡的图片，依然分别在低照度和高照度下进行了噪声估计，得到RMS噪声，此值为标准差，即可推出方差。

    + 之前的公式建模出的噪声进行训练，测试图片结果是黑色的。shot noise符合泊松，它的参数通过低照度图像估计出来，fpn同样也符合泊松分布，它的参数是通过计算同一曝光时间下的30张图片的均值在计算标准差得出，read noise符合高斯分布，它的参数通过高照度图像估计出来的。现在改了噪声建模的生成代码和公式，对模型重新训练。
    
+  拍摄了一些连续的带有动作的图片，来进行测试
    
    + 问题：会存在合并错误的地方。**增加了对齐和合并阶段的图像数量，可以解决一些合并中的问题**

+ 校正白平衡

实验结果及其他方法的对比结果见[文件](https://github.com/qilinsun/UltralLowLightRawVideoISP/tree/main/Docs/Images/12.17结果)

------

### 12.6-10

+ vevid方法的复现

+ 调整预去噪网络

+ 根据handheld方法调整部分pipeline

#### 实验结果

+ pipeline之间的对比结果

|      HDR+      |   Denoising+HDR+  |   Deblur+HDR+   | Denoising+Deblur+HDR+ |
| :--: | --------- | --------- | ----------- |
|![](../../Docs/Images/1128对比结果/hdr+/减去黑电平_hdr+/srgb_part.png)|![](../../Docs/Images/1210results/denoising+hdr/srgb_part.png)|![](../../Docs/Images/1128对比结果/estrnn_deblur_hdr+/减去黑电平_pipelinev2/srgb_part.png)|![](../../Docs/Images/1210results/denoising+deblur+hdr/srgb_part.png)|

+ 加入vived的结果

|      Vevid+HDR+      |   Denoising+Vevid+HDR+  |   Vevid+Deblur+HDR+   | Denoising+Vevid+Deblur+HDR+ |
| :--: | --------- | --------- | ----------- |
|![](../../Docs/Images/1210results/hdr+vevid_simple_g0.6_b_0.5/srgb_part.png)|![](../../Docs/Images/1210results/denoising+vevid+hdr/srgb_part.png)|![](../../Docs/Images/1210results/vevid+pipelinev2/srgb_part.png)|![](../../Docs/Images/1210results/denoising+vevid+deblur+hdr/srgb_part.png)|

+ 对比试验

| HDR+ | Maskdngan | Starlight | Denoise+vevid+Deblur+HDR+ |
| :--: | --------- | --------- | ----------- |
|![](../../Docs/Images/1128对比结果/hdr+/减去黑电平_hdr+/srgb_part.png)|![](../../Docs/Images/1128对比结果/maskdngan_result/rawpy后处理结果/减去黑电平/frame3_denoised_sRGB.png)|![](../../Docs/Images/1128对比结果/hrnet_starlight/1129starlight原本结果/减去黑电平/denoise_raw.png)|![](../../Docs/Images/1210results/denoising+vevid+deblur+hdr/srgb_part.png)|

### 11.28-12.2

+ 补充了对比实验，算法分别为hdr+，maskdngan，starlight，结果如下

**这里的我们的算法，只增加了ESTRNN网络做deblur，预去噪的部分还未加入，等自己的噪声模型数据训练好之后再加入，并且以下减去了黑电平**

| HDR+ | Maskdngan | Starlight | Deblur+HDR+ |
| :--: | --------- | --------- | ----------- |
|![](../../Docs/Images/1128对比结果/hdr+/减去黑电平_hdr+/srgb_part.png)|![](../../Docs/Images/1128对比结果/maskdngan_result/rawpy后处理结果/减去黑电平/frame3_denoised_sRGB.png)|![](../../Docs/Images/1128对比结果/hrnet_starlight/1129starlight原本结果/减去黑电平/denoise_raw.png)|![](../../Docs/Images/1128对比结果/estrnn_deblur_hdr+/减去黑电平_pipelinev2/srgb_part.png)|

+ 在ISO 2500, F2.8, 曝光时间1/30的条件下，拍摄了两组灰阶卡照片，一组是低光照，一组是高光照。因为imatest软件的文档说，一般情况下是符合高斯分布，但是在低光情况下是更符合泊松分布，因
此拍摄两组照片，通过分析得到高斯和泊松分布的参数，分析过程和结果见[文件](https://github.com/qilinsun/UltralLowLightRawVideoISP/blob/main/Docs/Images/1128%E5%99%AA%E5%A3%B0%E5%BB%BA%E6%A8%A1%E7%BB%93%E6%9E%9C/noise.pdf)，低光照下的估计噪声为0.0204，高照度下的估计噪声为0.02295

+ 研究Fixed pattern noise的分析

    + imatest软件没具体介绍fpn的估计，参考了网上的代码对fpn进行估计，[code](https://github.com/qilinsun/UltralLowLightRawVideoISP/blob/main/noise_model/Dark_fpn.py)，估计的噪声为9.936909

+ 根据上面分析出的噪声，根据噪声模型𝒙_𝒑~ 𝝈_𝒔^𝟐 𝓟(𝒚_𝒑/𝝈_𝒔^𝟐) + 𝓟(𝑵_𝑭𝑷𝑵) + 𝓝(𝟎,𝝈_𝒓^𝟐)，[code](https://github.com/qilinsun/UltralLowLightRawVideoISP/blob/main/noise_model/generate_noise.py)，得出噪声图放入到神经网络中进行训练，结果没那么的好，有部分未恢复出来，且在整个pipeline代码运行时，部分patch里会有0，因此会在空间去噪时有问题，会出现除以0的情况，在这加了个1e-6，目前结果如下![](../../Docs/Images/1210results/denoising+deblur+hdr/srgb_part.png)

+ Raw video的处理，ffmpeg方法抽帧不能输出成raw格式，只能输出rgb格式

#### 后续工作

+ 可能需要调整一下目前的噪声模型

+ 将复现的vevid方法放进pipeline中提升一下信息，目前在调整该方法的代码

+ raw视频的处理

### 2022.11.26组会

+ 跟现有的sota算法在自己的数据集上做对比

+ 噪声模型需要完善，重新拍图不做去马赛克处理

+ 对denoise的图重新检查

+ 需要做白平衡

### 2022.11.21-25

+ 调整了deblur和side window filtering的位置——pipeline v5，结果如下

![](../../Docs/Images/221120结果/deblur_side_pipeline/3.png)

+ 与之前的pipeline v4结果对比

| Method | Pipeline v4 | Pipeline v5 |
| :----: | :---------: | :---------: |
| Result       | ![](../../Docs/Images/221120结果/Pipeline_side_deblur(15)/3.png)           |![](../../Docs/Images/221120结果/deblur_side_pipeline/3.png)             |

+ 按照sony的阵列进行调整，阵列如下：

![](../../Docs/Images/sony-quad-bayer.jpg)

**截取了部分图像进行测试，且第一行和第一列截取掉，截取的shape[1:1425, 1:2129],结果还是呈粉色，这个结果的颜色是不是和参数的设置相关，使用rawpy读取的raw图像，pattern显示是RGBG**

+ 将之前的去噪网络，换了数据集之后进行重新训练，现在的数据是SID的GT和GT+fixed pattern noise，然后在starlight数据集上在测试

#### 自己拍摄的数据集调试的结果

+ 之前颜色的问题应该是在raw2rgb时一些过程参数的设置不对，然后更换了比较简单的处理方式进行尝试,更换方法如下

```python
# raw2rgb
self.rawpyParam1 = {
    'demosaic_algorithm': rawpy.DemosaicAlgorithm.AHD,  # used in HDR+ supplement
    'half_size': False,
    'use_camera_wb': True,
    'use_auto_wb': False,
    'no_auto_bright': True,
    'output_color': rawpy.ColorSpace.sRGB,
    'output_bps': 16}
# hdr+ postprocess
pre_raw = self.load_video_raw('/media/cuhksz-aci-03/数据/CUHK_SZ/',seqID='8')
pre_raw.raw_image_visible[0:1420, 0:2120] = mergedImage[:]
post_image = pre_raw.postprocess(**self.rawpyParam1)

# To HSV and local tone mapping
cfa = (post_image / 65535. * 255.).astype(np.float32)# scale and set type for cv2
hsv = cv2.cvtColor(cfa, cv2.COLOR_RGB2HSV) 
hsvOperator = AutoGammaCorrection(hsv)
enhanceV = hsvOperator.execute()
hsv[...,-1] = enhanceV*255.
enhanceRGB = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
```
+ 结果如下，但是转换出的rgb图像偏绿色，local tone mapping的结果也是偏绿色

**用的deblur网络是9个block，参数1.82M**

**所有结果见[文件](https://github.com/qilinsun/UltralLowLightRawVideoISP/tree/main/Docs/Images/1124%E7%BB%93%E6%9E%9Csrgb)**

| Method |Original|Pipeline v1|Pipeline v2|Pipeline v3|Pipeline v4|
| :----: | -------------------------------------------------------- | :------------------------------------------------------: | :------------------------------------------------------: | :------------------------------------------------------: | :------------------------------------------------------: |
| Result | ![](../../Docs/Images/1124结果srgb/seq8_ori/srgb.png) | ![](../../Docs/Images/1124结果srgb/seq8/srgb_part.png) | ![](../../Docs/Images/1124结果srgb/seq8/pipeline2param1/srgb_part.png) | ![](../../Docs/Images/1124结果srgb/seq8/hdr+denoise/srgb_part.png) | ![](../../Docs/Images/1124结果srgb/seq8/pipelinev4param/srgb_part.png) |

+ 建立噪声模型：𝒙_𝒑~ 𝝈_𝒔^𝟐 𝓟(𝒚_𝒑/𝝈_𝒔^𝟐) + 𝓟(𝑵_𝑭𝑷𝑵) + 𝓝(𝟎,𝝈_𝒓^𝟐)

#### 问题

+ 使用预去噪的方法会改变颜色，出现颜色上的偏差

+ 目前的噪声模型建立的是否正确

+ 在低iso拍摄的数据中，效果不好，且预去噪和debur都会对图像的颜色有明显影响

------

### 2022.11.19组会

#### 目前的pipeline问题：细节信息部分提升了，但背景信息丢失了

#### 预处理使用的降噪网络

+ 使用unet网络进行预处理降噪，效果一般，可以使用19年cvpr的side window filtering方法进行实验

#### Starlight数据集结果

+ 由目前增加了denoising网络的结果看，颜色变得过多，颜色不正确，检查是否弄反了通道，且背景信息丢失严重，背景中的星星没有恢复出来

   + **解决方法**：针对颜色问题，在Raw图像放入到网络前，做均衡化；背景信息丢失，可以考虑使用黑电平的信息

#### 自己拍摄的数据结果

+ 在pipeline的测试当中，整体图像颜色呈现粉色

   + **解决方法**：索尼A7相机的bayer排布可能不是一样的，需要进行调整

------

### 2022.11.13-11.20

+ 使用DRV数据集和自己拍摄的数据集验证了一下目前的框架

   + **问题**：颜色错误，整体显示粉色；DRV和自己拍摄的数据应该都是静止的图片，没有模糊情况存在，实验出的效果不好

   + **Raw video的采集**: 需要一个外录屏采集Raw video，直接拍摄出的视频是mp4格式

+ 在D65光源的条件下，重新拍摄了灰阶图，并根据目前的一些文献，学习噪声标定

   + 泊松分布的噪声: fixed pattern noise + shot noise

   + 高斯分布的噪声: Read noise

+ 训练调试深度降噪网络作为一个预处理

   + 使用的数据集：SID短曝光图像 + Statlight的fixed pattern noise和作为gt的SID长曝光图像，其中fixed pattern noise裁减成512 * 512大小的

   + **目前的问题**：由以上数据训练出的unet网络效果一般，后面对此网络的训练使用的数据是否只用在gt图像上加入fixed pattern noise，从而达到去除目前starlight数据集中fixed 
                   pattern noise的目的

   + 统计了Deblur网络使用不同block的参数量，并训练了block为9的网络，同时放入到了pipelien中进行实验
   
 + side window filtering方法的实验

### 2022.11.13-11.20 代码和对应测试结果

#### 代码已经上传到github中

+ 预处理的降噪网络和训练代码，[Code](https://github.com/qilinsun/UltralLowLightRawVideoISP/tree/main/deep_denoising)

+ 去模糊代码，[ESTRNN](https://github.com/qilinsun/UltralLowLightRawVideoISP/tree/main/model)和[参数文件](https://github.com/qilinsun/UltralLowLightRawVideoISP/tree/main/para)

+ Side window filtering [python code](https://github.com/qilinsun/UltralLowLightRawVideoISP/blob/main/Utility/SideWindowFilter.py)
   
#### 实验结果，所有结果在[github](https://github.com/qilinsun/UltralLowLightRawVideoISP/tree/main/Docs/Images/221120结果)
#### 以下实验结果使用的deblur网络均是15个block的，参数量为2.82M

+ Starlight数据集：Starlight, Pipeline v1(hdr+), Pipleine v2(deblur+hdr+), Pipeline v3(deep denoiser+deblur+hdr+), Pipeline v4(side+deblur+hdr+)

| Method | Starlight                                                |                       Pipeline v1                        |                       Pipeline v2                        |                       Pipeline v3                        |                       Pipeline v4                        |
| :----: | -------------------------------------------------------- | :------------------------------------------------------: | :------------------------------------------------------: | :------------------------------------------------------: | :------------------------------------------------------: |
| Result | ![](../../Docs/Images/221120结果/starlight/3.png) | ![](../../Docs/Images/221120结果/pipeline_crop/3.png) | ![](../../Docs/Images/221120结果/pipeline_deblur(15)/3.png) | ![](../../Docs/Images/221120结果/pipeline_deblur(15)_denoise(900)/3.png) | ![](../../Docs/Images/221120结果/Pipeline_side_deblur(15)/3.png) |

+ 自己拍摄的数据集结果

| Method | Pipeline v1 | Pipeline v3 |
| :----: | :---------: | :---------: |
| Result       | ![](../../Docs/Images/221120结果/pipeline_shoot/0.png)            |![](../../Docs/Images/221120结果/pipelin_deblur_denoise_shoot/0.png)             |

------

### 2022.11.12组会

+ Pipeline实现的目标：最小的计算量实现较好的效果

+ 去模糊网络和深度降噪网络的优化，尽量使网络的参数少，模型小

+ 拍摄灰阶图做噪声标定

   + 将相机和光源都调成D65

   + 拍摄灰阶卡，使用imatest生成噪声模型，然后根据此噪声模型，将其建模出来（高斯噪声和泊松噪声，且泊松噪声包括fixed pattern noise）

+ 测试自己拍摄的图像

------

### 2022.11.6-11.12

+ 搭建简单的unet网络，选择并下载合适的数据集(SID和DRV)

+ 拍摄了不同色温下的灰阶图像，在imatest里进行噪声分析

   + 对imatest该软件里的部分参数要学习调试，目前了解度不够

+ 拍摄了不同iso，f参数下的低光数据集

------

### 2022.11.5组会

+ 目前处理的图像中存在fixed pattern noise

   + 可以使用三五层简单的神经网络将其去除

+ 拍摄低光图像

+ 做噪声标定

------

### 2022.10.31-11.6

+ 对上一周工作中出现的细节错误进行了修正，重新梳理了整个框架

+ 调整了Deblur网络的参数，训练了几个不同的模型，争取使deblur的效果最好

+ 学习imatest软件的使用

------

### 2022.10.29组会

+ 视频3D降噪方法学习，捋清思路，画图，整理好代码框架

+ 数据类型是float32，不能用uint8处理，会丢失信息

+ 维纳滤波要加TV约束，或者尝试ADMM+TV

#### 根据以上意见和本周的结果，继续调整deblur部分，提高图像的效果

------

### 2022.10.23-30 

### 整体思路，目前的pipeline(增加了神经网络去模糊)

**目标**：基于传统的视频去噪方法，实现一个动态且实时的视频去噪过程

**传统视频实时降噪方法**：使用快速的对齐算法，或者用运动检测代替运动估计，根据检测到的运动强度，对时域滤波和空域滤波的结果做加权平均。

**我们的基础框架**：利用HDR+方法中的对齐算法，通过金字塔方法计算对齐位移，根据位移得出运动向量，再根据运动向量找出相似块。然后根据HDR+的合并方法，利用这些像素块实现时域和空域上的降噪。

**实现难点**：

   + 对齐：会受到环境光变化，噪声，运动目标以及长曝光引起的模糊等影响。
  
   + 融合：需要一个精准的噪声建模方法估计噪声强度应用到多帧融合去噪的过程中。

**初步完善的具体方法**：

   + 先进行噪声标定，根据估计出的噪声先进行预处理

   + 利用bucket实现一个动态的处理过程

   + 利用视频图像的帧间信息去除模糊（使用ESTRNN(神经网络)对视频图像去模糊）

   + 基础的降噪方法基于HDR+的多帧融合进行去噪 (金字塔对齐，然后根据对齐的结果进行融合去噪)

      + 对齐：利用金字塔算法，由粗到精计算出alignment，根据alignment估计出运动向量，找到相似的像素块

      + 合并：利用相似的像素块和噪声方差应用到多帧融合中进行时域和空域上的去噪。
     
   + 目前的框架图
   
   ![](../../Docs/Images/Pipeline1101.png)
   
+ 根据以上pipeline图(噪声建模和预处理未加入)得到的结果如下(输入已更改，不是int8),输入16帧由于有些图像并没有经过去模糊处理的则舍弃，在对齐和合并阶段使用的帧数是1个当前帧和11个参考帧，共12帧

![](../../Docs/Images/20221030结果/with_deblur/3.png)

+ starlight的算法，无运动模糊处理的算法，上述pipeline算法对比结果如下

![](../../Docs/Images/20221030结果/三个方法对比结果.png)

具体结果见(https://github.com/qilinsun/UltralLowLightRawVideoISP/tree/main/Docs/Images/20221030结果)

------

### 2022.10.22组会

#### 组会意见总结
##### 整体思路想清楚可以重新画一下图，指导后续工作进行

+ 传统的3D方法

+ 增加图像的数量，可以选取H264用的16帧或19年论文中的13帧做
+ + Comment：也可以试试这个 https://github.com/codeslake/PVDNet

+ 使用深度学习方法（注意使用video的时序信息）

+ downsample：估计mv是可以，直接去模糊不需要

+ 在估计之前先做个图像预处理，简单的去噪方法即可

+ **关于overlap，alignment和dublur一起做** 
+ + Comment：可以参考一下这篇文章。不要想象的太复杂了，尽可能有个简单清晰的pipeline，如果本身运动太大估计不准，可以用金字塔的方式，从粗到精，然后拿粗一级的结果做初始化再到精一级。
+ + https://ieeexplore.ieee.org/document/7025361 

+ 了解2019ASIA 的运动算法

+ 了解视频去噪VBM4D的思想

#### 后续工作

+ 去模糊前需要简单的去噪预处理

+ 找传统的3D方法，尝试去模糊

+ 增加图像的数量，使用更多帧数估计运动，从而进行去模糊

+ 使用深度学习方法

+ 使用2019年ASIA的方法

+ Alignment和Deblur一起做

--------

### 2022.10.8-10.14 总结

+ 在先前的pipline的基础上，在merge前使用维纳滤波(+ TV) / fast deconvoltion，效果不明显

+ 看了相关噪声标定的方法：ELD，方差和均值拟合的方法

+ 修改了pipline图，整理代码框架(待上传)

![](../../Docs/Images/pipeline_221013.png)

+ 导入matlab里运动估计方法，得到MV：[code](https://github.com/qilinsun/UltralLowLightRawVideoISP/blob/main/Utility/matlab/motionEstDS.m)

### 2022.10.8-10.14 代码和对应测试结果

+ 使用H264方法和维纳滤波得到的最终结果,gamma=0.5(无overlap的)

![](../../Docs/starlight/Motion_deblur/9_newpipeline_gamma0.5/0.jpg)

+ 使用H264方法和维纳滤波的结合, gamma=0.15(overlap的)

![](../../Docs/starlight/Motion_deblur/9_deblur_overlap_gamma0.15/0.jpg)

------

### 2022.9.28——2022.10.7

**Step 1**: 目前的算法已经估计出了运动向量，根据向量计算长度和角度，

motionVector, (shape：[7, 63, 107, 2]) 根据此算出长度和角度，length和angle的shape均为[7, 63, 107]

代码如下：

```python
def motion_angle(motionvector):
  motionvector_y = motionvector[..., 0]
  motionvector_x = motionvector[..., 1]
  angle = np.arctan2(motionvector_y, motionvector_x)
  # angle = np.rad2deg(theta)
  
  return angle

def motion_vector_length(motionvector):
    n, h, w, _ = motionvector.shape
    length = np.zeros((n, h, w))
    angle = np.zeros((n, h, w))
    for i in range(len(motionvector)):
        if i == 0:
            mv = motionvector[i]
            mv_length = np.sqrt(mv[...,0]**2 + mv[...,1]**2)
            length[i, :, :] = mv_length
            mv_angle = motion_angle(mv)
            angle[i, :, :] = mv_angle
        else:
            cur_mv = motionvector[i]
            ref_mv = motionvector[i-1]
            diff_mv = cur_mv - ref_mv
            diff_mv_length = np.sqrt(diff_mv[...,0]**2 + diff_mv[...,1]**2)
            diff_mv_angle = motion_angle(diff_mv)
            length[i, :, :] = diff_mv_length
            angle[i, :, :] = diff_mv_angle

    return length, angle
```
**Step 2**: 根据计算出的每个角度和长度再生成模糊核

可计算出length和angle的组成63*107对

[code](https://github.com/qilinsun/UltralLowLightRawVideoISP/blob/main/Utility/PSF.py)

```python
import numpy as np


def get_motion_blur(length, angle, aligntiles):
    # 点扩散函数
    n, h, w, size1, size2 = aligntiles.shape
    PSF_sum = np.zeros((aligntiles.shape[0] - 1, aligntiles.shape[1],
                        aligntiles.shape[2], aligntiles.shape[3], aligntiles.shape[4]))
    PSF_aver = np.zeros((int(h * size1), int(w * size2)))
    PSF_vis = np.zeros((n-1, int(h * size1/2+size1/2), int(w * size2/2+size1/2)))
    for i in range(aligntiles.shape[0] - 1):

        x_center = (aligntiles[i].shape[2] - 1) / 2
        y_center = (aligntiles[i].shape[3] - 1) / 2

        motion_length = length[i]
        motion_angle = angle[i]

        sin_val = np.sin(motion_angle)
        cos_val = np.cos(motion_angle)

        # 计算每个tiles的psf 再reshape
        for j in range(motion_length.shape[0]):
            for n in range((motion_length.shape[1])):
                PSF = np.zeros((aligntiles.shape[3], aligntiles.shape[4]))
                if motion_length == 0:
                    # 该块处的PSF置为0
                    PSF_sum[i, j, n, ...] = PSF
                else:
                    for m in range(int(np.round(motion_length[j][n]))):
                        x_offset = np.round(sin_val[j, n] * m)
                        y_offset = np.round(cos_val[j, n] * m)
                        x_1 = x_center - x_offset
                        y_1 = y_center + y_offset
                        if 0 <= x_1 < (aligntiles.shape[3]) and 0 <= y_1 < (aligntiles.shape[4]):
                            x = x_1
                            y = y_1
                        else:
                            x = x_center
                            y = y_center

                        PSF[int(x), int(y)] = 1

                    # 这部分有些算出得0是因为没有位移，length为0，所以加了判断
                    PSF = PSF / PSF.sum()
                    PSF_sum[i, j, n, ...] = PSF

                    PSF_aver[(int(size1/2) * j):(int(size1/2) * j + size1),\
                    (int(size2/2)* n):(int(size2/2) * n + size2)] = PSF
        
        PSF_vis[i, ...] = PSF_aver
    # 可视化   
    for v in range(PSF_vis.shape[0]):
        psf_path = "/home/cuhksz-aci-03/Documents/UltralLowLightRawVideoISP-main/psf_result/" + str(v) + '.png'
        cv2.imwrite(psf_path, PSF_vis[v])

    return PSF_sum
```
+ 根据MV计算出的PSF结果(该结果乘了255之后的显示结果)

![PSF结果](../../Docs/starlight/PSF/PSF_221007/0.png) 

+ HDR+ MV 

![运动矢量](../../Docs/starlight/align_mismatch/hdrplus_mv/vector_0.png)

**Step 3**: 普通维纳滤波

代码如下：

```python
 # 维纳滤波，K=0.01
def wiener(input, PSF, eps, K=0.01):       
    input_fft = fft.fft2(input)
    PSF_fft = fft.fft2(PSF) + eps
    PSF_fft_1 = np.conj(PSF_fft) / (np.abs(PSF_fft) ** 2 + K)
    result = np.fft.ifft2(input_fft * PSF_fft_1)
    result = np.abs(np.fft.fftshift(result))
    return result
```
**普通的维纳滤波得不出结果**，改为deep optics中的方法，[code(tf-->pytorch)](https://github.com/qilinsun/UltralLowLightRawVideoISP/blob/main/Utility/wiener.py)

不同gamma值的结果(随便选的), 以下这些图片和相应的视频[Click](https://github.com/qilinsun/UltralLowLightRawVideoISP/tree/main/Docs/starlight)

+ code: 在原代码的HDR+ pipelin中增加了deblur部分

```python
# Hdrplus pipeline
        motionVectors, alignedTiles = alignHdrplus(referenceImg,alternateImgs,self.mbSize)
        
        # Deblur
        motion_length, motion_angle = motion_vector_length(motionVectors)
        PSF = get_motion_blur(motion_length, motion_angle, alignedTiles)

        deb_img = np.zeros(alignedTiles.shape)
        deb_img[-1, ...] = alignedTiles[-1, ...]
        for j in range(PSF.shape[0]):
            for m in range(PSF.shape[1]):
                for n in range(PSF.shape[2]):
                    blurred = repeat(alignedTiles[j, m, n, ...], 'h w -> b h w c', b=1, c=1) 
                    PSF_trans = repeat(PSF[j, m, n, ...], 'h w -> h w c d', c=1, d=1)
                    # transfer tensor
                    blurred = torch.tensor(blurred)
                    PSF_trans = torch.tensor(PSF_trans)
                    deblur_img = inverse_filter(blurred, blurred, PSF_trans, init_gamma=1.5)
                    deblur_img = deblur_img.squeeze()
                    deblur_img = deblur_img.detach().numpy()

                    deb_img[j, m, n, ...] = deblur_img

        alignedTiles = deb_img
        
        mergedImage = mergeHdrplus(referenceImg, alignedTiles, self.padding, 
                                   self.lambdaS, self.lambdaR, self.params, self.options)
        mergedImage = np.clip(mergedImage,0,self.whiteLevel)
        
        # ISP process
        ISP_output = starlightFastISP(mergedImage)
```

+ deblur之前的结果_17Seq0

![deblur之前的结果_17Seq0](../../Docs/starlight/ori_img/0.jpg)

+ gamma=0.5_17Seq0

![gamma=0.5_17Seq0](../../Docs/starlight/Motion_deblur/gamma0.5/0.jpg)

+ gamma=1.5_17Seq0

![gamma=1.5_17Seq0](../../Docs/starlight/Motion_deblur/gamma1.5/0.jpg)

+ gamma=5_17Seq0

![gamma=5_17Seq0](../../Docs/starlight/Motion_deblur/gamma5/0.jpg)


### 目前的问题

+ HDR+ 的方法做运动估计可能不行，需要换运动估计方法, 换了ARPS方法(这部分代码之前的同学已经转换完，自己对照C++检查学习)，[code(C++-->Python)](https://github.com/qilinsun/UltralLowLightRawVideoISP/blob/main/Utility/utils.py)

### Todo

+ 将ARPS放入到Pipeline中，得到MV，再看motion deblur的效果
