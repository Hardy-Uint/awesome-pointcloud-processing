# awesome-pointcloud-processing
awesome pointcloud processing algorithm


> 公众号：[3D视觉工坊](https://mp.weixin.qq.com/s?__biz=MzU1MjY4MTA1MQ==&mid=2247484684&idx=1&sn=e812540aee03a4fc54e44d5555ccb843&chksm=fbff2e38cc88a72e180f0f6b0f7b906dd616e7d71fffb9205d529f1238e8ef0f0c5554c27dd7&token=691734513&lang=zh_CN#rd)
>
> 主要关注：3D视觉算法、SLAM、vSLAM、计算机视觉、深度学习、自动驾驶、图像处理以及技术干货分享
>
> 运营者和嘉宾介绍：运营者来自国内一线大厂的算法工程师，深研3D视觉、vSLAM、计算机视觉、点云处理、深度学习、自动驾驶、图像处理、三维重建等领域，特邀嘉宾包括国内外知名高校的博士硕士，旷视、商汤、百度、阿里等就职的算法大佬，欢迎一起交流学习

## 点云标注工具

### 开源

1. [Semantic.editor](https://github.com/MR-520DAI/semantic-segmentation-editor)

### 商用

> 商用软件很多，阿里、腾讯、百度、京东都有对应业务

## 点云获取

> 传统的点云获取技术包括非接触式测量和接触式测量两种，它们的主要区别在于，在测量过程中测头是否与工件的表面相接触。
>
> 非接触式测量是利用光学原理的方法采集数据，例如结构光法、测距法以及干涉法等。该方法的优点在于测量速度较快、测量精度高，并且能够获得高密度点云数据，但其测量精度易受外界因素干扰，而且测量物
> 体表面的反射光与环境光对测量精度也有一定影响。
>
> 相反，接触式测量是通过将测头上的测量传感器与被测物体的外表面相接触，然后通过移动测头来读取物体表面点的三维坐标值。该方法的优点在于测头的结构相对固定，并且其测量结果不受被测物体表面的材料与表面特性等因素的影响。这种方法的不足在于，由于测头长期与被测物体表面相接触，易产生磨损，并且这种测量方式的测量速度较慢，不适合测量几何结构较复杂的物体。

## 点云应用场景

> 逆向工程、游戏人物重建、文物保护、数字博物馆、医疗辅助、三维城市建模

## 点云种类

> 不同的点云获取技术获取的点云数据类型不同，根据点云数据中点的分布情况可将点云数据划分为以下四种类型

#### 散乱点云

散乱点云是指所有数据点在空间中以散乱状态分布，任意两点之间没有建立拓扑连接关系。一般而言，激光点测量系统获得的点云数据以及坐标测量机在随机扫描状态下获得的点云数据都为散乱点云数据。

#### 扫描线点云

测量设备所获得的三维点云数据是由多条直线或曲线构成，点与点之间有一定的拓扑连接关系。一般而言，这种点云数据类型常见于扫描式点云数据中。

#### 网格化点云

网格化点云是指点云数据中任意一点，均对应于其参数域所对应的一个均匀网格的顶点。当对空间散乱点云进行网格化插值时，所获得的点云数据即为网格化点云数据。

#### 多边形点云

多边形点云是指分布在一组平面内的点云数据，该组平面内的平面两两互相平行，并且一个平面内距离最近的点连接起来可以形成平面多边形。这种点云数据常见于等高线测量、CT 测量等获得的点云数据中。

## 点云去噪&滤波

> 主要包括双边滤波、高斯滤波、条件滤波、直通滤波、随机采样一致滤波、VoxelGrid滤波等
>
> 三角网格去噪算法、

1. [基于K-近邻点云去噪算法的研究与改进](http://www.cnki.com.cn/Article/CJFDTotal-JSJY200904032.htm)
2. [Point cloud denoising based on tensor Tucker decomposition](https://arxiv.org/abs/1902.07602v2)
3. [3D Point Cloud Denoising using Graph Laplacian Regularization of a Low Dimensional Manifold Model](https://arxiv.org/abs/1803.07252v2)

#### 有序点云去噪

> 孤立点排异法、曲线拟合法、弦高差法、全局能量法和滤波法.
>
> 孤立点排异法是通过观察点云数据，然后将与扫描线偏离较大的点剔除掉，从而达到去噪的目的。这类方法简单，可除去比较明显的噪声点，但缺点是只能对点云做初步的去噪处理，并不能滤除与真实点云数据混合在一起的噪声数据点。曲线拟合法是根据给定数据点的首末点，然后通过最小二乘等方法拟合一条曲线，通常为3到4 阶，最后计算中间的点到该曲线的距离，如果该距离大于给定阈值，则该点为噪声点，予以删
> 除，相反，如果该距离小于给定阈值，则该点为正常点，应该保留。弦高差法通过连接给定点集的首末点形成弦，然后求取中间每个点到该弦的距离，如果该距离小于给定阈值，则该点为正常点，予以保留，相反，若大于给定阈值，则该点为噪声点，予以删除。全局能量法通常用于网格式点云去噪，它通过建立整个曲面的能量方程，并求该方程在约束情况下的能量值的最小值。可以看出，这是一个全局最优化问题，因为网格数量比较大，因此会消耗大量的计算机资源与计算时间，而且由于约束方程是建立在整体网格的基础上，所以对于局部形状的去噪效果并不是很好。滤波法也是一种常用的有序点云去噪方法，它通过运用信号处理中的相关方法，使用合适的滤波函数对点云数据进行去噪处理，常用的滤波方法主要包括高斯滤波、均值滤波以及中值滤波法等。

#### 无序点云去噪&空间散乱点云去噪算法

> 目前，针对空间散乱点云数据去噪方法，主要分为两类方法，即基于网格模型的去噪方法和直接对空间点云数据进行去噪的方法。
>
> 其中，基于网格模型的去噪方法需要首先建立点云的三角网格模型，然后计算所有三角面片的纵横比和顶点方向的曲率值，并将该值与相应的阈值进行比较，若小于阈值，则为正常点，予以保留，相反，则为噪声点，予以删除。由于该方法需要对空间点云数据进行三角网格剖分，所以，往往比较复杂，并需要大量计算。

## 点云精简

> 采用三维激光扫描仪获得的点云数据往往**十分密集**，点云数据中点的数量往往高达千万级甚至数亿级，即使对点云数据进行了去噪处理，点云数据中点的数量还是很多，所以往往不会直接使用这些原始点云数据进行曲面重建等工作，因为这会使后续处理过程变得耗时并且消耗过多的计算机资源，而且重构的曲面，其精度也不一定高，甚至出现更大的误差。所以，在进行空间点云曲面重建之前，往往需要对高密度的点云数据进
> 行点云精简操作。点云精简的目的是在保持原始点云的形状特征以及几何特征信息的前提下，尽量删除多余的数据点。
>
> 目前，空间散乱点云数据的精简方法主要分为两大类：**基于三角网格模型的空间点云精简方法**与**直接基于数据点的空间点云精简方法**。
>
> 其中，基于三角网格模型的空间点云精简方法需要先对点云数据进行三角剖分处理，建立其相应的三角网格拓扑结构，然后再对该三角网格进行处理，并将区域内那些形状变化较小的三角形进行合并，最后删除相关的三角网格顶点，从而达到点云数据精简的目的。这种方法需要对点云数据建立其相应的三角网格，该过程比较复杂，且因为需要存储网格数据，故需要消耗大量的计算机系统资源，并且该方法的抗噪能力较弱，对含有噪声的点云数据，构造的三角网格可能会出现变形等情况，因此精简后的点云数据经过曲面重建后的
> 模型与原始点云经过曲面重建后的模型可能大不相同。因此，目前关于直接基于点云数据的精简方法成为点云精简方法的主流。这种方法依据点云数据点之间的空间位置关系来建立点云的拓扑连接关系，并根据建立的拓扑连接关系计算点云数据中每个数据点的几何特征信息，最后根据这些特征信息来对点云数据进行点云精简处理。相比基于三角网格的空间点云精简方法，由于直接基于点云数据点的精简方法无需计算和存储复杂的三角网格结构，使得其精简的效率相对较高。因此，本章只研究直接基于空间点云数据的精简算法。



其中基于空间点云精简方法主要有：空间包围盒法、基于聚类的方法、法向偏差法、曲率精简法、平局点距法以及均匀栅格划分法。

#### Paper

1. 点模型的几何图像简化法
2. 基于相似性的点模型简化算法
3. 基于最小曲面距离的快速点云精简算法
4. 大规模点云选择及精简
5. 一种基于模糊聚类的海量测量数据简化方法
6. 基于均值漂移聚类的点模型简化方法
7. 基于局部曲面拟合的散乱点云简化方法

## 点云关键点

> 常见的三维点云关键点提取算法有一下几种：ISS3D、Harris3D、NARF、SIFT3D，这些算法在PCL库中都有实现，其中NARF算法是用的比较多的

## 点云描述

> 如果要对一个三维点云进行描述，光有点云的位置是不够的，常常需要计算一些额外的参数，比如法线方向、曲率、文理特征等等。如同图像的特征一样，我们需要使用类似的方式来描述三维点云的特征。
>
> 常用的特征描述算法有：法线和曲率计算、特征值分析、PFH、FPFH、SHOT、VFH、CVFH、3D Shape Context、Spin Image等。PFH：点特征直方图描述子，FPFH：跨苏点特征直方图描述子，FPFH是PFH的简化形式。



## 点云线、面拟合

> 针对直线拟合：RANSAC算法、最小二乘法、平面相交法
>
> 针对曲线拟合：拉格朗日插值法、最小二乘法、Bezier曲线拟合法、B样条曲线法（二次、三次B样条曲线拟合）
>
> 针对平面拟合：主成成分分析、最小二乘法、粗差探测法、抗差估计法
>
> 针对曲面拟合：最小二乘法（正交最小二乘、移动最小二乘）、NURBS、 Bezier 

1. 三维激光扫描拟合平面自动提取算法
2. 点云平面拟合新方法
3. 海量散乱点的曲面重建算法研究
4. 一种稳健的点云数据平面拟合方法 
5. 迭代切片算法在点云曲面拟合中的应用
6. [基于最小二乘的点云叶面拟合算法研究](http://www.cqvip.com/QK/97059X/201405/661608237.html)
7. [点云曲面边界线的提取](http://d.wanfangdata.com.cn/Conference/7057652)

## 点云面积计算

> 针对曲面/平面可以使用积分方法、网格划分（三角网格）等方法计算面积。
>
> 常用的思路：三角剖分算法（Delaunay，Mesh之后）来得到由三角形构成的最小凸包。得到三角形集合后，计算所有三角形面积并求和。

## 点云体积计算

> 基于三维点云求取物理模型体积的研究算法大致可分为以下 4 大类。
>
> 1.凸包算法：使用凸包模型近似表示不规则体，再通过把凸包模型切片分割进行累加、或将凸包模型分解为上下两个三角网格面，采用正投影法求取两者的投影体积，其差即所求体积。此方法适用于凸模型，非凸模型误差较大。
>
> 2.模型重建法：在得到点云数据后，使用三角面片构建物理模型的方法求得体积。该算法受点云密度、生成的三角网格数量、点精度影响较大，易产生孔洞。
>
> 3.切片法：将点云沿某一坐标轴方向进行切片处理，再计算切片上下两表面的面积，通过累加切片体积求得总体积。该方法受到切片厚度的影响，切片厚度越小，计算精度越高但会导致计算效率下降。
>
> 4.投影法：先将点云投影进行三角形剖分，再将投影点与其原对应点构建出五面体，通过累加五面体体积求得总体积。该算法同样容易产生孔洞。上述算法，无论是通过三维点云先构建物理模型再求体积、还是基于三维点云通过几何方法直接求体积，当激光雷达采集的三维点云存在密度不均匀、空间物体存在过渡带或过渡线等问题时，重建三维模型的误差较大，体积计算精度不高。

## 点云识别&分类

> 分类：基于点的分类，基于分割的分类，监督分类与非监督分类
>
> 除此之外，还可以基于描述向量/关键点描述进行分类。

1. [3D ShapeNets: A Deep Representation for Volumetric Shapes](http://3dvision.princeton.edu/projects/2014/3DShapeNets/paper.pdf)
2. [PartNet: A Large-scale Benchmark for Fine-grained and Hierarchical Part-level 3D Object Understanding](https://arxiv.org/abs/1812.02713)
3. [Revisiting Point Cloud Classification: A New Benchmark Dataset and Classification Model on Real-World Data](https://arxiv.org/pdf/1908.04616.pdf)
4. [Escape from Cells: Deep Kd-Networks for the Recognition of 3D Point Cloud Models](http://openaccess.thecvf.com/content_ICCV_2017/papers/Klokov_Escape_From_Cells_ICCV_2017_paper.pdf)[ICCV2017]
5. [[ICCV2017](http://openaccess.thecvf.com/content_ICCV_2017/papers/Park_Colored_Point_Cloud_ICCV_2017_paper.pdf)] Colored Point Cloud Registration Revisited. 
6. [[ICRA2017](https://ieeexplore.ieee.org/document/7989618)] SegMatch: Segment based place recognition in 3D point clouds.
7. [[IROS2017](https://ieeexplore.ieee.org/document/8202239)] 3D object classification with point convolution network.
8. [[CVPR2018](http://openaccess.thecvf.com/content_cvpr_2018/papers/Hua_Pointwise_Convolutional_Neural_CVPR_2018_paper.pdf)] Pointwise Convolutional Neural Networks.
9. [[CVPR2018](http://openaccess.thecvf.com/content_cvpr_2018/papers/Li_SO-Net_Self-Organizing_Network_CVPR_2018_paper.pdf)] SO-Net: Self-Organizing Network for Point Cloud Analysis. 
10. [[CVPR2018](http://openaccess.thecvf.com/content_cvpr_2018/papers/Uy_PointNetVLAD_Deep_Point_CVPR_2018_paper.pdf)] PointNetVLAD: Deep Point Cloud Based Retrieval for Large-Scale Place Recognition. 
11. [[CVPR2018](http://openaccess.thecvf.com/content_cvpr_2018/papers/Le_PointGrid_A_Deep_CVPR_2018_paper.pdf)] PointGrid: A Deep Network for 3D Shape Understanding. 
12. [[CVPR2019](https://raoyongming.github.io/files/SFCNN.pdf)] Spherical Fractal Convolutional Neural Networks for Point Cloud Recognition. 
13. [[MM](https://dl.acm.org/citation.cfm?id=3343031.3351009)] MMJN: Multi-Modal Joint Networks for 3D Shape Recognition. 



## 点云匹配&配准&对齐&注册

> 点云配准的概念也可以类比于二维图像中的配准，只不过二维图像配准获取得到的是x，y，alpha，beta等放射变化参数，三维点云配准可以模拟三维点云的移动和对齐，也就是会获得一个旋转矩阵和一个平移向量，通常表达为一个4×3的矩阵，其中3×3是旋转矩阵，1x3是平移向量。严格说来是6个参数，因为旋转矩阵也可以通过罗格里德斯变换转变成1*3的旋转向量。
>
> 常用的点云配准算法有两种：**正太分布变换**和著名的**ICP点云配准**，此外还有许多其它算法，列举如下：
>
> ICP：稳健ICP、point to plane ICP、point to line ICP、MBICP、GICP
>
> NDT 3D、Multil-Layer NDT
>
> FPCS、KFPSC、SAC-IA
>
> Line Segment Matching、ICL

1. [An ICP variant using a point-to-line metric](https://authors.library.caltech.edu/18274/1/Censi2008p84782008_Ieee_International_Conference_On_Robotics_And_Automation_Vols_1-9.pdf)
2. [Generalized-ICP](https://www.researchgate.net/publication/221344436_Generalized-ICP)
3. [Linear Least-Squares Optimization for Point-to-Plane ICP Surface Registration](https://www.researchgate.net/publication/228571031_Linear_Least-Squares_Optimization_for_Point-to-Plane_ICP_Surface_Registration)
4. [Metric-Based Iterative Closest Point Scan Matching for Sensor Displacement Estimation](http://webdiis.unizar.es/~jminguez/MbICP_TRO.pdf)
5. [NICP: Dense Normal Based Point Cloud Registration](http://jacoposerafin.com/wp-content/uploads/serafin15iros.pdf)
6. [Efficient Global Point Cloud Alignment using Bayesian Nonparametric Mixtures](http://openaccess.thecvf.com/content_cvpr_2017/papers/Straub_Efficient_Global_Point_CVPR_2017_paper.pdf)[CVPR2017]
7. [3DMatch: Learning Local Geometric Descriptors from RGB-D Reconstructions](http://openaccess.thecvf.com/content_cvpr_2017/papers/Zeng_3DMatch_Learning_Local_CVPR_2017_paper.pdf)[CVPR2017]
8. [[CVPR2018](http://openaccess.thecvf.com/content_cvpr_2018/papers/Lawin_Density_Adaptive_Point_CVPR_2018_paper.pdf)] Density Adaptive Point Set Registration.
9. [[CVPR2018](http://openaccess.thecvf.com/content_cvpr_2018/papers/Vongkulbhisal_Inverse_Composition_Discriminative_CVPR_2018_paper.pdf)] Inverse Composition Discriminative Optimization for Point Cloud Registration.
10. [[CVPR2018](http://openaccess.thecvf.com/content_cvpr_2018/papers/Deng_PPFNet_Global_Context_CVPR_2018_paper.pdf)] PPFNet: Global Context Aware Local Features for Robust 3D Point Matching.
11. [[ECCV2018](http://openaccess.thecvf.com/content_ECCV_2018/papers/Lei_Zhou_Learning_and_Matching_ECCV_2018_paper.pdf)] Learning and Matching Multi-View Descriptors for Registration of Point Clouds. 
12. [[ECCV2018](http://openaccess.thecvf.com/content_ECCV_2018/papers/Zi_Jian_Yew_3DFeat-Net_Weakly_Supervised_ECCV_2018_paper.pdf)] 3DFeat-Net: Weakly Supervised Local 3D Features for Point Cloud Registration. 
13. [[ECCV2018](http://openaccess.thecvf.com/content_ECCV_2018/papers/Yinlong_Liu_Efficient_Global_Point_ECCV_2018_paper.pdf)] Efficient Global Point Cloud Registration by Matching Rotation Invariant Features Through Translation Search. 
14. [[IROS2018](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8593558)] Robust Generalized Point Cloud Registration with Expectation Maximization Considering Anisotropic Positional Uncertainties. 
15. [[CVPR2019](https://arxiv.org/abs/1903.05711)] PointNetLK: Point Cloud Registration using PointNet. 
16. [[CVPR2019](https://arxiv.org/abs/1904.03483)] SDRSAC: Semidefinite-Based Randomized Approach for Robust Point Cloud Registration without Correspondences. 
17. [[CVPR2019](https://arxiv.org/abs/1811.06879v2)] The Perfect Match: 3D Point Cloud Matching with Smoothed Densities. 
18. [[CVPR](https://arxiv.org/abs/1811.10136)] FilterReg: Robust and Efficient Probabilistic Point-Set Registration using Gaussian Filter and Twist Parameterization. 
19. [[CVPR2019](http://openaccess.thecvf.com/content_CVPR_2019/papers/Deng_3D_Local_Features_for_Direct_Pairwise_Registration_CVPR_2019_paper.pdf)] 3D Local Features for Direct Pairwise Registration. 
20. [[ICCV2019](https://arxiv.org/abs/1905.04153v2)] DeepICP: An End-to-End Deep Neural Network for 3D Point Cloud Registration.
21. [[ICCV2019](http://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_Deep_Closest_Point_Learning_Representations_for_Point_Cloud_Registration_ICCV_2019_paper.pdf)] Deep Closest Point: Learning Representations for Point Cloud Registration. 
22. [[ICRA2019](https://arxiv.org/abs/1904.09742)] 2D3D-MatchNet: Learning to Match Keypoints across 2D Image and 3D Point Cloud.
23. [[CVPR2019](https://arxiv.org/abs/1811.06879v2)] The Perfect Match: 3D Point Cloud Matching with Smoothed Densities.
24. [[CVPR2019](http://openaccess.thecvf.com/content_CVPR_2019/papers/Deng_3D_Local_Features_for_Direct_Pairwise_Registration_CVPR_2019_paper.pdf)] 3D Local Features for Direct Pairwise Registration.
25. [[ICCV2019](http://openaccess.thecvf.com/content_ICCV_2019/papers/Zhou_Robust_Variational_Bayesian_Point_Set_Registration_ICCV_2019_paper.pdf)] Robust Variational Bayesian Point Set Registration.
26. [[ICRA2019](https://arpg.colorado.edu/papers/hmrf_icp.pdf)] Robust low-overlap 3-D point cloud registration for outlier rejection. 
27. Learning multiview 3D point cloud registration[[CVPR2020]()]



## 点云匹配质量评估

1. [[IROS2017](https://ieeexplore.ieee.org/document/8206584)] Analyzing the quality of matched 3D point clouds of objects.

## 点云分割

> 点云的分割也算是一个大Topic了，这里因为多了一维就和二维图像比多了许多问题，点云分割又分为区域提取、线面提取、语义分割与聚类等。同样是分割问题，点云分割涉及面太广，确实是三言两语说不清楚的。只有从字面意思去理解了，遇到具体问题再具体归类。一般说来，点云分割是目标识别的基础。
>
> 分割主要有四种方法：基于边的区域分割、基于面的区域分割、基于聚类的区域分割、混合区域分割方法、深度学习方法
>
> 分割：区域声场、Ransac线面提取、NDT-RANSAC、K-Means（谱聚类）、Normalize Cut、3D Hough Transform(线面提取)、连通分析

1.  [基于局部表面凸性的散乱点云分割算法研究](http://ir.ciomp.ac.cn/handle/181722/57569?mode=full&submit_simple=Show+full+item+record)
2.  [三维散乱点云分割技术综述](http://www.cnki.com.cn/Article/CJFDTotal-ZZGX201005012.htm)
3.  [基于聚类方法的点云分割技术的研究](http://cdmd.cnki.com.cn/Article/CDMD-10213-1015979890.htm)
4.  [SceneEncoder: Scene-Aware Semantic Segmentation of Point Clouds with A Learnable Scene Descriptor](https://arxiv.org/abs/2001.09087v1)
5.  [From Planes to Corners: Multi-Purpose Primitive Detection in Unorganized 3D Point Clouds](https://arxiv.org/abs/2001.07360?context=cs.RO)
6.  [Learning and Memorizing Representative Prototypes for 3D Point Cloud Semantic and Instance Segmentation](http://arxiv.org/abs/2001.01349)
7.  [JSNet: Joint Instance and Semantic Segmentation of 3D Point Clouds](https://arxiv.org/abs/1912.09654v1)
8.  [PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](https://arxiv.org/abs/1612.00593v2)
9.  [PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space](https://arxiv.org/abs/1706.02413v1)
10.  [SyncSpecCNN: Synchronized Spectral CNN for 3D Shape Segmentation,CVPR2017]()
11.  [[ICRA2017](https://ieeexplore.ieee.org/document/7989618)] SegMatch: Segment based place recognition in 3D point clouds.
12.  [[3DV2017](http://segcloud.stanford.edu/segcloud_2017.pdf)] SEGCloud: Semantic Segmentation of 3D Point Clouds. 
13.  [[CVPR2018](http://openaccess.thecvf.com/content_cvpr_2018/papers/Huang_Recurrent_Slice_Networks_CVPR_2018_paper.pdf)] Recurrent Slice Networks for 3D Segmentation of Point Clouds. 
14.  [[CVPR2018](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_SGPN_Similarity_Group_CVPR_2018_paper.pdf)] SGPN: Similarity Group Proposal Network for 3D Point Cloud Instance Segmentation.
15.  [[CVPR2018](http://openaccess.thecvf.com/content_cvpr_2018/papers/Landrieu_Large-Scale_Point_Cloud_CVPR_2018_paper.pdf)] Large-scale Point Cloud Semantic Segmentation with Superpoint Graphs.
16.  [[ECCV2018](http://openaccess.thecvf.com/content_ECCV_2018/papers/Xiaoqing_Ye_3D_Recurrent_Neural_ECCV_2018_paper.pdf)] 3D Recurrent Neural Networks with Context Fusion for Point Cloud Semantic Segmentation.
17.  [[CVPR2019](https://arxiv.org/abs/1904.00699v1)] JSIS3D: Joint Semantic-Instance Segmentation of 3D Point Clouds with Multi-Task Pointwise Networks and Multi-Value Conditional Random Fields. 
18.  [[CVPR2019](https://arxiv.org/abs/1903.00709)] PartNet: A Recursive Part Decomposition Network for Fine-grained and Hierarchical Shape Segmentation. 
19.  [[ICCV2019](http://openaccess.thecvf.com/content_ICCV_2019/papers/Lahoud_3D_Instance_Segmentation_via_Multi-Task_Metric_Learning_ICCV_2019_paper.pdf)] 3D Instance Segmentation via Multi-Task Metric Learning.
20.  [[IROS2019](https://arxiv.org/pdf/1909.01643v1.pdf)] PASS3D: Precise and Accelerated Semantic Segmentation for 3D Point Cloud.

## 3D/点云目标检索

> 这是点云数据处理中一个偏应用层面的问题，简单说来就是Hausdorff距离常被用来进行深度图的目标识别和检索，现在很多三维[人脸识别](https://cloud.tencent.com/product/facerecognition?from=10680)都是用这种技术来做的。
>
> 3D检索方法主要包括基于统计特征、基于拓扑结构、基于空间图、基于局部特征、基于视图。
>
> 基于统计特征的方法包括：基于矩、体积、面积
>
> 基于拓扑结构的方法包括：基于骨架、基于Reeb图  
>
> 基于空间图的方法包括：形状直方图、球谐调和 、3D Zernike 、法线投影  
>
> 基于局部特征的方法包括：表面曲率  
>
> 基于视图的方法包括：光场描述子、草图  

## 点云三维重建

> 我们获取到的点云数据都是一个个孤立的点，如何从一个个孤立的点得到整个曲面呢，这就是三维重建的topic。
>
> 在玩kinectFusion时候，如果我们不懂，会发现曲面渐渐变平缓，这就是重建算法不断迭代的效果。我们采集到的点云是充满噪声和孤立点的，三维重建算法为了重构出曲面，常常要应对这种噪声，获得看上去很舒服的曲面。
>
> 常用的三维重建算法和技术有：
>
> 泊松重建、Delauary triangulatoins(Delauary三角化)
>
> 表面重建，人体重建，建筑物重建，输入重建
>
> 实时重建：重建纸杯或者农作物4D生长台式，人体姿势识别，表情识别

1. [改进的点云数据三维重建算法](http://www.opticsjournal.net/Articles/Abstract?aid=OJ4723ebf4b0b0762)
2. [Scalable Surface Reconstruction from Point Clouds with Extreme Scale and Density Diversity,CVPR2017](http://openaccess.thecvf.com/content_cvpr_2017/papers/Mostegel_Scalable_Surface_Reconstruction_CVPR_2017_paper.pdf)
3. [[ICCV2017](http://openaccess.thecvf.com/content_ICCV_2017/papers/Nan_PolyFit_Polygonal_Surface_ICCV_2017_paper.pdf)] PolyFit: Polygonal Surface Reconstruction from Point Clouds.
4. [[ICCV2017](http://openaccess.thecvf.com/content_ICCV_2017/papers/Ladicky_From_Point_Clouds_ICCV_2017_paper.pdf)] From Point Clouds to Mesh using Regression.
5. [[ECCV2018](http://openaccess.thecvf.com/content_ECCV_2018/papers/Kejie_Li_Efficient_Dense_Point_ECCV_2018_paper.pdf)] Efficient Dense Point Cloud Object Reconstruction using Deformation Vector Fields. 
6. [[ECCV2018](http://openaccess.thecvf.com/content_ECCV_2018/papers/Benjamin_Eckart_Fast_and_Accurate_ECCV_2018_paper.pdf)] HGMR: Hierarchical Gaussian Mixtures for Adaptive 3D Registration.
7. [[AAAI2018](https://aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16530/16302)] Learning Efficient Point Cloud Generation for Dense 3D Object Reconstruction. 
8. [[CVPR2019](https://www.researchgate.net/publication/332240602_Robust_Point_Cloud_Based_Reconstruction_of_Large-Scale_Outdoor_Scenes)] Robust Point Cloud Based Reconstruction of Large-Scale Outdoor Scenes. 
9. [[AAAI2019](https://arxiv.org/abs/1811.11731)] CAPNet: Continuous Approximation Projection For 3D Point Cloud Reconstruction Using 2D Supervision. 
10. [[MM](https://dl.acm.org/citation.cfm?id=3350960)] L2G Auto-encoder: Understanding Point Clouds by Local-to-Global Reconstruction with Hierarchical Self-Attention. 
11. [SurfNet: Generating 3D shape surfaces using deep residual networks]()

## 点云其它

1. [[CVPR2018](http://openaccess.thecvf.com/content_cvpr_2018/papers/Yun_Reflection_Removal_for_CVPR_2018_paper.pdf)] Reflection Removal for Large-Scale 3D Point Clouds. 
2. [[ICML2018](https://arxiv.org/abs/1707.02392)] Learning Representations and Generative Models for 3D Point Clouds.
3. [[3DV](https://arxiv.org/abs/1808.00671)] PCN: Point Completion Network. 
4. [[CVPR2019](https://arxiv.org/abs/1812.02713)] PartNet: A Large-scale Benchmark for Fine-grained and Hierarchical Part-level 3D Object Understanding. 
5. [[CVPR2019](http://www.linliang.net/wp-content/uploads/2019/04/CVPR2019_PointClound.pdf)] ClusterNet: Deep Hierarchical Cluster Network with Rigorously Rotation-Invariant Representation for Point Cloud Analysis.
6. [[ICCV2019](https://arxiv.org/pdf/1812.07050.pdf)] LPD-Net: 3D Point Cloud Learning for Large-Scale Place Recognition and Environment Analysis. 
7. [[ICRA2019](https://ras.papercept.net/conferences/conferences/ICRA19/program/ICRA19_ContentListWeb_2.html)] Speeding up Iterative Closest Point Using Stochastic Gradient Descent. 

## 点云数据集

1. [[KITTI](http://www.cvlibs.net/datasets/kitti/)] The KITTI Vision Benchmark Suite.
2. [[ModelNet](http://modelnet.cs.princeton.edu/)] The Princeton ModelNet . 
3. [[ShapeNet](https://www.shapenet.org/)] A collaborative dataset between researchers at Princeton, Stanford and TTIC.
4. [[PartNet](https://shapenet.org/download/parts)] The PartNet dataset provides fine grained part annotation of objects in ShapeNetCore. 
5. [[PartNet](http://kevinkaixu.net/projects/partnet.html)] PartNet benchmark from Nanjing University and National University of Defense Technology. 
6. [[S3DIS](http://buildingparser.stanford.edu/dataset.html#Download)] The Stanford Large-Scale 3D Indoor Spaces Dataset.
7. [[ScanNet](http://www.scan-net.org/)] Richly-annotated 3D Reconstructions of Indoor Scenes. 
8. [[Stanford 3D](https://graphics.stanford.edu/data/3Dscanrep/)] The Stanford 3D Scanning Repository. 
9. [[UWA Dataset](http://staffhome.ecm.uwa.edu.au/~00053650/databases.html)] . 
10. [[Princeton Shape Benchmark](http://shape.cs.princeton.edu/benchmark/)] The Princeton Shape Benchmark.
11. [[SYDNEY URBAN OBJECTS DATASET](http://www.acfr.usyd.edu.au/papers/SydneyUrbanObjectsDataset.shtml)] This dataset contains a variety of common urban road objects scanned with a Velodyne HDL-64E LIDAR, collected in the CBD of Sydney, Australia. There are 631 individual scans of objects across classes of vehicles, pedestrians, signs and trees.
12. [[ASL Datasets Repository(ETH)](https://projects.asl.ethz.ch/datasets/doku.php?id=home)] This site is dedicated to provide datasets for the Robotics community with the aim to facilitate result evaluations and comparisons.
13. [[Large-Scale Point Cloud Classification Benchmark(ETH)](http://www.semantic3d.net/)] This benchmark closes the gap and provides a large labelled 3D point cloud data set of natural scenes with over 4 billion points in total. 
14. [[Robotic 3D Scan Repository](http://asrl.utias.utoronto.ca/datasets/3dmap/)] The Canadian Planetary Emulation Terrain 3D Mapping Dataset is a collection of three-dimensional laser scans gathered at two unique planetary analogue rover test facilities in Canada.
15. [[Radish](http://radish.sourceforge.net/)] The Robotics Data Set Repository (Radish for short) provides a collection of standard robotics data sets.
16. [[IQmulus & TerraMobilita Contest](http://data.ign.fr/benchmarks/UrbanAnalysis/#)] The database contains 3D MLS data from a dense urban environment in Paris (France), composed of 300 million points. The acquisition was made in January 2013. 
17. [[Oakland 3-D Point Cloud Dataset](http://www.cs.cmu.edu/~vmr/datasets/oakland_3d/cvpr09/doc/)] This repository contains labeled 3-D point cloud laser data collected from a moving platform in a urban environment.
18. [[Robotic 3D Scan Repository](http://kos.informatik.uni-osnabrueck.de/3Dscans/)] This repository provides 3D point clouds from robotic experiments，log files of robot runs and standard 3D data sets for the robotics community.
19. [[Ford Campus Vision and Lidar Data Set](http://robots.engin.umich.edu/SoftwareData/Ford)] The dataset is collected by an autonomous ground vehicle testbed, based upon a modified Ford F-250 pickup truck.
20. [[The Stanford Track Collection](https://cs.stanford.edu/people/teichman/stc/)] This dataset contains about 14,000 labeled tracks of objects as observed in natural street scenes by a Velodyne HDL-64E S2 LIDAR.
21. [[PASCAL3D+](http://cvgl.stanford.edu/projects/pascal3d.html)] Beyond PASCAL: A Benchmark for 3D Object Detection in the Wild.
22. [[3D MNIST](https://www.kaggle.com/daavoo/3d-mnist)] The aim of this dataset is to provide a simple way to get started with 3D computer vision problems such as 3D shape recognition.
23. [[WAD](http://wad.ai/2019/challenge.html)] [[ApolloScape](http://apolloscape.auto/tracking.html)] The datasets are provided by Baidu Inc. 
24. [[nuScenes](https://d3u7q4379vrm7e.cloudfront.net/object-detection)] The nuScenes dataset is a large-scale autonomous driving dataset.
25. [[PreSIL](https://uwaterloo.ca/waterloo-intelligent-systems-engineering-lab/projects/precise-synthetic-image-and-lidar-presil-dataset-autonomous)] Depth information, semantic segmentation (images), point-wise segmentation (point clouds), ground point labels (point clouds), and detailed annotations for all vehicles and people. [[paper](https://arxiv.org/abs/1905.00160)] 
26. [[3D Match](http://3dmatch.cs.princeton.edu/)] Keypoint Matching Benchmark, Geometric Registration Benchmark, RGB-D Reconstruction Datasets. 
27. [[BLVD](https://github.com/VCCIV/BLVD)] (a) 3D detection, (b) 4D tracking, (c) 5D interactive event recognition and (d) 5D intention prediction. [[ICRA 2019 paper](https://arxiv.org/abs/1903.06405v1)] 
28. [[PedX](https://arxiv.org/abs/1809.03605)] 3D Pose Estimation of Pedestrians, more than 5,000 pairs of high-resolution (12MP) stereo images and LiDAR data along with providing 2D and 3D labels of pedestrians. [[ICRA 2019 paper](https://arxiv.org/abs/1809.03605)] 
29. [[H3D](https://usa.honda-ri.com/H3D)] Full-surround 3D multi-object detection and tracking dataset. [[ICRA 2019 paper](https://arxiv.org/abs/1903.01568)] 
30. [[Matterport3D](https://niessner.github.io/Matterport/)] RGB-D: 10,800 panoramic views from 194,400 RGB-D images. Annotations: surface reconstructions, camera poses, and 2D and 3D semantic segmentations. Keypoint matching, view overlap prediction, normal prediction from color, semantic segmentation, and scene classification. [[3DV 2017 paper](https://arxiv.org/abs/1709.06158)] [[code](https://github.com/niessner/Matterport)] [[blog](https://matterport.com/blog/2017/09/20/announcing-matterport3d-research-dataset/)]
31. [[SynthCity](https://arxiv.org/abs/1907.04758)] SynthCity is a 367.9M point synthetic full colour Mobile Laser Scanning point cloud. Nine categories. 
32. [[Lyft Level 5](https://level5.lyft.com/dataset/?source=post_page)] Include high quality, human-labelled 3D bounding boxes of traffic agents, an underlying HD spatial semantic map.
33. [[SemanticKITTI](http://semantic-kitti.org/)] Sequential Semantic Segmentation, 28 classes, for autonomous driving. All sequences of KITTI odometry labeled. [[ICCV 2019 paper](https://arxiv.org/abs/1904.01416)] 
34. [[NPM3D](http://npm3d.fr/paris-lille-3d)] The Paris-Lille-3D has been produced by a Mobile Laser System (MLS) in two different cities in France (Paris and Lille).
35. [[The Waymo Open Dataset](https://waymo.com/open/)] The Waymo Open Dataset is comprised of high resolution sensor data collected by Waymo self-driving cars in a wide variety of conditions. 
36. [[A*3D: An Autonomous Driving Dataset in Challeging Environments](https://github.com/I2RDL2/ASTAR-3D)] A*3D: An Autonomous Driving Dataset in Challeging Environments. 
37. [[PointDA-10 Dataset](https://github.com/canqin001/PointDAN)] Domain Adaptation for point clouds.
38. [[Oxford Robotcar](https://robotcar-dataset.robots.ox.ac.uk/)] The dataset captures many different combinations of weather, traffic and pedestrians.
