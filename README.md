# classification-of-breast-cancer
using different way to classify 4 classes about breast cancer
### 项目介绍
####  为了能够得到病理图片上的乳腺癌区域而做的四分类模型。四个类别分别为正常（normal）、良性癌（benign）、原位癌（insitu）、浸润癌（invasive）。

#### 数据集
##### 2018 bach challenge 数据集
- 1、400张2048x1536的有标签tif图片，每个类分别有100张
- 2、10张有标注的svs图片
- 3、100张无标签的测试tif图片
- 4、10张无标注的测试用svs图片

##### cnn模型
- 1、切割图片：patch大小、是否重叠、注意图片保存的格式
- 2、背景区域过滤：去除白色像素点过多的patch
- 3、数据增强：旋转、翻转、颜色增强、模糊、放大缩小
- 4、svs图片的标注：xml文件的读取
- 5、训练模型：lr、bs的选取，
- 6、预测svs图：svs图片的切割以及遍历循环，保存预测得到的结果矩阵
- 7、画结果热图：结合结果矩阵与svs缩略图，画出热图

##### cnn+lstm模型
- 1、切割图片，构建数据集：patch为512x512
- 2、背景区域过滤
- 3、数据增强
- 4、训练cnn分类模型
- 5、获取特征：训练好的cnn模型作为特征提取器，取瓶颈特征作为输入lstm训练的特征。
- 6、用于提取特征的图片：有标签2048x1536的大图，只取中间的1536x1536，共9个patch，即共9个特征，一张大图对应特征有9个。
- 7、预测svs图：舍弃svs图片的512像素边缘，预测一个512x512的patch要裁剪1536x1536大小的patch，要预测的图片在正中央，再把这个大的patch裁剪为9个512x512大小的patch，按顺序输入cnn模型中提取特征，再将特征输入训练好的lstm模型中进行预测。
- 8、画结果热图

##### mil模型
###### 多示例的思想是将一个有粗略标注的图片作为一个bag，从这个bag里面挑选出最有可能是bag对应label的patch作为训练样本，从而使数据更加可信。本项目中的bag是2048x1536的大图，每个bag切512x512的patch，重叠256，一个bag可以有35个样本，从中挑选3个最大概率是bag对应标签的patch作为训练集样本。
###### 每训练一轮都要重新挑选训练集数据
- 1、构建数据集（大图层面）
- 2、挑选数据
- 3、挑选数据的可视化：保存每个patch预测得到的概率并打印到对应的patch上面，并将处理好的patch按照切割的x、y的顺序拼接起来。用一个二值矩阵来保存被挑选patch对应的位置。将拼接图与二值矩阵融合起来，得到对应的可视化图片，被挑选图片会比较亮一些。
- 3、训练模型并保存
- 4、用上一轮保存的模型挑选新的训练集


#### 目录以及文件说明
##### /buile_dataset/ 存放生成数据集的脚本
- cut_dataset: 无重叠切割图片
- cut_dataset_overlap:有重叠切割图片

##### /utils/ 保存需要调用的函数
- classes4_preview: 根据xml文件画出svs图片的标注（bach challeng data）
- get_preview_2: 根据xml文件画出svs图片的标注（公司数据）
- DelaunayFill：
将矩阵经过Delaunay三角剖分算法处理，并将三角部分进行填充
- generators： 添加了颜色增强的keras的ImageDataGenerator
- get_img_data: 提取图片的cnn特征
- point_polygon_utils

##### /dataset_processing/ 存放数据集增强处理的脚本
- data_color_normalization：对图片进行染色标准化
- get_generators_data：数据增强脚本，分别对一张图片进行旋转、翻转等操作。

##### /predict/ 存放预测图片的脚本
- cnn_predict_svs: cnn模型预测svs大图并得到结果矩阵以及热图
- lstm_predict_svs: lstm模型预测svs大图并得到结果矩阵以及热图
- predict_bach_testdata: 预测bach challeng中的100张测试图片并打印预测结果
- predict_get_false_pathch: 预测切割好且知道标签的512x512patch，并保存预测错误的样本名
