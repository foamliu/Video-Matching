# 视频匹配

## 原理：

![image](https://github.com/foamliu/Video-Matching/raw/master/images/video_match.JPG)

## 算法流程

1. 把视频素材逐帧计算特征。以妙可蓝多为例：帧速=25，10秒广告共251帧，特征512维。
2. 形成矩阵mat（帧数x维度）
3. 把巡视照片计算特征得向量 feature
4. 计算：cosine = np.dot(mat, feature)
5. 计算： max_index = np.argmax(cosine)
6. 计算：max_value = cosine[max_index]
7. 计算：theta = math.acos(max_value)
8. 阈值计算：若 theta < threshold then return OK, max_index.

## 作图

余弦相似度与帧的时刻作图：

![image](https://github.com/foamliu/Video-Matching/raw/master/images/cosine_similarity_vs_time.png)

## 阈值

阈值：25.50393648495902

![image](https://github.com/foamliu/Video-Matching/raw/master/images/theta_dist.png)


## 结果

属性名|属性值|
|---|---|
|视频帧数|251|
|max(余弦相似度)|0.948707|
|theta(角度)|18.43065670378278|
|theta 阈值|25.50393648495902|
|是否匹配|是(*)|
|置信度|0.9967838283534795|
|匹配位置(帧)|82|
|匹配位置(秒)|3.28|

注释：因为theta角度小于阈值。