# GAIIC2022-JingDong-Product-Entity-Recognition
此仓库代码为本人参加的 GAIIC-2022 初赛和复赛的提交代码 (京东商品标题识别-NER)，复赛rank-29，初赛rank-42。队员：Furen Xu，Rui Chen, Shuailong Wang
# 一、赛题背景
京东商品标题包含了商品的大量关键信息，商品标题实体识别是NLP应用中的一项核心基础任务，能为多种下游场景所复用，从标题文本中准确抽取出商品相关实体能够提升检索、推荐等业务场景下的用户体验和平台效率。本赛题要求选手使用模型抽取出商品标题文本中的实体。
与传统的实体抽取不同，京东商品标题文本的实体密度高、实体粒度细，赛题具有特色性。
 

# 二、比赛数据
本赛题数据来源于特定类目的京东商品标题短文本，分为有标注样本和无标注样本，供选手选择使用。  

数据格式：训练集数据每一行第一列为一个字符或空格（汉字、英文字母、数字、标点符号、特殊符号、空格），第二列为BIO形式的标签，两列以空格分隔。  

两条标注样本之间以空行为分割。  

训练集：有标注训练样本：4万条左右（包括验证集，不再单独提供验证集，由选手自己切分；总量根据baseline模型效果可能会稍作调整）；无标注样本：100万条。  
初赛A榜测试集：1万条（与训练样本格式相同，差异仅在于无标注）  
初赛B榜测试集：1万条（与训练样本格式相同，差异仅在于无标注）  
复赛测试集：1万条（与训练样本格式相同，差异仅在于无标注）  
