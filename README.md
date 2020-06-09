## 人工智能写诗 AICHPOEM.COM

一步步教你搭建人工智能写诗平台，包括：
* 写诗模型搭建
* 写诗奖惩机制
* 后端发布
* 高并发架构
* 前端开发
* 敏感词过滤
* 运营推广SEO优化

### 写诗模型搭建
写诗模型采用苏建林老师的[bert4keras](https://github.com/bojone/bert4keras/tree/master/)的[seq2seq](https://github.com/bojone/bert4keras/blob/master/examples/task_seq2seq_autotitle.py), 用[RoBerta_wwm_ext](https://github.com/ymcui/Chinese-BERT-wwm)预训练模型训练的

数据集用了[70万首格律诗](https://github.com/Werneror/Poetry)，预处理格式如下：

输入： 题目&&格律 （格律包括 五言绝句，七言绝句，五言律诗，七言律诗）
输出： 对应的古诗

例：

输入：相思&&五言绝句
输出：红豆生南国，春来发几枝。愿君多采撷，此物最相思。

输入：春望&&五言律诗
输出：国破山河在，城春草木深。感时花溅泪，恨别鸟惊心。烽火连三月，家书抵万金。白头搔更短，浑欲不胜簪。

输入: 早发白帝城&&七言绝句
输出：朝辞白帝彩云间，千里江陵一日还。两岸猿声啼不住，轻舟已过万重山。

输入：锦瑟&&七言律诗
输出：锦瑟无端五十弦，一弦一柱思华年。庄生晓梦迷蝴蝶，望帝春心托杜鹃。沧海月明珠有泪，蓝田日暖玉生烟。此情可待成追忆，只是当时已惘然。





