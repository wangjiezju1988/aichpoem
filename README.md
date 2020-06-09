## 人工智能写诗 AICHPOEM.COM

一步步教你搭建人工智能写诗平台，包括：
* 数据集预处理
* 写诗模型搭建
* BeamSearch奖惩机制
* 后端发布
* 高并发架构
* 前端开发
* 敏感词过滤
* 运营推广SEO优化

### 数据集预处理

数据集采用了[Werneror收集的85万古诗词](https://github.com/Werneror/Poetry)，预处理格式如下：

需要自己写个程序判断一下古诗词输入什么格律或词牌, 并转换成如下格式保存到csv文件

输入： 题目&&格律    
输出： 对应的古诗

格律详情如下：  
古诗： 五言绝句，七言绝句，五言律诗，七言律诗  
藏头诗： 五言绝句_藏头诗，七言绝句_藏头诗, 五言律诗_藏头诗， 七言律诗_藏头诗  
词牌： 浣溪沙、鹧鸪天、蝶恋花、西江月、清平乐、菩萨銮、点绛唇、念奴娇、满江红、浪淘沙等  
对联： 上联  

例子：

输入：相思&&五言绝句  
输出：红豆生南国，春来发几枝。愿君多采撷，此物最相思。

输入：春望&&五言律诗  
输出：国破山河在，城春草木深。感时花溅泪，恨别鸟惊心。烽火连三月，家书抵万金。白头搔更短，浑欲不胜簪。

输入：早发白帝城&&七言绝句  
输出：朝辞白帝彩云间，千里江陵一日还。两岸猿声啼不住，轻舟已过万重山。

输入：锦瑟&&七言律诗  
输出：锦瑟无端五十弦，一弦一柱思华年。庄生晓梦迷蝴蝶，望帝春心托杜鹃。沧海月明珠有泪，蓝田日暖玉生烟。此情可待成追忆，只是当时已惘然。

输入：朝千两轻&&七言绝句_藏头诗
输出：朝辞白帝彩云间，千里江陵一日还。两岸猿声啼不住，轻舟已过万重山。

输入：一曲新词酒一杯&&浣溪沙
输出：一曲新词酒一杯，去年天气旧亭台。夕阳西下几时回？无可奈何花落去，似曾相识燕归来。小园香径独徘徊。

输入：一觉睡西天，谁知梦里乾坤大&&上联
输出：只身眠净土，只道其中日月长

输入：只身眠净土，只道其中日月长&&上联
输出：一觉睡西天，谁知梦里乾坤大

经过预处理，大概统计了一下，古诗有70多万首，对应的藏头诗也有70多万首，藏头诗对应的主题不一定取所有字，可以随机取前几个字，词有3万多首，对联有70万副，上下联对调可以扩增到140万副。由于词牌比较少，需要做数据增强，我用tfidf + textrank算法对每首词取top10个主题词作为输入，对应的词内容还是不变，这样扩增到30多万首左右。

### 写诗模型搭建
写诗模型采用苏建林老师的[bert4keras](https://github.com/bojone/bert4keras/tree/master/)的[seq2seq](https://github.com/bojone/bert4keras/blob/master/examples/task_seq2seq_autotitle.py), 采用[RoBerta_wwm_ext](https://github.com/ymcui/Chinese-BERT-wwm)预训练模型进行训练，对应好古诗数据集的输入输出就可以训练模型了。

每训练一个epoch，可以在训练集里随机找三个样本进行测试，评估效果，参考代码如下：
```
class Evaluate(Callback):
  def __init__(self,model, type):
      self.lowest = 1e10
      self.model = model
      self.type = type

  def on_epoch_end(self, epoch, logs=None):
    idxs = np.random.choice(range(len(newsdata)), 3)
    for idx in idxs:
      if self.type == 'coupletpoem':
        cptype = newsdata['input'].iloc[idx].split('&&')[1]
        print('格式：', cptype if cptype !='上联' else '对联')
        print('输入：', newsdata['input'].iloc[idx].split('&&')[0])
        try:
          if cptype == '上联':
            res = couplet_gen_sent(self.model, newsdata['input'].iloc[idx][:maxlen])
          elif cptype in ['五言绝句','七言绝句','五言律诗','七言律诗','五言绝句_藏头诗','七言绝句_藏头诗','五言律诗_藏头诗','七言律诗_藏头诗']:
            res = poem_gen_sent(self.model, newsdata['input'].iloc[idx][:maxlen])
          else:
            res = ci_gen_sent(self.model, newsdata['input'].iloc[idx][:maxlen])
          print('输出：', newsdata['output'].iloc[idx])
          print('预测：', res)
        except Exception as e:
          traceback.print_exc()
          print(e)
```

RTX2080Ti下大概训练一天左右，loss降到3.5以下，模型已经基本上能学会写诗了，包含不同格律诗的长度及押韵





