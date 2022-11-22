## Paddle 配置文件优先级

1.配置来源1：入口配置文件的`_BASE_`之下的各个文件。从上到下遍历，后来者覆盖前者。

```yaml
# 1.配置来源1：`_BASE_`
_BASE_: [
  '../datasets/coco_instance.yml',
  '../runtime.yml',
  '_base_/solov2_r50_fpn.yml',
  '_base_/optimizer_1x.yml',
  '_base_/solov2_light_reader.yml',
]

# 2.配置来源2
pretrain_weights:  https://paddledet.bj.bcebos.com/models/pretrained/ResNet50_vd_ssld_v2_pretrained.pdparams
#weights: output/solov2_r50_fpn_3x_coco/model_final
weights: output/solov2_r50_enhance_coco/model_final
epoch: 100
use_ema: true
ema_decay: 0.9998

ResNet:
  depth: 50
  variant: d
  freeze_at: 0
  freeze_norm: false
  norm_type: sync_bn
  return_idx: [0,1,2,3]
  dcn_v2_stages: [1,2,3]
  lr_mult_list: [0.05, 0.05, 0.1, 0.15]
  num_stages: 4

SOLOv2Head:
  seg_feat_channels: 256
  stacked_convs: 3
  num_grids: [40, 36, 24, 16, 12]
  kernel_out_channels: 128
  solov2_loss: SOLOv2Loss
  mask_nms: MaskMatrixNMS
  dcn_v2_stages: [2]
  drop_block: True

SOLOv2MaskHead:
  mid_channels: 128
  out_channels: 128
  start_level: 0
  end_level: 3
  use_dcn_in_tower: True

LearningRate:
  base_lr: 0.01
  schedulers:
  - !PiecewiseDecay
    gamma: 0.1
    milestones: [24, 33]
  - !LinearWarmup
    start_factor: 0.
    steps: 1000

```

配置来源2：入口配置文件的剩余部分

配置来源3：命令行：argumentparser

**优先级逐级上升。**

```yaml
1.命令行
2.入口除了_BASE_的部分
3._BASE_中的文件
```

`_BASE_`中的文件优先级：

```
  5.'../datasets/coco_instance.yml',
  4.'../runtime.yml',
  3.'_base_/solov2_r50_fpn.yml',
  2.'_base_/optimizer_1x.yml',
  1.'_base_/solov2_light_reader.yml',
```

**注意：** 修改参数时，修改最高优先级的配置文件，或者不知道有几个配置文件中共有某参数时（即不确定某参数的最高优先级配置文件），直接 `-o` 指定。

## [命令行的使用](https://blog.csdn.net/yugongpeng_blog/article/details/46693471)



### 1.在文件内模拟命令行

因为参数解析，缺省的是，sys.argv[1:] 所以：

```python
if not sys.argv[1:]:
    # # 接着训练版本
    args_list = f'-c {CONFIG_FILE} --use_vdl -o epoch=100 TrainReader.batch_size=4 -r {CHECKPOINT}'.split()
    args = parser.parse_args(args_list)
else:
    args = parser.parse_args()
```

### 2.`-o`  最高优先级

由于命令行具有最高优先级，所以 `-o` 可以覆盖配置文件中的所有指定参数。下面 `FLAGS.opt`  即 命令行 的 `-o` 接收的各个关键字参数。 

* 当配置文件中参数是一级参数 直接用参数名。如： `weights`  `epoch`

* 当配置文件中参数是二级参数 使用 `一级.二级` 的形式。 如：`TrainReader.batch_size`

  ```python
      FLAGS = parse_args()
  
      cfg = load_config(FLAGS.config)
      cfg['fp16'] = FLAGS.fp16
      cfg['fleet'] = FLAGS.fleet
      cfg['use_vdl'] = FLAGS.use_vdl
      cfg['vdl_log_dir'] = FLAGS.vdl_log_dir
      cfg['save_prediction_only'] = FLAGS.save_prediction_only
      cfg['profiler_options'] = FLAGS.profiler_options
      cfg['save_proposals'] = FLAGS.save_proposals
      cfg['proposals_path'] = FLAGS.proposals_path
      # FLAGS.opt 命令行 -o 接收的各个关键字参数
      # merge_config 只传入一个参数时，对 全局字典 global_dict 进行更新。
      merge_config(FLAGS.opt)
  ```

  

## labelme 转码coco数据集

训练过程中会读取 bbox 信息，如果没有bbox 信息，则对应的 instance 无效。 如果bbox 的坐标都为0，对应的instance也无效。

```python
for inst in instances:
    # check gt bbox
    if inst.get('ignore', False):
        continue
    if 'bbox' not in inst.keys():
        continue
    else:
        if not any(np.array(inst['bbox'])):
            continue
```

> #### [COCO数据集的标注格式](https://zhuanlan.zhihu.com/p/29393415)

自定义的数据集，实例分割的test.json 中，categories需加入一个Background，其他类别id依次+1。解决infer.py 类别预测与实际不对应的问题。（预测id是对的，是clsid2catid 的映射出了问题）

![image-20220127182832268](Readme_Lv.assets\image-20220127182832268.png)

##   实例分割数据集增广思路

1. **量**的扩大（数据集不是太多，需要增广）

2. 场景模拟。对手中的数据集经过数据增强后，达到模拟真实应用场景数据集的**质量**。

3. 加速。labelme数据集 转 coco 格式，用labelme自带的脚本速度有点慢。用paddle的脚本，出界框会报错。需要解决一下，来个并行化方案最好。

   加速方案初步：一个样本一个json，到最后来个合并json 的逻辑。分两步，可中断，可并行。

> > [传统分割方案哪里不可行](https://www.codenong.com/cs105341010/)
>
> 对于标注不全的数据，或者图像增强后出界的标注实例，该如何恰当处理。
>
> 案例一：左下角四本书显示不全，标注还是不标注。（达到某个阈值标还是？）
>
> 如果不标注，是直接当背景显示，还是用某种思路涂抹掉。
>
> ![image-20220127111124442](Readme_Lv.assets\image-20220127111124442.png)
>
> 案例二：图片上面的一排书显示不全，标还是不标。
>
> ![image-20220127112552052](Readme_Lv.assets\image-20220127112552052.png)
>
> ![image-20220127113956806](Readme_Lv.assets\image-20220127113956806.png)
>
> 案例三：背景的两排图书标还是不标，抹还是不抹。
>
> ![image-20220127111439111](Readme_Lv.assets\image-20220127111439111.png)
>
> 案例四：下①右侧三本，这种全乎的数据，要不要补标注，下②上面一排的标注要不要补，不补的话会对模型有怎样的影响。 下③属于标注疏漏了吧。
>
> ![image-20220127113345110](Readme_Lv.assets\image-20220127113345110.png)
>
> ![image-20220127113704194](Readme_Lv.assets\image-20220127113704194.png)
>
> ![image-20220127114119600](Readme_Lv.assets\image-20220127114119600.png)
>
> ![qfmCCf+8eHM](Readme_Lv.assets\qfmCCf+8eHM.jpg)
>
> 思路一:
>
> 涂抹掉标记区域之外的部分。（保留有三个以上点不出界的实例） 这样不影响模型对边界的检测能力（要不模型会因为在标注不全的部分存在边界，而标注信息没有标记为书籍，导致模型困惑，摇摆不定）。还有就是，有些漏标导致的模型困惑，使用这种涂抹策略应该会有所缓解。但是，这样训练的模型，在推理时，模型可能还是会探测到边界而去检测显示不全的实例。
>
> 思路二：
>
> 对已有样本补充标注，对增强数据集后显示不全的样本标注也保留。增加后处理手段，推断检测到的实例是否是完整实例。或者在网络中加入一个判断实例完整性的推理层。 这个完整性判断，可以考虑融入到seg的像素类别预测，当前book预测只有一个类别，修改为：1.完整 2.略微残缺，不影响识别 3.残缺严重等细粒度的类别。
>
> 思路三：
>
> 模型的调整，提高模型对于噪声数据的抗性。让模型，模拟人的学习方式，模型也不是所有知识都一股脑儿学到，而是在训练一段时间后，让模型拥有自己的判别能力，对数据信息选择性学习。  对于标注不好的地方，模型应有基本的判断，不刻意拟合标注不好的部分。
>
> 问题一：
>
> 自己在做透视变换的时候，areas需要同步变化，手动计算么。
>
> > 不需要，areas 是在转换coco格式时才有的。 labelme没有这个字段。



接下来的青蛙：

1. 数据集增强，拟真，扩大，裁切等等，数据集增广。（标注但残缺的实例，未标注但完整的实例，妥善解决）
2. 高速转码。

3. 寻找更快更准配置要求更少的分割/关键点检测模型。
