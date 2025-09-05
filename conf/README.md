## 配置文件使用
配置文件的主要作用是在命令行中通过"rainfallmodel + 命令 + 配置文件"的方式来减少命令行长度，而且配置文件本身可以复用

使用范例:  
1).预训练使用范例:  
rainfallmodel pretrain "D:\Project\Py\RainFallModel\conf\pretrain_demo.yaml"  
2).推理使用范例:  
rainfallmodel infer "D:\Project\Py\RainFallModel\conf\infer_demo.yaml"   



配置文件目前仅支持yaml格式

## 配置文件的内部规范
1).配置文件在每个命令的入口总是被定义为user_conf  
2).然后具体每个模块会有默认配置，比如数据集会有default_dataset_conf，同名配置项会被user_conf覆盖 

## 配置文件的使用建议
1).用户在配置user_conf时还是建议保留原始数据类型，比如设置为True或者False时不加引号  
2).系统也有一定的兼容能力，比如learing_rate会被转为float类型，但是还是建议用户保留数据类型   
3).配置文件尽量不要加一些特殊字符，比如emoji表情，☺，以避免发生意想不到的情况  
4).配置文件统一使用utf-8格式，避免产生兼容性问题  
5).一般来说，除了资源部分必填之外，其他的通常都有默认值，因为资源缺失会导致没有数据执行 