## Joint extraction for legal document

#### 说明

进行法律文本实体关系联合提取，数据在./data文件夹中，主函数为main.py，模型为joint_model.py，超参数设置文件为my_config.ini。

#### 运行

训练：`python main.py --config=./my_config.ini --srcpath=./data --trgpath=./data/test --mode=train`

测试：`python main.py --config=./my_config.ini --srcpath=./data --trgpath=./data/test --mode=test`