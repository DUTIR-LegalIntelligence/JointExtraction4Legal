## Joint extraction for legal document

#### Introduction

Work for paper ***Joint Entity and Relation Extraction for Legal Documents with Legal Feature Enhancement***, at COLING2020.

The data is shown in "./data", and the model is defined in "joint_model.py". Hyper parameters are set in "my_config.ini".

#### Start

Training：`python main.py --config=./my_config.ini --srcpath=./data --trgpath=./data/test --mode=train`

Testing：`python main.py --config=./my_config.ini --srcpath=./data --trgpath=./data/test --mode=test`
