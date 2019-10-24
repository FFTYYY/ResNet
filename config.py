import argparse
import os
from utils.logger import Logger
from pprint import pformat

_par = argparse.ArgumentParser()

#---------------------------------------------------------------------------------------------------

#dataloader
_par.add_argument("--data" 			, type = str , default = "cifar-10" , choices = ["cifar-10"])
_par.add_argument("--data_load" 	, type = str , default = "datas/")
_par.add_argument("--data_save" 	, type = str , default = "processed_datas/")
_par.add_argument("--force_reprocess" 	, action = "store_true", default = False)
_par.add_argument("--smallize" 			, action = "store_true", default = False)

#model universal
_par.add_argument("--model" 		, type = str , default = "transformer" , choices = ["transformer"])

#model transformer
_par.add_argument("--d_model" 		, type = int , default = 512)
_par.add_argument("--d_hid" 		, type = int , default = 1024)
_par.add_argument("--d_posenc" 		, type = int , default = 512)
_par.add_argument("--num_layers" 	, type = int , default = 24)
_par.add_argument("--h" 			, type = int , default = 8)
_par.add_argument("--n_persis" 		, type = int , default = 512)
_par.add_argument("--drop_p" 		, type = float , default = 0.3)
_par.add_argument("--max_k" 		, type = float , default = 4)


#train & test
_par.add_argument("--batch_size" 	, type = int , default = 64)
_par.add_argument("--n_epochs" 		, type = int , default = 32)
_par.add_argument("--lr" 			, type = float , default = 1e-3)
_par.add_argument("--gpus" 			, type = str , default = "0")
_par.add_argument("--valid_size" 	, type = int , default = 1000)
_par.add_argument("--init_steps" 	, type = int , default = 0)
_par.add_argument("--step_size" 	, type = int , default = 1)
_par.add_argument("--valid_data" 	, type = str , default = "test")


#solely test
_par.add_argument("--model_load" 	, type = str , default = "")
	#example : best_DataParallel_acc_2019-10-24-01-09-53
_par.add_argument("--n_crop" 		, type = int , default = 0)

#others
_par.add_argument("--model_save" 	, type = str , default = "trained_models/")
_par.add_argument("--seed" 			, type = int , default = 2333)
_par.add_argument("--log_file" 		, type = str , default = "log.txt")

#---------------------------------------------------------------------------------------------------

C = _par.parse_args()

C.data_load = os.path.join(C.data_load , C.data)

C.gpus = [int(x) for x in filter(lambda x:x , C.gpus.strip().split(","))]

C.test_mode = False
if C.model_load:
	C.test_mode = True

if C.test_mode:
	C.log_file += ".test"

logger = Logger(C.log_file)
logger.log = logger.log_print_w_time

logger.log ("------------------------------------------------------")
logger.log (pformat(C.__dict__))
logger.log ("------------------------------------------------------")