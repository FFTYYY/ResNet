from utils.watch_time import time_str

class Logger:
	def __init__(self , fil_path):
		if not fil_path:
			self.log_fil = None
		else:
			self.log_fil = open(fil_path , "w")
	def only_log(self , cont = ""):
		if self.log_fil is not None:
			self.log_fil.write(str(cont) + "\n")

	def log_print(self , cont = ""):
		if self.log_fil is not None:
			self.log_fil.write(str(cont) + "\n")
		print (cont)
	def log_print_w_time(self , cont = ""):
		self.log_print(str(cont) + " | " + time_str())

	def close(self):
		self.log_fil.close()