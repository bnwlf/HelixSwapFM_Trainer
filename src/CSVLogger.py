import csv
import torch
from composer.loggers import LoggerDestination

class TrainLogEntry:
	def __init__(self,step):
		self.step = step
		self.values = {"step":step}

	@property
	def is_eval_step(self):
		return 'metrics/eval/LanguageCrossEntropy' in self.values

	def add_values(self, values):
		for key, value in values.items():
			if isinstance(value, torch.Tensor):
				value = value.item()
			self.values[key] = value



class CSVLogger(LoggerDestination):

    def __init__(
        self,
        train_filename="train_metrics.csv",
        eval_filename="eval_metrics.csv",
    ):

        print("CSVLogger:  writing to ->", train_filename)
        print("CSVLogger:  writing to ->", eval_filename)

        self.train_file = open(train_filename, "w", newline="")
        self.eval_file = open(eval_filename, "w", newline="")


        self.__train_fields = ["step",'time/total','metrics/train/LanguageCrossEntropy','metrics/train/MaskedAccuracy', 'time/batch','time/sample','time/batch_in_epoch','time/sample_in_epoch','time/token','time/token_in_epoch','trainer/device_train_microbatch_size','loss/train/total','time/train','time/val','lr-DecoupledAdamW/group0']
        self.__eval_fields =  ["step",'time/total','metrics/eval/LanguageCrossEntropy','metrics/eval/MaskedAccuracy']
        self.train_writer = csv.DictWriter(self.train_file, fieldnames=self.__train_fields, extrasaction="ignore")
        self.train_writer.writeheader()

        self.eval_writer = csv.DictWriter(self.eval_file, fieldnames=self.__eval_fields,extrasaction="ignore")
        self.eval_writer.writeheader()
        self.__current_logelem__ = TrainLogEntry(0)
        print("Init CSVLogger")


    def log_metrics(self, metrics: dict, step: int | None = None):
        if (self.__current_logelem__.step < step):
            print("CSVLoggernew step",step)
            self.train_writer.writerow(self.__current_logelem__.values)
            self.train_file.flush()
            if self.__current_logelem__.is_eval_step:
                self.eval_writer.writerow(self.__current_logelem__.values)
                self.eval_file.flush()
            self.__current_logelem__ = TrainLogEntry(step)

        self.__current_logelem__.add_values(metrics)


    def log_traces(self, traces: dict):
        pass

    def log_hyperparameters(self, hyperparameters: dict):
        pass

    def close(self):
        print("CSVLOGGER_close")
        self.train_file.close()
        self.eval_file.close()
