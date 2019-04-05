import tensorflow as tf
import numpy as np


color2code = dict(
    warning='\033[1;41m',
    ok='\033[1;32;40m',
    info='\033[1;36;40m'
)


def log(str="", color="info"):
    col = color2code[color]
    end = "\033[0m" + "\n"
    dash = '-'*60
    print("\n" + col + dash)
    print(str)
    print(dash + end)

def logStr(str=""):
    col = color2code['ok']
    end = "\033[0m"
    print(col + str + end)


class Logger():

    def __init__(self):

        self.metrics = dict()
        self.summary_writer = tf.summary.create_file_writer('./tmp/summaries')


    def store(self, name, value):

        if name not in self.metrics.keys():
            self.metrics[name] = tf.keras.metrics.Mean(name=name)
        self.metrics[name].update_state(value)


    def log_metrics(self, step):

        log('MEAN METRICS START', color="ok")

        if not self.metrics:
            logStr('NO METRICS')
        else:
            for key, metric in self.metrics.items():
                value = metric.result()
                logStr('{:<10s}{:>10.5f}'.format(key, value))
                metric.reset_states()

                with self.summary_writer.as_default():
                    tf.summary.scalar(key, value, step=step)
        
        log('MEAN METRICS END', color="ok")


       
                

        




