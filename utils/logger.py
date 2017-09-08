# A simple torch style logger
# (C) Wei YANG 2017

# Modified by Shi Qiu
#   8/29/2017

from __future__ import absolute_import
import matplotlib.pyplot as plt
import os
import sys
import numpy as np

__all__ = ['Logger', 'LoggerMonitor', 'savefig']

def savefig(fname, dpi=None):
    dpi = 150 if dpi == None else dpi
    plt.savefig(fname, dpi=dpi)

class Logger(object):
    '''Save training process to log file with simple plot function.
        Metrics for 'train' and 'val' phases are stored separately
    '''
    def __init__(self, fpath, title=None, names=[], resume=True, log_interval=20, print_to_screen=False):
        self.log_dict = {'train': {}, 'val': {}}
        self.log_interval = log_interval
        self.print_to_screen = print_to_screen
        self.title = '' if title == None else title
        self.file = None

        if fpath is not None:
            if resume and os.path.exists(fpath):
                self.load_logs(fpath)
                self.file = open(fpath, 'a')
            else:
                self.file = open(fpath, 'w')
                self.set_names(names)

    # load previous logs
    def load_logs(self, fpath):
        self.file = open(fpath, 'r')

        # skip leading raw strings
        line = self.file.readline()
        while line.startswith('## '):
            line = self.file.readline()

        self.names = line.strip().split()[1:]
        for name in self.names:
            self.log_dict['train'][name] = []
            self.log_dict['val'][name] = []

        for line in self.file:
            if line.startswith('## '):
                continue
            strs = line.strip().split()
            phase = strs[0][1:-1]
            for idx, s in enumerate(strs[1:]):
                self.log_dict[phase][self.names[idx]].append(float(s))

        self.file.close()


    def set_names(self, names):
        # initialize numbers as empty list
        self.names = names
        self.file.write('phase\t')
        for _, name in enumerate(self.names):
            self.file.write(name + '\t')
            self.log_dict['train'][name] = []
            self.log_dict['val'][name] = []
        self.file.write('\n')
        self.file.flush()


    def append(self, phase, numbers):
        assert len(self.names) == len(numbers), 'Numbers do not match names'
        str_to_write = '[%s]\t'.ljust(7) %(phase)
        self.file.write(str_to_write)
        if self.print_to_screen:
            print str_to_write,
        for index, num in enumerate(numbers):
            if index == 0:
                str_to_write = "%7d\t" %(num)
            else:
                str_to_write = "%.3f\t" %(num)
            self.file.write(str_to_write)
            if self.print_to_screen:
                print str_to_write,
            self.log_dict[phase][self.names[index]].append(num)

        self.file.write('\n')
        self.file.flush()
        if self.print_to_screen:
            print '\n',

    def append_raw(self, s):
        for line in s.split('\n'):
            output_s = '## {} \n'.format(line)
            self.file.write(output_s)
            self.file.flush()
            if self.print_to_screen:
                print output_s,

    def plot(self, names=None):
        names = self.names[1:] if names == None else names
        legend = []
        for phase in self.log_dict.keys():
            for name in names:
                x = self.log_dict[phase][self.names[0]]
                plt.plot(x, self.log_dict[phase][name])
                legend.append(self.title + '-[' + phase + ']-' + name)
        plt.legend(legend)
        plt.grid(True)
        plt.show(block=False)
        return legend

    def close(self):
        if self.file is not None:
            self.file.close()

class LoggerMonitor(object):
    '''Load and visualize multiple logs.'''
    def __init__ (self, logs):
        '''logs is a distionary with {name:filepath} pair'''
        self.loggers = []
        for title, path in logs.items():
            logger = Logger(path, title=title, resume=True)
            self.loggers.append(logger)

    def plot(self, names=None, separate_windows=True):
        plt.figure()
        if not separate_windows:
            legend_text = []
            for logger in self.loggers:
                legend_text += logger.plot(names)
            plt.legend(legend_text)
        else:
            row = (len(self.loggers) + 1) / 2
            for i, logger in enumerate(self.loggers):
                print i
                plt.subplot(row, 2, i + 1)
                logger.plot(names)
        plt.show()


if __name__ == '__main__':
    # # Example
    # logger = Logger('test2.txt',
    #                 title='Test',
    #                 names=['iter', 'loss', 'accuracy'],
    #                 resume=False,
    #                 print_to_screen=True
    #                 )

    # length = 100
    # t = np.arange(length)
    # train_loss = np.exp(-t / 5.0) + np.random.randn(length) * 0.1
    # train_acc = 1 - np.exp(-t / 5.0) + np.random.randn(length) * 0.1
    # test_loss = np.exp(-t / 10.0) + np.random.randn(length) * 0.1
    # test_acc = 1 - np.exp(-t / 10.0) + np.random.randn(length) * 0.1

    # for i in range(0, length):
    #     logger.append('train', [i, train_loss[i], train_acc[i]])
    #     if i % 10 == 0:
    #         logger.append('val', [i, test_loss[i], test_acc[i]])

    # logger.plot()

    # Example: logger monitor
    logs = {
    #'NetA': 'checkpoint/fashion_NetA/log.txt',
    'NetB': 'checkpoint/fashion_NetB_nopadding/log.txt',
    'NetC': 'checkpoint/fashion_NetC/log.txt',
    'NetD': 'checkpoint/fashion_NetD/log.txt'
    }

    fields = ['acc_top1', 'acc_top5']

    monitor = LoggerMonitor(logs)
    monitor.plot(names=fields, separate_windows=True)
