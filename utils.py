import os, sys


class Logger:
    def __init__(self, file):
        self.file = file

    def logging(self, *logs):
        if self.file:
            with open(self.file, 'a+') as f:
                for log in logs:
                    f.write(log)
                    f.write('\n')
        else:
            for log in logs:
                sys.stdout.write(log)
                sys.stdout.write('\n')


def makedirs(*dirs):
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)