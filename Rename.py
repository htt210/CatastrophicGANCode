import os
import argparse
import re


def rename(folder):
    try:
        newfolder = re.sub(r'[_ ]+', r'-', folder)
        os.rename(folder, newfolder)
        files = os.listdir(newfolder)
        for f in files:
            fpath = newfolder + '/' + f
            if os.path.isdir(fpath):
                rename(fpath)
            else:
                newfpath = re.sub(r'[_ ]+', r'-', fpath)
                os.rename(fpath, newfpath)
                print(fpath)
    except Exception as e:
        print(e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dir', default='/home/thanhtung/github/figs/aistats2020-figs/figs/')
    args = parser.parse_args()

    rename(args.dir)
