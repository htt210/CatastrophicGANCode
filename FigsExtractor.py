import re
import argparse
import os
import shutil


def parse(line):
    start = line.find('includegraphics')
    open_brace = 0
    open_brack = True
    for i in range(start, len(line)):
        if line[i] == '[':
            open_brack = True
        if line[i] == ']':
            open_brack = False
        if open_brack:
            continue
        if line[i] == '{':
            open_brace = i + 1
        elif line[i] == '}':
            return line[open_brace:i]
    return None


def copy_file(file_path, prefixes, suffixes, dest):
    for prefix in prefixes:
        if prefix is None:
            continue

        for suffix in suffixes:
            full_path = prefix + '/' + file_path + suffix
            if os.path.exists(full_path):
                subdirs = file_path.split('/')
                tempdir = dest
                for subdir in subdirs[:-1]:
                    if not os.path.exists(tempdir + '/' + subdir):
                        os.mkdir(tempdir + '/' + subdir)
                    tempdir += '/' + subdir
                shutil.copyfile(full_path, dest + '/' + file_path + suffix)
                print('success')
                return
    print('fail', file_path, suffixes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-file', default='/home/thanhtung/Dropbox/CatastrophicGAN/AISTATS2020-Tung/aistats_arxiv.tex',
                        help='input file')
    parser.add_argument('-prefixes',
                        default='/home/thanhtung/github/figs/aistats2020-figs/figs/ /home/thanhtung/Dropbox/CatastrophicGAN/',
                        help='prefixes')
    parser.add_argument('-suffixes', default='.pdf .png')
    parser.add_argument('-dest', default='figs/', help='destination folder')

    args = parser.parse_args()

    args.prefixes = args.prefixes.split()
    print(args.prefixes)
    args.suffixes = args.suffixes.split() + ['']

    if not os.path.exists(args.dest):
        os.mkdir(args.dest)

    with open(args.file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line.startswith('%'):
                continue
            if 'includegraphics' in line:
                file_path = parse(line)
                if file_path is not None:
                    copy_file(file_path, args.prefixes, args.suffixes, args.dest)
