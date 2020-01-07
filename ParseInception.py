import numpy as np
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-path', type=str, default='inception.txt', help='path to file')
    args = parser.parse_args()

    with open(args.path, 'r') as f:
        lines = f.readlines()
        models = {}
        for line in lines:
            mni = line.index('/2019') + 1
            model_name = line[:mni]
            msi = line.index(' (') + 2
            model_score = float(line[msi:msi+5])
            value = models.get(model_name, None)
            if value is None:
                models[model_name] = [model_score]
            else:
                value.append(model_score)
    print(len(models))
    for k, v in models.items():
        print(k, np.mean(v), np.var(v))
