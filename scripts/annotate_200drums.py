import numpy as np
import pandas as pd
import sys

from glob import glob
from pathlib import PurePath

INCLUSIONS = {
    'hhc': ['hhc', 'hh_c', 'hh c', 'hh-c', 'hatc', 'hat_c', 'hat c', 'hat-c',
            'clhh', 'cl_hh', 'cl hh', 'cl-hh', 'clhat', 'cl_hat', 'cl hat', 'cl-hat',
            'clhihat', 'closed hh', 'closed_hh', 'closed hi', 'closed_hi'],

    'hho': ['hho', 'hh_o', 'hh o', 'hh-o', 'hato', 'hat_o', 'hat o', 'hat-o',
            'ophh', 'op_hh', 'op hh', 'op-hh', 'ophat', 'op_hat', 'op hat', 'op-hat',
            'ophihat', 'open hh', 'open_hh', 'open hi', 'open_hi'],

    'kd': ['kick'],
    'sd': ['snare'],
}

def filter_name(path, inclusions):
    st = PurePath(path).stem.lower()
    for inc in inclusions:
        if inc in st:
            return True
    return False

def filter_name_all(paths, inclusions):
    return np.array([filter_name(p, inclusions) for p in paths])

def annotate_200drums(root_dir):
    root_dir = PurePath(root_dir)
    drums_names = np.array([
        PurePath(p).relative_to(root_dir)
        for p in glob(str(root_dir / 'drums/*/*.wav'))
    ])
    df = pd.DataFrame(columns=['file', 'class'])

    for cl, incs in INCLUSIONS.items():
        idxs = filter_name_all(drums_names, incs)
        paths = drums_names[idxs]
        print(f'class {cl}: {len(paths)} found')
        df = df.append(pd.DataFrame.from_dict({
            'file': paths,
            'class': [cl]*len(paths)
        }))

    df.to_csv(root_dir / 'annotation.csv')

if __name__ == '__main__':
    annotate_200drums(sys.argv[1])
