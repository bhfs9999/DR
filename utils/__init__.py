
def read_fname(fpath):
    fnames = []
    with open(fpath, 'r') as f:
        for one_line in f:
            fname = one_line.strip().split('\t')[0]
            cls   = one_line.strip().split('\t')[1]
            if cls != '0':
               fnames.append(fname)
    return fnames

