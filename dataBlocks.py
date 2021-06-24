# author is He Zhao
# The time to create is 2:18 PM, 7/12/16

import numpy as np
from multiprocessing import Process, Queue


class DataIterator(object):
    def __init__(self, *data, **params):
        '''
        PARAMS:
            fullbatch (bool): decides if the number of examples return after every
                              iteration should be always a full batch.
        '''
        self.data = data
        self.batchsize = params['batchsize']
        if 'fullbatch' in params:
            self.fullbatch = params['fullbatch']
        else:
            self.fullbatch = False

    def __iter__(self):
        self.first = 0
        return self

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, key):
        outs = []
        for val in self.data:
            outs.append(val[key])
        return self.__class__(*outs, batchsize=self.batchsize, fullbatch=self.fullbatch)


class SequentialIterator(DataIterator):
    '''
    batchsize = 3
    [0, 1, 2], [3, 4, 5], [6, 7, 8]
    '''
    def __next__(self):
        if self.fullbatch and self.first+self.batchsize > len(self):
            raise StopIteration()
        elif self.first >= len(self):
            raise StopIteration()

        outs = []
        for val in self.data:
            outs.append(val[self.first:self.first+self.batchsize])
        self.first += self.batchsize
        return outs


class StepIterator(DataIterator):
    '''
    batchsize = 3
    step = 1
    [0, 1, 2], [1, 2, 3], [2, 3, 4]
    '''
    def __init__(self, *data, **params):
        super(self, StepIterator).__init__(self, *data, **params)
        self.step = params['step']

    def __next__(self):
        if self.fullbatch and self.first+self.batchsize > len(self):
            raise StopIteration()
        elif self.first >= len(self):
            raise StopIteration()

        outs = []
        for val in self.data:
            outs.append(val[self.first:self.first+self.batchsize])
        self.first += self.step
        return outs


def np_load_func(path):
    arr = np.load(path)
    return arr


class DataBlocks(object):

    def __init__(self, data_paths, batchsize=32, load_func=np_load_func, allow_preload=False):
        """
        DESCRIPTION:
            This is class for processing blocks of data, whereby dataset is loaded
            and unloaded into memory one block at a time.
        PARAM:
            data_paths (list or list of list): contains list of paths for data loading,
                            example:
                                [f1a.npy, f1b.npy, f1c.npy]  or
                                [(f1a.npy, f1b.npy, f1c.npy), (f2a.npy, f2b.npy, f2c.npy)]
            load_func (function): function for loading the data_paths, default to
                            numpy file loader
            allow_preload (bool): by allowing preload, it will preload the next data block
                            while training at the same time on the current datablock,
                            this will reduce time but will also cost more memory.
        """

        assert isinstance(data_paths, (list)), "data_paths is not a list"
        self.data_paths = data_paths
        self.batchsize = batchsize
        self.load_func = load_func
        self.allow_preload = allow_preload
        self.q = Queue()


    def __iter__(self):
        self.files = iter(self.data_paths)
        if self.allow_preload:
            self.lastblock = False
            bufile = next(self.files)
            self.load_file(bufile, self.q)
        return self


    def __next__(self):
        if self.allow_preload:
            if self.lastblock:
                raise StopIteration

            try:
                arr = self.q.get(block=True, timeout=None)
                self.iterator = SequentialIterator(*arr, batchsize=self.batchsize)
                bufile = next(self.files)
                p = Process(target=self.load_file, args=(bufile, self.q))
                p.start()
            except:
                self.lastblock = True
        else:
            fpath = next(self.files)
            arr = self.load_file(fpath)
            self.iterator = SequentialIterator(*arr, batchsize=self.batchsize)

        return self.iterator


    def load_file(self, paths, queue=None):
        '''
        paths (list or str): []
        '''
        data = []
        if isinstance(paths, (list, tuple)):
            for path in paths:
                data.append(self.load_func(path))
        else:
            data.append(self.load_func(paths))
        if queue:
            queue.put(data)
        return data


    @property
    def nblocks(self):
        return len(self.data_paths)


class SimpleBlocks(object):

    def __init__(self, data_paths, batchsize=32, load_func=np_load_func, allow_preload=False):
        """
        DESCRIPTION:
            This is class for processing blocks of data, whereby dataset is loaded
            and unloaded into memory one block at a time.
        PARAM:
            data_paths (list or list of list): contains list of paths for data loading,
                            example:
                                [f1a.npy, f2a.npy, f3a.npy] ==> 1 col, 3 blocks or
                                [(f1a.npy, f1b.npy, f1c.npy), (f2a.npy, f2b.npy, f2c.npy)] ==> 3 cols, 2 blocks
            load_func (function): function for loading the data_paths, default to
                            numpy file loader
            allow_preload (bool): by allowing preload, it will preload the next data block
                            while training at the same time on the current datablock,
                            this will reduce time but will also cost more memory.
        """

        assert isinstance(data_paths, (list)), "data_paths is not a list"
        self.data_paths = data_paths
        self.batchsize = batchsize
        self.load_func = load_func
        self.allow_preload = allow_preload
        self.q = Queue()


    def __iter__(self):
        self.files = iter(self.data_paths)
        if self.allow_preload:
            self.lastblock = False
            bufile = next(self.files)
            self.load_file(bufile, self.q)
        return self


    def __next__(self):
        if self.allow_preload:
            if self.lastblock:
                raise StopIteration

            try:
                arr = self.q.get(block=True, timeout=None)
                self.iterator = SequentialIterator(*arr, batchsize=self.batchsize)
                bufile = next(self.files)
                p = Process(target=self.load_file, args=(bufile, self.q))
                p.start()
            except:
                self.lastblock = True
        else:
            fpath = next(self.files)
            arr = self.load_file(fpath)
            self.iterator = SequentialIterator(*arr, batchsize=self.batchsize)

        return self.iterator


    def load_file(self, paths, queue=None):
        '''
        paths (list or str): []
        '''
        data = []
        if isinstance(paths, (list, tuple)):
            for path in paths:
                data.append(self.load_func(path))
        else:
            data.append(self.load_func(paths))
        if queue:
            queue.put(data)
        return data


    @property
    def nblocks(self):
        return len(self.data_paths)


class DataBlocks(SimpleBlocks):

    def __init__(self, data_paths, train_valid_ratio=[5,1], batchsize=32, load_func=np_load_func, allow_preload=False):
        """
        DESCRIPTION:
            This is class for processing blocks of data, whereby dataset is loaded
            and unloaded into memory one block at a time.
        PARAM:
            data_paths (list or list of list): contains list of paths for data loading,
                            example:
                                [f1a.npy, f1b.npy, f1c.npy]  or
                                [(f1a.npy, f1b.npy, f1c.npy), (f2a.npy, f2b.npy, f2c.npy)]
            load_func (function): function for loading the data_paths, default to
                            numpy file loader
            allow_preload (bool): by allowing preload, it will preload the next data block
                            while training at the same time on the current datablock,
                            this will reduce time but will also cost more memory.
        """

        assert isinstance(data_paths, (list)), "data_paths is not a list"
        self.data_paths = data_paths
        self.train_valid_ratio = train_valid_ratio
        self.batchsize = batchsize
        self.load_func = load_func
        self.allow_preload = allow_preload
        self.q = Queue()


    def __next__(self):
        if self.allow_preload:
            if self.lastblock:
                raise StopIteration

            try:
                train, valid = self.q.get(block=True, timeout=None)
                self.train_iterator = SequentialIterator(*train, batchsize=self.batchsize)
                self.valid_iterator = SequentialIterator(*valid, batchsize=self.batchsize)
                bufile = next(self.files)
                p = Process(target=self.load_file, args=(bufile, self.q))
                p.start()
            except:
                self.lastblock = True
        else:
            fpath = next(self.files)
            train, valid = self.load_file(fpath)
            self.train_iterator = SequentialIterator(*train, batchsize=self.batchsize, fullbatch=True)
            self.valid_iterator = SequentialIterator(*valid, batchsize=self.batchsize, fullbatch=True)
        return self.train_iterator, self.valid_iterator


    def load_file(self, paths, queue=None):
        '''
        paths (list or str): []
        '''
        train = []
        valid = []
        if isinstance(paths, (list, tuple)):
            for path in paths:
                X = self.load_func(path)
                num_train = len(X) * self.train_valid_ratio[0] * 1.0 / sum(self.train_valid_ratio)
                num_train = int(num_train)
                train.append(X[:num_train])
                valid.append(X[num_train:])
        else:
            X = self.load_func(paths)
            # np.random.shuffle(X)
            num_train = len(X) * self.train_valid_ratio[0] * 1.0 / sum(self.train_valid_ratio)
            num_train = int(num_train)
            train.append(X[:num_train])
            valid.append(X[num_train:])
        data = [train, valid]
        if queue:
            queue.put(data)
        return data
