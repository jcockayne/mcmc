import numpy as np


class DiskBackedStorage(object):
    def __init__(self, store_shape, file):
        try:
            import tables
        except:
            raise Exception('Use of DiskBackedStorage required PyTables to be installed (try pip install tables).')

        self.__store_shape__ = store_shape
        self.__array__ = np.empty(store_shape)
        self.__max_rows__ = store_shape[0]

        self.__current_index__ = 0

        if type(file) in (str, unicode):
            self.__f__ = tables.open_file(file, 'a')
        else:
            self.__f__ = file

        # check whether samples already exist in the file
        try:
            self.__f__.get_node('/samples')
            raise Exception('File already contains samples!')
        except:
            pass

        self.__dest_table__ = self.__f__.create_earray('/', 'samples', tables.Float64Atom(), (0, store_shape[1]))

    def add_sample(self, value):
        self.__array__[self.__current_index__, :] = value

        self.__current_index__ += 1
        if self.__current_index__ >= self.__max_rows__:
            self.__flush_to_disk__()
            self.__current_index__ = 0

    def __flush_to_disk__(self, up_to=-1):
        self.__dest_table__.append(self.__array__[:up_to, :])
        # dangerous not to reassign but let's assume we're doing this right.
        # self.__array__ = np.empty(self.__store_shape__)

    def close(self):
        self.__flush_to_disk__(self.__current_index__ + 1)
        self.__f__.close()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()
        return False
