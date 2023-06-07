import json
import numpy as np
import nibabel
from os import path
from random import shuffle, seed as __seed__
from scipy import stats
import random
import time
from lib.image.tools import resize_images, central_crop
from typing import List

from lib.misc.args import CustomUserModule

biobank_list_path = '../private/biobank.json'
basepath = path.dirname(__file__)
biobank_list_path = path.abspath(path.join(basepath, biobank_list_path))

def remainders(num: int):
    if num % 2 == 0:
        return int(num/2), int(num/2)
    else:
        return int((num+1)/2), int((num-1)/2)




class Dataset(CustomUserModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config.batch_size = self.module_arg(
            'batch_size', false_val=[64, 64, 64])
        print("Batch size: {}".format(self.config.batch_size))
        self.config._3d_Nslices = self.module_arg('Nslices', false_val=None)
        self.config.downsample_resolution = self.module_arg(
            'downsample_resolution', false_val=None, instance_check=int)
        self.config.central_crop_size = self.module_arg(
            'central_crop_size', false_val=None, instance_check=int)
        self.config.num_test_files = self.module_arg(
            'num_test_files', false_val=None)
        self.config.reduced_validation = self.module_arg(
            'reduced_validation', false_val=True)

        self.debug = self.module_arg('debug', false_val=False)

        '''
        Select folds and fold number
        '''
        if 'cv_folds' in kwargs.keys():
            self.config.cv_folds = kwargs['cv_folds']
            if 'cv_fold_number' in kwargs.keys():
                self.config.cv_fold_num = kwargs['cv_fold_number']
            else:
                self.config.cv_fold_num = 1
        else:
            self.config.cv_fold_num = 1
            self.config.cv_folds = 3

        self.default_dimensions = [256, 256]
        
        
        
        
        self.config.epochs = None # inf

        self.current = self.Config()
        self.current.epoch = -1
        self.current.step = None
        self.current.file = 0
        self.generator = self.Config()
        
        self.init_rng = np.random.default_rng(1117)
        self.rng = np.random.default_rng(1117)


    def gen_int(self, min, max, N=None):
        '''
        inclusive of min and max
        '''
        vs = []
        N_ = 1 if N is None else N
        for _ in range(N_):
            vs.append(
                int(
                    self.rng.uniform(low=min, high=max+1)
                )
            )
        if N is None:
            return vs[0]
        else:
            return vs

    def normalise(self, input_image):
        """[summary]

        Args:
            input_image ([type]): [description]

        Returns:
            [type]: [description]
        """
        clip_min = 0
        #clip_max = tfp.stats.percentile(input_image, 99)
        clip_max = np.percentile(input_image, 99)
        out = np.clip(
            input_image,
            clip_min,
            clip_max
        )
        norm = (clip_max - clip_min)

        return (out - clip_min) / norm

    def resize(self, input_image, gen_ints=None, ret_ints=False,
               ignore_slicing=False, default_dimensions=None, validation=False):
        s = input_image.shape  # [H, W, Slice, Time]

        t0 = time.time()
        _3d_Nslices = self.config._3d_Nslices
        if validation and self.config.reduced_validation:
            _3d_Nslices = 1

        l = [self.gen_int(0, s[2]-1, N=_3d_Nslices)
             ] if gen_ints is None else gen_ints
        if validation:
            l = [self.validation_slice_number for _ in l]
        l = np.asarray(l).flatten()
        image = input_image[:, :, l]
        t = time.time() - t0

        if self.debug:
            print("Time to slice: {:.2f} seconds".format(t))

        new_s = image.shape
        if default_dimensions is not None:
            t_s = default_dimensions
        else:
            t_s = self.default_dimensions
        
        if self.debug:
            print("Resize dimensions of image {}".format(new_s))
            print("Resize default dimensions of image {}".format(t_s))
        '''
        Pad
        '''
        pads = [[0, 0] for _ in range(len(new_s))]
        for dim in range(len(t_s)):
            if new_s[dim] < t_s[dim]:
                diff = -(new_s[dim] - t_s[dim])
                ll, r = remainders(diff)
                pads[dim][0] = ll
                pads[dim][1] = r

        t0 = time.time()
        image = np.pad(image, pads)
        t = time.time() - t0
        if self.debug:
            print("Time to pad: {:.2f} seconds".format(t))

        '''
        Crop
        '''
        slice_begins = []
        to_slice = False
        for dim in range(len(t_s)):
            if new_s[dim] > t_s[dim]:
                diff = new_s[dim] - t_s[dim]
                ll, _ = remainders(diff)
                slice_begins.append(ll)
                to_slice = True
            else:
                slice_begins.append(0)
        if to_slice:
            image = slicing(image, slice_begins, self.default_dimensions)

        if self.config.downsample_resolution is not None:
            r = self.config.downsample_resolution
            image_ = np.expand_dims(image, axis=0)
            image_ = resize_images(image_, r, r)
            image = np.squeeze(image_, axis=0)

        if self.config.central_crop_size is not None:
            # [H, W, Slice, Time]
            image = central_crop(
                image, self.config.central_crop_size, images_2d=True,
                auto_shape=not(ignore_slicing))

        if self.debug:
            print("End resize shape: {}".format(image.shape))

        return image

    '''
    Read the Dataset information
    '''

    def import_list(self, list_path, skip_files_check=False):
        self.file_list = json.load(open(list_path, 'rb'))
        '''
        Remove files that do not exist
        '''
        if skip_files_check is False:
            list_orig = self.file_list[:]
            self.file_list_without_Nones = [
                x for x in self.file_list if path.isfile(x) is True]
            self.file_list = [x
                              if path.isfile(x) is True else None
                              for x in self.file_list]
            if set(list_orig) != set(self.file_list_without_Nones):
                print(
                    """BioBank filelist is not up-to-date.
                    Saving new list to biobank.update.json. Please update
                    by renaming it to biobank.json in the {} folder""".format(
                        basepath
                    )
                )
                new_bb_filepath = '../private/biobank.update.json'
                with open(path.join(basepath, new_bb_filepath), 'w') as f:
                    json.dump(self.file_list_without_Nones, f, indent=4)
        else:
            self.file_list_without_Nones = self.file_list
        '''
        Set the number of files available to the generator
        '''
        self.generator.num_files = len(self.file_list_without_Nones)

    def shuffle_and_split(self, cv_folds=3, cv_fold_num=1):
        '''
        Shuffle the imported file list and set the cross-validation
        folds by determining which files belong in which fold

        Also remove "None" values after folds created.

        (We perform "after" to keep the approximate data distribution in
        each fold approximately the same in the event that a data record
        is removed from the BioBank dataset)
        '''

        '''
        # Files
        '''
        n = len(self.file_list)
        self.init_rng.shuffle(self.file_list)
        n_ = np.int(n/cv_folds)

        '''
        Fold Sizing
        '''
        fold_remainder = n-((cv_folds-1)*n_)
        fold_sizes = [n_]*(cv_folds-1) + [fold_remainder]

        '''
        File positions for Fold-starts
        '''
        fold_start_positions = [
            0] + [np.sum(fold_sizes[0:i+1]) for i in range(cv_folds-1)]
        fold_end_positions = [fold_start_positions[i] + fold_sizes[i]
                              for i in range(cv_folds)]

        '''
        Create folds
        '''
        file_list = self.file_list[:]
        folds = [
            file_list[fold_start_positions[i]: fold_end_positions[i]]
            for i in range(cv_folds)]

        '''
        Create file lists
        '''
        self.test_file_list = folds[cv_fold_num-1]
        train_folds = folds[:]
        del(train_folds[cv_fold_num-1])
        self.train_file_list = [tfile
                                for fold_files in train_folds
                                for tfile in fold_files]
        self.validation_file_list = [self.test_file_list[0]]

        '''
        Remove deleted/'None' Files
        '''
        self.train_file_list = [
            x for x in self.train_file_list if x is not None]
        self.test_file_list = [x for x in self.test_file_list if x is not None]
        self.validation_file_list = [
            x for x in self.validation_file_list if x is not None]

        '''
        Remove invalid files
        '''
        for invalid_file in self.invalid_files:
            self.train_file_list = [
                x for x in self.train_file_list if invalid_file not in x]
            self.test_file_list = [
                x for x in self.test_file_list if invalid_file not in x]
            self.validation_file_list = [
                x for x in self.validation_file_list if invalid_file not in x]

        '''
        Reduced validation dataset
        '''
        if self.config.reduced_validation:
            print("Using reduced validation dataset (Val Size: 1)")
            self.validation_file_list = self.validation_file_list[0:1]

        '''
        Dataset length
        '''
        if self.config.num_test_files is not None:
            self.test_file_list = self.test_file_list[0:self.config.num_test_files]
        self.train_dataset_length = len(self.train_file_list)
        self.test_dataset_length = len(self.test_file_list)
        self.validation_dataset_length = len(self.validation_file_list)

        '''
        Modify data length for Nslices and Nframes
        '''
        
        if self.config._3d_Nslices is not None:
            nslices = self.config._3d_Nslices
        else:
            nslices = 1

        n = nslices
        self.train_dataset_length = n*self.train_dataset_length
        self.test_dataset_length = n*self.test_dataset_length
        self.validation_dataset_length = n*self.validation_dataset_length

        '''
        Perception steps
        '''
        batch_sizes = self.get_batch_sizes()
        self.train_dataset_steps = int(
            self.train_dataset_length / batch_sizes[0])
        self.test_dataset_steps = int(
            self.test_dataset_length / batch_sizes[2])
        self.validation_dataset_steps = int(
            self.validation_dataset_length / batch_sizes[1])

        '''
        Set 'num_files'
        '''
        self.generator.num_files = len(self.train_file_list)
        self.generator.num_test_files = len(self.test_file_list)

    def __config__(self):
        '''
        This is executed when create() is called. It is the first method
        executed in create()
        '''
        # if self.config.batch_size != 1:
        #    raise ValueError('This BioBank dataset only' +\
        #        ' supports batch size 1 due to images being different sizes')
        #    printt("Note: batching along the slice axis", warning=True)
        self.import_list(skip_files_check=not(self.file_check))
        self.shuffle_and_split(cv_folds=self.config.cv_folds,
                               cv_fold_num=self.config.cv_fold_num)

    def generator_skip(self, steps, current_file, epoch):
        self.current.step = steps
        self.current.file = current_file
        self.current.epoch = epoch

    def get_batch_sizes(self):
        '''
        Interprets the user specific self.config.batch_size variable
        and then return a list with three elements:
        [train_batch_size, validation_batch_size, test_batch_size]
        '''
        if isinstance(self.config.batch_size, list):
            batch_sizes = self.config.batch_size
        elif isinstance(self.config.batch_size, dict):
            batch_sizes = [
                self.config.batch_size["train"],
                self.config.batch_size["validation"],
                self.config.batch_size["test"]
            ]
        else:
            batch_sizes = [self.config.batch_size]*3
        return batch_sizes



def slicing(image_, slice_begins: List[int], target_dim: List[int]):
    image = image_
    N_ = len(image.shape)
    assert(len(slice_begins) == len(target_dim))
    N = min(N_, len(slice_begins))
    for i in range(N):
        forward = list(range(N_))
        del(forward[i])
        forward = [i] + forward
        image = np.transpose(image, forward)

        image = image[
            slice_begins[i]:slice_begins[i]+target_dim[i]
        ]

        backward = list(range(1, N_))
        backward.insert(i, 0)
        image = np.transpose(image, backward)
    return image
