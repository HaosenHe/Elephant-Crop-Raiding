'''
Haosen He 2020

This file contains the abstract Elephant class and one of the implementations.
Note that for parameter tuning and modeling of short-term behaviors, the
encapsulation below is unnecessary.
'''
from algorithms import *
from visualization import *
from resources import *
import abc
import sys
import random


class Elephant(metaclass=abc.ABCMeta):
    '''
    This abstract class is the abstract elephant agent.
    '''
    walk = None

    def __init__(self, **kwargs):
        allowed_keys = {'crop_amount', 'tree_amount', 'step', 'trespass_cost', 'mg_tree',
                        'mg_crop', 'mg_water', 'habitat', 'alpha', 'moving_cost'}
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in allowed_keys)
        self.fitness = 0

    def update_resources(self):
        '''
        Update resources in the lattice
        '''
        if self.walk == None:
            sys.exit("Error: No available walk to forage.")
        for loc in self.walk:
            if loc in cropset:
                if self.crop_amount[loc] >= self.mg_crop:
                    self.crop_amount[loc] = self.crop_amount[loc] - self.mg_crop
                else:
                    self.crop_amount[loc] = 0
            elif loc in treeset:
                if self.tree_amount[loc] >= self.mg_tree:
                    self.tree_amount[loc] = self.tree_amount[loc] - self.mg_tree
                else:
                    self.tree_amount[loc] = 0

    def set_habitat(self, hab):
        '''
        Reset the habitat
        '''
        self.habitat = hab

    def forage(self):
    '''
    This method represent a night's forage.
    '''
    pass


class CRM_Elephant(Elephant):
    '''
    Implementatio of the CRM elephant agent
    '''

    def forage(self):
        '''
        This method represent a night's forage.
        '''
        self.walk = random.sample(Tsearch_walk(self.crop_amount, self.tree_amount,
                                               self.step, self.trespass_cost, self.mg_tree, self.mg_crop,
                                               self.mg_water, self.habitat, self.alpha, 1-self.alpha,
                                               self.moving_cost), 1)[0]
        self.update_resources(self)
        self.walk = None
