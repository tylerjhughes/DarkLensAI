import yaml
import collections.abc
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class YamlEditor(object):

    '''
    This class contains functions to update yaml files with new inputs.
    '''

    def __init__(self, config_file_path):
        self.config_file_path = config_file_path


    def _find(self, key, dictionary):

        for k, v in dictionary.items():
            if k == key:
                yield v
            elif isinstance(v, dict):
                for result in self._find(key, v):
                    yield result

    def _update(self, d, u):
        for k, v in u.items():
            if isinstance(v, collections.abc.Mapping):
                d[k] = self._update(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    def update_yaml(self, input):

        '''
        This function takes a yaml file and a dictionary of inputs and updates the yaml file with the new inputs.

        Parameters
        ----------
        input : dict
            If the parameter is nested in the yaml file, the dictionary should be nested as well. 
        '''
        with open(self.config_file_path, 'r') as stream:
            #try:
            data = yaml.safe_load(stream)
            #except yaml.YAMLError as exc:
            #    print(exc)

        data = self._update(data, input)

        with open(self.config_file_path, 'w') as outfile:
            yaml.dump(data, outfile, default_flow_style=False)

def plot_training_history(job_dir, log_file_name='trainingLog.log', max_epochs=None, vmax=None):
    
        '''
        This function plots the training history of a model.

        Parameters
        ----------
        job_dir : str
            The directory where the log file is located.
        log_file_name : str
            The name of the log file.
        '''
    
        log = pd.read_csv(job_dir + '/' + log_file_name)
    
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    
        ax[0].plot(log['loss'], label='train')
        ax[0].plot(log['val_loss'], label='val')
        ax[0].set_title('Loss')
        ax[0].legend()
        
        if np.max(log['val_loss'])/np.mean(log['val_loss']) > 2:
            ax[0].set_ylim([0, np.percentile(log['val_loss'], 90)])
    
        ax[1].plot(log['accuracy'], label='train')
        ax[1].plot(log['val_accuracy'], label='val')
        ax[1].set_title('Accuracy')
        ax[1].legend()

        if max_epochs:
            ax[0].set_xlim([0, max_epochs])
            ax[1].set_xlim([0, max_epochs])

        ax[1].set_ylim([0.45, 1])
    
        plt.savefig(job_dir + '/training_history.png')

        return fig
