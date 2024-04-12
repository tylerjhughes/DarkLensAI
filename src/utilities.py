import yaml
import collections.abc

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