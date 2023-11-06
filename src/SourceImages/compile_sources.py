
import os
import requests
import h5py
import numpy as np
import pandas as pd
import time


class SourceDataset(object):

    def __init__(self, name, path, sim='TNG50-1', ds_len=100, seed=int(time.time())):

        if seed is not None:
            np.random.seed(seed)
        self.name = name
        self.path = path
        self.api_key = os.environ.get('ILLUSTRISTNG_API_KEY', 'No key found')
        self.headers = {"api-key":self.api_key}
        self.sim = sim # Simulation to pull data from
        self.ds_len = ds_len
        self.df_snap_halo_ids = self._get_ids_per_snapshot()
        self.params = self._compile_params(self.ds_len)

    def __repr__(self):
        return f"SourceDataset(name={self.name}, path={self.path})"

    def __str__(self):
        return f"SourceDataset(name={self.name}, path={self.path})"
    
    def _generate_unique_array(self, rows, cols):
        arr = np.empty((rows, cols), dtype=int)
        
        for row in range(rows):
            a = np.random.randint(0, 3)
            b = a
            while b == a:
                b = np.random.randint(0, 3)
            arr[row, :] = [a, b]
        
        return arr
    
    def _print_keys(self, d, indent=0):
        for key in d.keys():
            print('\t'*indent + key)
            if isinstance(d[key], dict):
                self._print_keys(d[key], indent+1)

    def _dict_to_query(self, d):
        return '&'.join([key + '=' + str(d[key]) for key in d.keys()])

    def _get(self, path, params=None):
        # make HTTP GET request to path
        r = requests.get(path, params=params, headers=self.headers)

        # raise exception if response code is not HTTP SUCCESS (200)
        r.raise_for_status()

        if r.headers['content-type'] == 'application/json':
            return r.json() # parse json responses automatically
        return r
    
    def _get_ids_per_snapshot(self):

        print('Generating snap_ids.csv')

        snapshots = np.arange(33, 51, 1)

        snap_ids = pd.DataFrame(columns=['snap', 'halo_id_list'])

        snap_sample = np.random.uniform(33, 50, self.ds_len).astype(int)

        # get counts per snapshot
        snapshots, counts = np.unique(snap_sample, return_counts=True)

        urls = [f"http://www.tng-project.org/api/TNG100-1/snapshots/{snap}/subhalos/" for snap in snapshots]

        for i, url in enumerate(urls):
            halo = self._get(url, {'order_by': 'primary_flag', 'limit': counts[i]})
            ids = [d['id'] for d in halo['results'] if 'id' in d]
            snap_ids.loc[len(snap_ids)] = {'snap': snapshots[i], 'halo_id_list': ids}

        snap_ids.set_index('snap', inplace=True)

        snap_ids.to_csv('data/snap_ids.csv')

        return snap_ids
            
    def _request_to_numpy(self, request, worker_id):
        
        content = request.content

        with open(f'data/{worker_id}.hdf5', "wb") as f:
                f.write(content)

        if __name__ == '__main__':
            # Open the HDF5 file
            with h5py.File(f'data/{worker_id}.hdf5', 'r') as f:
                # Print the file structure
                galaxy_image = f['grid'][:]
        
        #Remove the file
        os.remove(f'data/{worker_id}.hdf5')

        return galaxy_image

    def _print_hdf5_file_structure(self, file, indent=0):
        """
        Recursively prints the structure of an HDF5 file, including datasets, groups, attributes, and datatypes.
        """
        for key in file.keys():
            if isinstance(file[key], h5py.Dataset):
                print(f"{' ' * indent}Dataset: {key} (shape={file[key].shape}, dtype={file[key].dtype})")
            elif isinstance(file[key], h5py.Group):
                print(f"{' ' * indent}Group: {key}")
                self._print_hdf5_file_structure(file[key], indent=indent+2)
            else:
                print(f"{' ' * indent}Unknown object type: {key}")

            # Print attributes for all objects
            for attr_key, attr_value in file[key].attrs.items():
                print(f"{' ' * (indent+2)}\u2514 Attribute: {attr_key}={attr_value}")
    
    def _compile_params(self, ds_len):
        
        print('Compiling params')

        exploded_df = self.df_snap_halo_ids.explode('halo_id_list')

        exploded_df.reset_index(inplace=True)

        snap_id_array = [exploded_df['halo_id_list'].values, exploded_df['snap'].values]

        axes = self._generate_unique_array(self.ds_len, 2)
        band = np.random.choice([
                                 'wfc3_uvis_f218w',
                                 'wfc3_uvis_f225w',
                                 'wfc3_uvis_f275w',
                                 'wfc3_uvis_f336w',
                                 'wfc3_uvis_f390w',
                                 'wfc3_uvis_f438w', 
                                 'wfc3_uvis_f475w',
                                 'wfc3_uvis_f555w', 
                                 'wfc3_uvis_f606w',
                                 'wfc3_uvis_f775w',  
                                 'wfc3_uvis_f814w', 
                                #  'wfc3_ir_f098w',
                                #  'wfc3_ir_f105w',
                                #  'wfc3_ir_f110w',
                                #  'wfc3_ir_f125w', 
                                #  'wfc3_ir_f140w',
                                #  'wfc3_ir_f160w',
                                 ], ds_len)
        
        params = {'sim':np.repeat(self.sim, self.ds_len),
                    'snapshot':snap_id_array[1],
                    'halo_id':snap_id_array[0],
                    'a1':axes[:,0],
                    'a2':axes[:,1],
                    'band':band,
                    'partType':np.repeat('stars', self.ds_len),
                    'partField':[f'stellarBandObsFrame-{band}' for band in band],
                    'size':np.repeat('100', self.ds_len),
                    'sizeType':np.repeat('kpc', self.ds_len),
                    'nPixels':np.repeat('2000', self.ds_len),
                    'axes':[f"{axes[0]},{axes[1]}" for axes in axes],
                    }
        
        # Convert to dataframe
        params = pd.DataFrame(params)
        
        return params

    def get_images(self):

        print('Getting images')

        urls = [f"http://www.tng-project.org/api/{row['sim']}/snapshots/{row['snapshot']}/subhalos/{row['halo_id']}" for _, row in self.params.iterrows()]
        vis_queries = [self._dict_to_query({'partType':row['partType'],
                                            'partField':row['partField'],
                                            'size':row['size'],
                                            'sizeType':row['sizeType'],
                                            'nPixels':row['nPixels'],
                                            'axes':row['axes']}) for _, row in self.params.iterrows()]


        # Combine the urls and queries
        vis_endpoints = [url + '/vis.hdf5?' + query for url, query in zip(urls, vis_queries)]

        # Run the get request and track the progress
        halo_vis = []

        for endpoint in vis_endpoints:
            print(f"Requesting {endpoint}")
            halo_vis.append(self._get(endpoint))

        galaxy_images = []

        for i in range(len(halo_vis)):
            galaxy_images.append(self._request_to_numpy(halo_vis[i], i))

        return np.array(galaxy_images)

    def get_metadata(self):

        urls = [f"http://www.tng-project.org/api/{row['sim']}/snapshots/{row['snapshot']}/halos/0/" for _, row in self.params.iterrows()]
        
        url = "http://www.tng-project.org/api/TNG100-1/snapshots/50/subhalos/?limit=1000"

        url = "http://www.tng-project.org/api/TNG100-1/snapshots/50/subhalos/0/"

        response = self._get(url)
        
        for key in response.keys():
            print(key,': ', response[key])

    def get_params(self):
        return self.params

source_ds = SourceDataset('Source_Dataset', 'data/', ds_len=10)
source_images = source_ds.get_images()

TIME = str(int(time.time()))

np.save(f'data/source_images_{TIME}.npy', source_images)


