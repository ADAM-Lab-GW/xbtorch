# __init__.py
import xbtorch

class XBParams:
    '''
    Singleton class
    '''
    _instance = None
    _global_dict = {'initialized': False,
                    'decomposition_algorithm': None,
                    }

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(XBParams, cls).__new__(cls)
        return cls._instance

    def set_var(self, key, value):
        self._global_dict[key] = value

    def get_var(self, key, default=None):
        return self._global_dict.get(key, default)
    
    def initialize(self, decomposition_algorithm, device_type, weight_range=(-1, 1)):
        print('\nInitializing XBTorch..')
        # Type checking
        if decomposition_algorithm and not issubclass(type(decomposition_algorithm), xbtorch.decomposition.base.GenericDecomposition):
            raise TypeError("Invalid decomposition algorithm provided")
        if device_type and not issubclass(type(device_type), xbtorch.devices.base.GenericDevice):
            raise TypeError("Invalid device type provided")
        if weight_range and not (type(weight_range) == tuple or len(weight_range) != 2 or weight_range[0] >= weight_range[1]):
            raise TypeError("Invalid weight range provided")
        # Setting the variables
        self._global_dict['initialized'] = True
        self._global_dict['decomposition_algorithm'] = decomposition_algorithm
        self._global_dict['device_type'] = device_type
        self._global_dict['weight_range'] = weight_range

        print('Initialization complete..\n')

def get_xbtorch_param(key, default=None):
    return XBParams().get_var(key, default)

def initialize(*args, **kwargs):
    XBParams().initialize(*args, **kwargs)

__all__ = ['initialize']