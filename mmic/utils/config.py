import munch

def parse_config(config_fname: str,
                 delimiter: str = '__',
                 strict_cast: bool = True,
                 verbose: bool = False,
                 **kwargs) -> munch.Munch:
    """Parse the given configuration file with additional options to overwrite.

    Parameters
    ----------
    config_fname: str
        A configuration file defines the structure of the configuration.
        The file should be serialized by any of [yaml, json, pickle, torch].

    delimiter: str, optional, default='__'
        A delimiter for the additional kwargs configuration.
        See kwargs for more information.

    strict_cast: bool, optional, default=True
        If True, the overwritten config values will be casted as the original type.

    verbose: bool, optional, default=False

    kwargs: optional
        If specified, overwrite the current configuration by the given keywords.
        For the multi-depth configuration, "__" is used for the default delimiter.
        The keys in kwargs should be already defined by config_fname (otherwise it will raise KeyError).
        Note that if `strict_cast` is True, the values in kwargs will be casted as the original type defined in the configuration file.

    Returns
    -------
    config: munch.Munch
        A configuration file, which provides attribute-style access.
        See `Munch <https://github.com/Infinidat/munch>`_ project for the details.

    Examples
    --------
    >>> # simple_config.json => {"opt1": {"opt2": 1}, "opt3": 0}
    >>> config = parse_config('simple_config.json')
    >>> print(config.opt1.opt2, config.opt3, type(config.opt1.opt2), type(config.opt3))
    2 1 <class 'int'> <class 'int'>

    >>> config = parse_config('simple_config.json', opt1__opt2=2, opt3=1)
    >>> print(config.opt1.opt2, config.opt3, type(config.opt1.opt2), type(config.opt3))
    2 1 <class 'int'> <class 'int'>

    >>> parse_config('test.json', **{'opt1__opt2': '2', 'opt3': 1.0})
    >>> print(config.opt1.opt2, config.opt3, type(config.opt1.opt2), type(config.opt3))
    2 1 <class 'int'> <class 'int'>

    >>> parse_config('test.json', **{'opt1__opt2': '2', 'opt3': 1.0}, strict_cast=False)
    >>> print(config.opt1.opt2, config.opt3, type(config.opt1.opt2), type(config.opt3))
    2 1.0 <class 'str'> <class 'float'>
    """
    config = _loader(config_fname, verbose)

    if kwargs:
        _print(f'overwriting configurations: {kwargs}', verbose)

    for arg_key, arg_val in kwargs.items():
        keys = arg_key.split(delimiter)
        n_keys = len(keys)

        _config = config
        for idx, _key in enumerate(keys):
            if n_keys - 1 == idx:
                if strict_cast:
                    typecast = type(_config[_key])
                    _config[_key] = typecast(arg_val)
                else:
                    _config[_key] = arg_val
            else:
                _config = _config[_key]

    config = munch.munchify(config)
    return config