import pickle


def save(data, fname):
    """ Save data to specific path.

    Parameters
    ----------
    fname : str, file name, including data path.

    Returns
    -------
    bool.

    """
    with open(fname, "wb") as f:
        pickle.dump(data, f)


def load(fname):
    """ Load model from specific path.

    Parameters
    ----------
    fname : str, file name, including data path.

    Returns
    -------
    bool.

    """
    with open(fname, "rb") as fobj:
        return pickle.load(fobj)
