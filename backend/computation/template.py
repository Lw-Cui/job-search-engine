import nltk

nltk.data.path.append("./nltk_data")  # set local path for nltk lib


def init(filename):
    """
    Initialize global data structure for current algorithm.

    Parameters
    ----------
    filename : the filename of job description, such as `amazon_jobs_dataset.csv`
    """
    pass


def query(intro: str):
    """
    Conduct query.

    Parameters
    ----------
    intro : user self-introduction

    Returns
    -------
    job lists : list of dict, which shall include below fields:

        * company
        * title
        * category
        * location
        * description
        * minimum_qualification
        * preferred_qualification
    """
    pass


# local test
if __name__ == '__main__':
    pass
