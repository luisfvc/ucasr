class BColors:
    """
    Colored command line output formatting
    """
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    BOXEDBLUE = '\033[44m'
    BOXEDGREEN = '\033[42m'

    def __init__(self):
        """ Constructor """
        pass

    @staticmethod
    def colored(string, color):
        """ Change color of string """
        return color + string + BColors.ENDC