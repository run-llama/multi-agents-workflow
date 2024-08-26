import builtins


class PrintPrefix:
    def __init__(self, prefix):
        self.prefix = prefix
        self.original_print = builtins.print

    def __enter__(self):
        builtins.print = self._print_with_prefix

    def __exit__(self, exc_type, exc_val, exc_tb):
        builtins.print = self.original_print

    def _print_with_prefix(self, *args, **kwargs):
        self.original_print(self.prefix, *args, **kwargs)
