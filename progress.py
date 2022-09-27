import sys

is_active = False


def start():
    global is_active
    is_active = True


def write(text: str):
    global is_active
    if not is_active:
        raise IOError("There is no dynamic line at the moment.")
    sys.stdout.flush()
    sys.stdout.write(f"\r{text}")


def end(stay=False):
    global is_active
    is_active = False
    end_str = "\n" if stay else "\r"
    sys.stdout.write(end_str)


class Progress:
    def __init__(self, n: int, name="Progress", progress_bar_length=None, stay=False) -> None:
        self.n = n
        self.name = name
        self.progress_bar_length = progress_bar_length if progress_bar_length is not None else n
        self.stay = stay
        start()
        self._progress = 0
        self.update(self._progress)

    def update(self, progress: int):
        if progress > self.n:
            raise IOError("Progress can't be beyond completed.")
        self._progress = progress
        filled = (progress * self.progress_bar_length) // self.n
        empty = self.progress_bar_length - filled
        write(f"{self.name}: [{'=' * filled}{' ' * empty}] [{progress}/{self.n}]")
        if progress == self.n:
            end(self.stay)
    
    def increment(self):
        self.update(self._progress + 1)
