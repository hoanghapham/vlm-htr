# From Riksarkivet with slight modifications
class Ratio:
    """
    An unormalized `fraction.Fraction`

    This class makes it easy to compute the total error rate (and other
    fraction-based metrics) of several documents.

    Example: Let A and B be two documents with WER(A) = 1/5 and
    WER(B) = 1/100. In total, there are 2 errors and 105 words, so
    WER(A+B) = 2/105.

    Addition of two `Ratio` instances supports this:
    >>> Ratio(1, 5) + Ratio(1, 100)
    Ratio(2, 105)
    """

    def __init__(self, a, b):
        self.a = int(a)
        self.b = int(b)

    def __add__(self, other):
        if other == 0:
            # sum(a, b) performs the addition 0 + a + b internally,
            # which means that Ratio must support addition with 0 in
            # order to work with sum().
            return self
        return Ratio(self.a + other.a, self.b + other.b)

    __radd__ = __add__  # redirects int + Ratio to __add__

    def __float__(self):
        return 0.0 if self.b == 0 else float(self.a / self.b)

    def __gt__(self, other):
        return float(self) > float(other)

    def __lt__(self, other):
        return float(self) < float(other)

    def __eq__(self, other):
        return float(self) == float(other)

    def __str__(self):
        return f"{self.a}/{self.b}"