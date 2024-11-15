__all__ = ['Space', 'For', 'Table', 'Unit']

import abc
import itertools
from typing import List, Type, Iterable, Any, Dict


class Space(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def __iter__(self):
        pass

    @abc.abstractmethod
    def __len__(self):
        pass

    def __add__(self, other):
        return Concat(self, other)

    def __mul__(self, other):
        return Product(self, other)

    @abc.abstractmethod
    def fields(self):
        pass

    def iter_dict(self, dict_class: Type = dict):
        return itertools.starmap(lambda *args: dict_class(args), iter(self))


class Unit(Space):

    def __iter__(self):
        return iter(((),))

    def __len__(self):
        return 1

    def fields(self):
        return tuple()


class For(Space):

    def __init__(self, name: str, values: List):
        self._name = name
        self._values = list(values)

    def __iter__(self):
        x = zip(itertools.repeat(self._name), self._values)
        x = itertools.starmap(lambda a, b: ((a, b),), x)
        return x

    def __len__(self):
        return len(self._values)

    def fields(self):
        return (self._name,)


class Product(Space):

    def __init__(self, a: Space, b: Space):
        self._a = a
        self._b = b

        a_fields = a.fields()
        b_fields = b.fields()
        if len(a_fields) + len(b_fields) != len(set(a_fields + b_fields)):
            raise RuntimeError("Cannot have duplicated fields")

    def __iter__(self):
        x = itertools.product(self._a, self._b)
        x = itertools.starmap(lambda a, b: (*a, *b), x)
        return x

    def __len__(self):
        return len(self._a) * len(self._b)

    def fields(self):
        return self._a.fields() + self._b.fields()


class Concat(Space):

    def __init__(self, a: Space, b: Space):
        self._a = a
        self._b = b
        if set(self._a.fields()) != set(self._b.fields()):
            raise RuntimeError("Fields must be equal")

    def __iter__(self):
        return itertools.chain(self._a, self._b)

    def __len__(self):
        return len(self._a) + len(self._b)

    def fields(self):
        return self._a.fields()


class Table(Space):

    def __init__(self, headers: Iterable[str], rows: Iterable[Iterable[Any]]):
        self._headers = list(headers)
        self._rows = list(rows)

    @classmethod
    def from_dicts(cls, dicts: Iterable[Dict[str, Any]]):
        headers = None
        rows = []
        for d in dicts:
            if headers is None:
                headers = tuple(d.keys())
            if headers != tuple(d.keys()):
                raise RuntimeError("All dicts must have same keys")
            rows.append(d.values())
        if headers is None:
            headers = []
        return Table(headers, rows)

    def __iter__(self):
        x = zip(itertools.repeat(self._headers), self._rows)
        x = itertools.starmap(lambda a, b: list(zip(a, b)), x)
        return x

    def __len__(self):
        return len(self._rows)

    def fields(self):
        return tuple(self._headers)
