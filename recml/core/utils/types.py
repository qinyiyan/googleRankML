# Copyright 2024 RecML authors <recommendations-ml@google.com>.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Configuration tools for dealing with abstract types."""

import abc
import dataclasses
from typing import Generic, Protocol, TypeVar

from typing_extensions import dataclass_transform


T = TypeVar("T")


@dataclass_transform(field_specifiers=dataclasses.field)  # type: ignore[literal-required]
class Dataclass:
  """A dataclass transform that converts a class to a dataclass."""

  def __init_subclass__(cls, **kwargs):
    def replace(self, **updates):
      return dataclasses.replace(self, **updates)

    data_cls = dataclasses.dataclass(**kwargs)(cls)
    data_cls.replace = replace

  def __init__(self, *args, **kwargs):
    # stub for pytype
    raise NotImplementedError

  def replace(self: T, **overrides) -> T:
    # stub for pytype
    raise NotImplementedError


# TODO(aahil): Share code with `Dataclass`.
@dataclass_transform(field_specifiers=dataclasses.field)  # type: ignore[literal-required]
class FrozenDataclass:
  """A dataclass transform that converts a class to a frozen dataclass."""

  def __init_subclass__(cls, **kwargs):
    if "frozen" not in kwargs:
      kwargs["frozen"] = True

    def replace(self, **updates):
      return dataclasses.replace(self, **updates)

    data_cls = dataclasses.dataclass(**kwargs)(cls)
    data_cls.replace = replace

  def __init__(self, *args, **kwargs):
    # stub for pytype
    raise NotImplementedError

  def replace(self: T, **overrides) -> T:
    # stub for pytype
    raise NotImplementedError


class Factory(abc.ABC, Generic[T], Dataclass):
  """A factory interface for configuring an arbitary object via a dataclass.

  This is useful for creating objects that require run-time information.
  """

  @abc.abstractmethod
  def make(self, *args, **kwargs) -> T:
    """Builds the object instance."""


class FactoryProtocol(Protocol, Generic[T]):
  """A protocol for typing factories."""

  def make(self, *args, **kwargs) -> T:
    """Builds the object instance."""

  def replace(self, **overrides) -> T:
    """Replaces the object instance."""


class ObjectFactory(Factory[T]):
  """A factory that wraps around the constructor of an object.

  This is useful when a library only accepts a factory but creating a factory
  introduces unnecessary boilerplate.

  Example usage:
  ```
  class MyObject:
    def __init__(self, x: int, y: int):
      self._x = x
      self._y = y


  factory = ObjectFactory(MyObject, x=1, y=2)
  obj = factory.make()
  assert obj._x == 1
  assert obj._y == 2
  ```
  """

  def __new__(cls, *args, **kwargs) -> Factory[T]:
    if args:
      raise ValueError(
          "`StaticFactory` does not accept positional arguments. Got args:"
          f" {args}."
      )
    if "type" not in kwargs:
      raise ValueError(
          "`StaticFactory` requires a `type` keyword argument. Got kwargs:"
          f" {kwargs}."
      )

    class _ObjectFactory(Factory):

      def __init_subclass__(cls, **kwargs):
        # Override the dataclass transform from the base class.
        pass

      def make(self):
        return getattr(self, "type")(**{
            f.name: getattr(self, f.name)
            for f in dataclasses.fields(self)
            if f.name != "type"
        })

    sub_cls = dataclasses.make_dataclass(
        cls_name=cls.__name__,
        fields=[(k, type(v)) for k, v in kwargs.items()],
        bases=(_ObjectFactory,),
        kw_only=True,
    )
    obj = sub_cls(**kwargs)
    return obj

  def __init__(self, *, type: type[T], **kwargs):  # pylint: disable=redefined-builtin
    # Stub for pytype.
    raise NotImplementedError()

  @property
  def type(self) -> type[T]:
    # Stub for pytype.
    raise NotImplementedError()

  def make(self) -> T:
    # Stub for pytype.
    raise NotImplementedError()
