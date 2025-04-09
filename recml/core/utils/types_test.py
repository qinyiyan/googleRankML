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
"""Tests for type utilities."""

import dataclasses

from absl.testing import absltest
from absl.testing import parameterized
from recml.core.utils import types


class TypesTest(parameterized.TestCase):

  @parameterized.named_parameters(
      {'testcase_name': 'dataclass', 'cls': types.Dataclass},
      {'testcase_name': 'frozen_dataclass', 'cls': types.FrozenDataclass},
  )
  def test_dataclass_transform(self, cls: type[types.Dataclass]):
    class Foo(cls):
      x: int
      y: int
      z: int = dataclasses.field(default_factory=lambda: 1)

    class Bar(Foo):
      u: int = dataclasses.field(default_factory=lambda: 2)

    foo = Foo(x=1, y=2)
    self.assertEqual(foo.x, 1)
    self.assertEqual(foo.y, 2)
    self.assertEqual(foo.z, 1)
    self.assertTrue(dataclasses.is_dataclass(Foo))
    self.assertTrue(dataclasses.is_dataclass(foo))

    bar = Bar(x=1, y=2, u=3)
    self.assertEqual(bar.x, 1)
    self.assertEqual(bar.y, 2)
    self.assertEqual(bar.z, 1)
    self.assertEqual(bar.u, 3)
    self.assertTrue(dataclasses.is_dataclass(Bar))
    self.assertTrue(dataclasses.is_dataclass(bar))

  def test_frozen_dataclass(self):

    class Foo(types.Dataclass):
      x: int
      y: int

    class Bar(types.FrozenDataclass):
      x: int
      y: int

    def _mutate_foo_or_bar(foo_or_bar: Foo | Bar):
      foo_or_bar.x = 2

    # Mutating Foo is allowed.
    _mutate_foo_or_bar(Foo(x=1, y=2))

    self.assertRaises(
        dataclasses.FrozenInstanceError,
        _mutate_foo_or_bar,
        Bar(x=1, y=2),
    )

  def test_object_factory(self):
    class Foo(types.Dataclass):
      x: int
      y: int

    factory = types.ObjectFactory(type=Foo, x=1, y=2)
    self.assertEqual(factory.type, Foo)
    self.assertEqual(factory.x, 1)
    self.assertEqual(factory.y, 2)
    self.assertEqual(factory.make(), Foo(x=1, y=2))


if __name__ == '__main__':
  absltest.main()
