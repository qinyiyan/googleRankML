# Copyright 2024 RecML authors <no-reply@google.com>.
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
"""Tests for configuration utilities."""

from collections.abc import Sequence
import dataclasses

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
import fiddle as fdl
from mlrx.utils import config as config_lib


@dataclasses.dataclass
class _X:
  value: int


@dataclasses.dataclass
class _Y:
  value: int


@dataclasses.dataclass
class _Object:
  x: _X
  y: _Y


def config_1() -> fdl.Config[_Object]:
  return fdl.Config(
      _Object,
      x=fdl.Config(_X, value=1),
      y=fdl.Config(_Y, value=2),
  )


def fiddler_1(cfg: fdl.Config[_Object]):
  cfg.x.value = 3


def fiddler_2(cfg: fdl.Config[_Object], value: int):
  cfg.y.value = value


class ConfigTest(parameterized.TestCase):

  @parameterized.named_parameters(
      {
          'testcase_name': 'base_config',
          'args': ['config:__main__.config_1'],
          'expected_config': config_1(),
      },
      {
          'testcase_name': 'relative_fiddler',
          'args': [
              'config:__main__.config_1',
              'fiddler:fiddler_1',
              'fiddler:fiddler_2(value=4)',
          ],
          'expected_config': fdl.Config(
              _Object,
              x=fdl.Config(_X, value=3),
              y=fdl.Config(_Y, value=4),
          ),
      },
      {
          'testcase_name': 'absolute_fiddler',
          'args': [
              'config:__main__.config_1',
              'fiddler:__main__.fiddler_2(3)',
          ],
          'expected_config': fdl.Config(
              _Object,
              x=fdl.Config(_X, value=1),
              y=fdl.Config(_Y, value=3),
          ),
      },
      {
          'testcase_name': 'set',
          'args': [
              'config:__main__.config_1',
              'set:x.value=0',
              'set:y.value=0',
          ],
          'expected_config': fdl.Config(
              _Object,
              x=fdl.Config(_X, value=0),
              y=fdl.Config(_Y, value=0),
          ),
      },
  )
  def test_fiddle_flag(
      self, args: Sequence[str], expected_config: fdl.Config[_Object]
  ):
    fdl_flag = config_lib.FiddleFlag(
        name='test_flag',
        default=None,
        parser=flags.ArgumentParser(),
        serializer=None,
        help_string='My fiddle flag',
    )
    fdl_flag.parse(args)
    self.assertEqual(expected_config, fdl_flag.value)

  @parameterized.named_parameters(
      {
          'testcase_name': 'bad_base_config',
          'args': ['config:__main__.config_3'],
          'expected_error': AttributeError,
          'expected_error_regex': 'Could not init a buildable from .*',
      },
      {
          'testcase_name': 'bad_fiddler',
          'args': ['config:__main__.config_1', 'fiddler:fiddler_3'],
          'expected_error': ValueError,
          'expected_error_regex': 'Could not init a buildable from .*',
      },
  )
  def test_invalid_fiddle_flag(
      self,
      args: Sequence[str],
      expected_error: type[Exception],
      expected_error_regex: str,
  ):
    fdl_flag = config_lib.FiddleFlag(
        name='test_flag',
        default=None,
        parser=flags.ArgumentParser(),
        serializer=None,
        help_string='My fiddle flag',
    )

    def _value(args: Sequence[str]):
      fdl_flag.parse(args)
      return fdl_flag.value

    self.assertRaisesRegex(expected_error, expected_error_regex, _value, args)


if __name__ == '__main__':
  absltest.main()
