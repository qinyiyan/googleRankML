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
"""Fiddle configuration utilities."""

import ast
import dataclasses
import enum
import importlib
import inspect
import re
import types
import typing
from typing import Any, Self

from absl import flags
from absl import logging
import fiddle as fdl
from fiddle import absl_flags as fdl_flags
from fiddle.experimental import auto_config


class FiddleFlag(fdl_flags.FiddleFlag):
  """ABSL flag class for a fiddle configuration.

  Wraps the fiddle flag to allow for using local module fiddlers in the module
  where the base config is defined.

  See the documentation of the parent class for more details.
  """

  def _initial_config(self, expression: str):
    call_expr = CallExpression.parse(expression)
    base_name = call_expr.func_name
    base_fn = _resolve_function_reference(
        base_name,
        ImportDottedNameDebugContext.BASE_CONFIG,
        self.default_module,
        self.allow_imports,
        'Could not init a buildable from',
    )
    self.default_module = inspect.getmodule(base_fn)
    try:
      if auto_config.is_auto_config(base_fn):
        cfg = base_fn.as_buildable(*call_expr.args, **call_expr.kwargs)
      else:
        cfg = base_fn(*call_expr.args, **call_expr.kwargs)
    except (AttributeError, ValueError) as e:
      raise ValueError(
          f'Failed to init a buildable from expression: {expression}.'
      ) from e

    if cfg is None:
      raise ValueError(
          f'Could not init a buildable from {expression}. Make sure the'
          ' function name is valid and that it returns a fiddle buildable.'
      )
    return cfg


def DEFINE_fiddle_config(  # pylint: disable=invalid-name
    name: str,
    *,
    default: Any = None,
    help_string: str,
    pyref_policy: Any | None = None,
    flag_values: flags.FlagValues = flags.FLAGS,
    required: bool = False,
) -> flags.FlagHolder[Any]:
  r"""Declare and define an fiddle config line flag object.

  When used in a python binary, after the flags have been parsed from the
  command line, this command line flag object contain a `fdl.Config` of the
  object.

  Example usage in a python binary:
  ```
  _EXPERIMENT_CONFIG = DEFINE_experiment_config(
      "experiment_cfg", help_string="My experiment config",
  )

  def experiment() -> fdl.Config[Experiment]:
    return fdl.Config(Experiment,...)

  def set_steps(experiment_cfg: fdl.Config[Experiment], steps: int):
    experiment_cfg.task.trainer.train_steps = steps

  def main(argv):
    experiment_cfg = _EXPERIMENT_CONFIG.value
    experiment = fdl.build(experiment_cfg)
    run_experiment(experiment, mode="train_and_eval")

  if __name__ == "__main__":
    app.run(main)
  ```

  results in the `_EXPERIMENT_CONFIG.value` set to a fiddle configuration of the
  experiment with all the command line flags applied in the order they were
  passed in.

  Args:
    name: name of the command line flag.
    default: default value of the flag.
    help_string: help string describing what the flag does.
    pyref_policy: a policy for importing references to Python objects.
    flag_values: the ``FlagValues`` instance with which the flag will be
      registered. This should almost never need to be overridden.
    required: bool, is this a required flag. This must be used as a keyword
      argument.

  Returns:
    A handle to defined flag.
  """
  return flags.DEFINE_flag(
      FiddleFlag(
          name=name,
          default_module=None,
          default=default,
          pyref_policy=pyref_policy,
          parser=flags.ArgumentParser(),
          serializer=fdl_flags.FiddleFlagSerializer(pyref_policy=pyref_policy),
          help_string=help_string,
      ),
      flag_values=flag_values,
      required=required,
  )


class ImportDottedNameDebugContext(enum.Enum):
  """Context of importing a sumbol, for error messages."""

  BASE_CONFIG = 1
  FIDDLER = 2

  def error_prefix(self, name: str) -> str:
    if self == ImportDottedNameDebugContext.BASE_CONFIG:
      return f'Could not init a buildable from {name!r}'
    assert self == ImportDottedNameDebugContext.FIDDLER
    return f'Could not load fiddler {name!r}'


def _is_dotted_name(name: str) -> bool:
  return len(name.split('.')) >= 2


def _import_dotted_name(
    name: str,
    mode: ImportDottedNameDebugContext,
    module: types.ModuleType | None,
) -> Any:
  """Returns the Python object with the given dotted name.

  Args:
    name: The dotted name of a Python object, including the module name.
    mode: Whether we're looking for a base config function or a fiddler, used
      only for constructing error messages.
    module: A common namespace to use as the basis for resolving the import, if
      None, we will attempt to use absolute imports.

  Returns:
    The named value.

  Raises:
    ValueError: If `name` is not a dotted name.
    ModuleNotFoundError: If no dotted prefix of `name` can be imported.
    AttributeError: If the imported module does not contain a value with
      the indicated name.
  """

  if not _is_dotted_name(name):
    raise ValueError(
        f'{mode.error_prefix(name)}: Expected a dotted name including the '
        'module name.'
    )

  name_pieces = name.split('.')
  if module is not None:
    name_pieces = [module.__name__] + name_pieces

  # We don't know where the module ends and the name begins; so we need to
  # try different split points.  Longer module names take precedence.
  for i in range(len(name_pieces) - 1, 0, -1):
    try:
      value = importlib.import_module('.'.join(name_pieces[:i]))
      for j, name_piece in enumerate(name_pieces[i:]):
        try:
          value = getattr(value, name_piece)  # Can raise AttributeError.
        except AttributeError:
          available_names = ', '.join(
              repr(n) for n in dir(value) if not n.startswith('_')
          )
          module_name = '.'.join(name_pieces[: i + j])
          failing_name = name_pieces[i + j]
          raise AttributeError(
              f'{mode.error_prefix(name)}: module {module_name!r} has no '
              f'attribute {failing_name!r}; available names: {available_names}'
          ) from None
      return value
    except ModuleNotFoundError:
      if i == 1:  # Final iteration through the loop.
        raise

  # The following line should be unreachable -- the "if i == 1: raise" above
  # should have raised an exception before we exited the loop.
  raise ModuleNotFoundError(f'No module named {name_pieces[0]!r}')


@dataclasses.dataclass(frozen=True)
class CallExpression:
  """Parsed components of a call expression (or bare function name).

  Examples:

  >>> CallExpression.parse("fn('foo', False, x=[1, 2])")
  CallExpression(func_name='fn', args=('foo', False'), kwargs={'x': [1, 2]})
  >>> CallExpression.parse("fn")  # Bare function name: empty args/kwargs.
  CallExpression(func_name='fn', args=()), kwargs={})

  Attributes:
    func_name: The name fo the function that should be called.
    args: Parsed values of positional arguments for the function.
    kwargs: Parsed values of keyword arguments for the function.
  """

  func_name: str
  args: tuple[Any, ...] | None
  kwargs: dict[str, Any] | None

  _PARSE_RE = re.compile(r'(?P<func_name>[\w\.]+)(?:\((?P<args>.*)\))?')

  @classmethod
  def parse(cls, value: str) -> Self:
    """Returns a CallExpression parsed from a string.

    Args:
      value: A string containing positional and keyword arguments for a
        function.  Must consist of an open paren followed by comma-separated
        argument values followed by a close paren.  Argument values must be
        literal constants (i.e., must be parsable with `ast.literal_eval`).
        var-positional and var-keyword arguments are not supported.

    Raises:
      SyntaxError: If `value` is not a simple call expression with literal
        arguments.
    """
    m = re.fullmatch(cls._PARSE_RE, value)
    if m is None:
      raise SyntaxError(
          f'Expected a function name or call expression; got: {value!r}'
      )
    if m.group('args') is None:  # Bare function name
      return CallExpression(m.group('func_name'), (), {})

    node = ast.parse(value)  # Can raise SyntaxError.
    if not (
        isinstance(node, ast.Module)
        and len(node.body) == 1
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Call)  # pylint: disable=attribute-error
    ):
      raise SyntaxError(
          f'Expected a function name or call expression; got: {value!r}'
      )
    call_node = node.body[0].value  # pylint: disable=attribute-error
    args = []
    for arg in call_node.args:
      if isinstance(arg, ast.Starred):
        raise SyntaxError('*args is not supported.')
      try:
        args.append(ast.literal_eval(arg))
      except ValueError as exc:
        raise SyntaxError(
            'Expected arguments to be simple literals; got '
            f'{ast.unparse(arg)!r} (while parsing {value!r})'
        ) from exc
    kwargs = {}
    for kwarg in call_node.keywords:
      if kwarg.arg is None:
        raise SyntaxError('**kwargs is not supported.')
      try:
        kwargs[kwarg.arg] = ast.literal_eval(kwarg.value)
      except ValueError as exc:
        raise SyntaxError(
            'Expected arguments to be simple literals; got '
            f'{ast.unparse(kwarg.value)!r} (while parsing {value!r})'
        ) from exc
    return CallExpression(m.group('func_name'), tuple(args), kwargs)


def _resolve_function_reference(
    function_name: str,
    mode: ImportDottedNameDebugContext,
    module: types.ModuleType | None,
    allow_imports: bool,
    failure_msg_prefix: str,
):
  """Returns function that produces `fdl.Buildable` from its name.

  Args:
    function_name: The name of the function.
    mode: Whether we're looking for a base config function or a fiddler.
    module: A common namespace to use as the basis for finding configs and
      fiddlers. May be `None`; if `None`, only fully qualified Fiddler imports
      will be used (or alternatively a base configuration can be specified using
      the `--fdl_config_file` flag.). Dotted imports are resolved relative to
      `module` if not None, by preference, or else absolutely.
    allow_imports: If true, then fully qualified dotted names may be used to
      specify configs or fiddlers that should be automatically imported.
    failure_msg_prefix: Prefix string to prefix log messages in case of
      failures.

  Returns:
    The named value.
  """
  if hasattr(module, function_name):
    return getattr(module, function_name)
  elif allow_imports and _is_dotted_name(function_name):
    # Try a relative import first.
    if module is not None:
      try:
        return _import_dotted_name(
            function_name,
            mode=mode,
            module=module,
        )
      except (ModuleNotFoundError, ValueError, AttributeError):
        # Intentionally ignore the exception here. We will reraise after trying
        # again without relative import.
        pass

    # Try absolute import for the provided function name / symbol.
    try:
      return _import_dotted_name(
          function_name,
          mode=mode,
          module=None,
      )
    except ModuleNotFoundError as e:
      raise ValueError(f'{failure_msg_prefix} {function_name!r}: {e}') from e
  else:
    available_names = _find_base_config_like_things(module)
    raise ValueError(
        f'{failure_msg_prefix} {function_name!r}: Could not resolve reference '
        f'to named function, available names: {", ".join(available_names)}.'
    )


def _find_base_config_like_things(source_module: Any) -> list[str]:
  """Returns names of attributes of 0-arity functions that might return Configs.

  A base config-producting function is a function that takes no (required)
  arguments, and returns a `fdl.Buildable`.

  Args:
    source_module: A module upon which to search for `base_config`-like
      functions.

  Returns:
    A list of attributes on `source_module` that appear to be base
    config-producing functions.
  """
  available_base_names = []
  for name in dir(source_module):
    if name.startswith('__'):
      continue
    if name in dir(typing):
      continue
    try:
      sig = inspect.signature(getattr(source_module, name))

      # Exclude functions that do not return a single config.
      return_type = sig.return_annotation
      origin = typing.get_origin(return_type)
      if origin is not None and not issubclass(origin, fdl.Buildable):
        continue

      def is_required_arg(name: str) -> bool:
        param = sig.parameters[name]  # pylint: disable=cell-var-from-loop
        return (
            param.kind == param.POSITIONAL_ONLY
            or param.kind == param.POSITIONAL_OR_KEYWORD
        ) and param.default is param.empty

      if not any(filter(is_required_arg, sig.parameters)):
        available_base_names.append(name)
    except Exception:  # pylint: disable=broad-except
      logging.debug(
          'Encountered exception while inspecting function called: %s', name
      )
  return available_base_names
