"""Prompt syntax is not a local policy gate; provider calls are mocked."""
import ast
import pathlib
import types
import unittest
from unittest.mock import Mock

TREE = ast.parse(pathlib.Path(__file__).with_name('plugin.py').read_text(encoding='utf-8'))
CLASS = next(n for n in TREE.body if isinstance(n, ast.ClassDef) and n.name == 'NekoDraw')
NODES = [n for n in CLASS.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
         and n.name in {'_draw', 'cmd_draw', '_planner_instruction'}]
for node in NODES:
    node.decorator_list = []
    node.returns = None
    for arg in node.args.args + node.args.kwonlyargs:
        arg.annotation = None
    if node.args.kwarg:
        node.args.kwarg.annotation = None
namespace = {'MAX_PROMPT_CHARS': 4000}
exec(compile(ast.Module(body=NODES, type_ignores=[]), 'plugin.py', 'exec'), namespace)
namespace['draw'] = namespace['_draw']


class PromptSyntaxTests(unittest.IsolatedAsyncioTestCase):
    def bot(self, allowed=True, active=None):
        bot = types.SimpleNamespace(
            _generation_id=lambda: 0,
            config=types.SimpleNamespace(plugin=types.SimpleNamespace(enabled=True)),
            ctx=types.SimpleNamespace(logger=Mock()),
            _stream_id=lambda kw: kw.get('stream_id', ''),
            _active_task=lambda sid: active,
            _allowed=lambda sender, platform: allowed,
            _command_identity=lambda stream, user, kw: dict(stream_id=stream, user_id=user, platform='qq'),
            _status_text=lambda sid: 'status',
            _start_task=Mock(return_value=types.SimpleNamespace(summary=lambda: 'queued')),
        )
        async def draw(**kwargs):
            return await namespace['draw'](bot, **kwargs)
        bot._draw = draw
        return bot

    async def test_tags_short_phrases_and_substrings_accepted_by_tool_and_command(self):
        for text in ['masterpiece, best quality, 1girl, solo, red hair, smile, blush, indoors',
                     '红发, 红瞳, 短发, 发夹, 碎发, 特写, 坏笑, 脸红',
                     'A musician performs a solo beside a highres poster', '蓝色的圆']:
            for entry in ['draw', 'cmd_draw']:
                bot = self.bot()
                if entry == 'draw':
                    result = await namespace[entry](bot, prompt=text, stream_id='chat')
                    self.assertTrue(result['success'])
                else:
                    result = await namespace[entry](bot, stream_id='chat', user_id='user', matched_groups={'payload': text})
                    self.assertTrue(result[0])
                self.assertEqual(bot._start_task.call_args.args[1], text)

    async def test_empty_disabled_missing_chat_and_active_task_still_rejected(self):
        for case in ['empty', 'disabled', 'missing', 'active']:
            bot = self.bot(active=types.SimpleNamespace(summary=lambda: 'running') if case == 'active' else None)
            if case == 'disabled':
                bot.config.plugin.enabled = False
            result = await namespace['draw'](bot, prompt='' if case == 'empty' else 'solo', stream_id='' if case == 'missing' else 'chat')
            self.assertFalse(result['success'])
            bot._start_task.assert_not_called()

    async def test_command_whitelist_still_enforced(self):
        bot = self.bot(allowed=False)
        result = await namespace['cmd_draw'](bot, stream_id='chat', user_id='user', matched_groups={'payload': 'solo'})
        self.assertFalse(result[0])
        bot._start_task.assert_not_called()

    def test_self_portrait_memory_requirement_remains(self):
        text = namespace['_planner_instruction']()
        self.assertIn('自画像先调用 query_memory', text)
        self.assertIn('neko_edit_image', text)
        self.assertIn('标签或混合描述均可', text)
        self.assertNotIn('不要使用 masterpiece', text)
