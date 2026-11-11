"""Opt-in real Host context integration, isolated instance and no network/IO calls.

Run with the maintained Host's Python environment and --host-source checkout.
Normal CI uses the no-anchor envelope fixtures in test_tool_context.py instead.
"""
import argparse
import asyncio
from contextlib import ExitStack
import copy
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace, ModuleType
from unittest.mock import AsyncMock, Mock, patch


async def verify(host):
    from src.maisaka.reasoning_engine import MaisakaReasoningEngine
    from src.maisaka.focus.tools import execution_context
    from src.plugin_runtime.component_query import ComponentQueryService
    from src.plugin_runtime.host.component_registry import ToolEntry
    from src.core.tooling import ToolInvocation
    from src.common.database.database import engine
    from src.common.logger import close_handlers
    from maibot_sdk.components import collect_components

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from test_tool_context import tool_context

    package = ModuleType('draw_host_integration')
    package.__path__ = [str(root)]
    sys.modules[package.__name__] = package
    mod = importlib.import_module(package.__name__ + '.plugin')

    class Bot(mod.NekoDraw):
        config = mod.PluginConfig()
        ctx = SimpleNamespace(logger=Mock())

    bot = Bot()
    bot._start_task = Mock(return_value=SimpleNamespace(summary=lambda: 'queued'))
    bot._generate = AsyncMock(side_effect=AssertionError('No provider request'))
    bot._send = AsyncMock(side_effect=AssertionError('No message send'))
    entries = {item['name']: ToolEntry(item['name'], 'TOOL', 'neko.draw', item['metadata'])
               for item in collect_components(bot) if item['type'] == 'TOOL'}
    source_hashes = {}
    for method in (MaisakaReasoningEngine._build_tool_execution_context, execution_context,
                   ComponentQueryService._build_tool_invocation_payload):
        path = Path(method.__code__.co_filename).resolve()
        assert path.is_relative_to(host), 'Unexpected Host checkout'
        source_hashes[str(path.relative_to(host))] = hashlib.sha256(path.read_bytes()).hexdigest()

    passed = []
    try:
        for mode in ('normal', 'focus'):
            for platform in ('qq', 'telegram'):
                for group in (False, True):
                    session = f'{mode}-{platform}-{"group" if group else "private"}'
                    group_id = ('-100123' if platform == 'telegram' else '789') if group else ''
                    user_id = '' if group else '123'
                    runtime = SimpleNamespace(session_id=session, chat_stream=SimpleNamespace(
                        platform=platform, group_id=group_id, user_id=user_id, is_group_session=group))
                    # Actual normal and Focus builders: no hand-built context or account.
                    context = (MaisakaReasoningEngine(runtime)._build_tool_execution_context('offline')
                               if mode == 'normal' else execution_context(runtime, 'offline'))
                    assert not context.metadata.get('anchor_message')
                    for name, handler, business in (
                        ('neko_draw', bot.draw, {'prompt': 'cat'}),
                        ('neko_edit_image', bot.edit_image, {'prompt': 'edit', 'source_message_id': 'm'}),
                        ('neko_draw_status', bot.draw_status, {}),
                    ):
                        entry = entries[name]
                        assert entry.metadata['trusted_context'] == 'neko.tools.context.v1'
                        forged = dict(business, stream_id='foreign', session_id='foreign', account_id='999',
                                      platform='other', user_id='999', message={'forged': True},
                                      _host_context=tool_context('foreign', account_id='999'))
                        before = copy.deepcopy(forged)
                        invocation = ToolInvocation(tool_name=name, arguments=forged, session_id=session, stream_id=session)
                        payload = ComponentQueryService._build_tool_invocation_payload(invocation, context, entry=entry)
                        assert forged == before
                        assert payload['_host_context'] == tool_context(session, platform, user_id, group_id=group_id)
                        assert payload['_host_context']['account_id'] == ''
                        # Host intentionally preserves ordinary model kwargs: plugin must ignore them.
                        assert payload['stream_id'] == 'foreign'
                        with patch.object(mod, 'resolve_edit_images', new_callable=AsyncMock, return_value=[b'source']) as source, \
                                patch.object(bot, '_status_text', return_value='trusted status') as status:
                            result = await handler(**payload)
                            assert result['success'], (session, name, result)
                            if name == 'neko_edit_image':
                                assert source.call_args.args[1:4] == (session, 'm', None)
                            if name == 'neko_draw_status':
                                status.assert_called_once_with(session)
                            else:
                                assert bot._start_task.call_args.args[0] == session
                                assert bot._start_task.call_args.args[2] == user_id
                        # No Host context must overwrite even a fully forged valid envelope.
                        missing = ComponentQueryService._build_tool_invocation_payload(invocation, None, entry=entry)
                        assert missing['_host_context'] == {}
                        assert not (await handler(**missing))['success']
                        # Plugin validates malformed trusted sessions independently of extra kwargs.
                        for bad in ({'session_id': '', 'stream_id': ''}, {'stream_id': 'other'}):
                            invalid = {**payload, '_host_context': {**payload['_host_context'], **bad}}
                            assert not (await handler(**invalid))['success']
                        passed.append(session + ':' + name)
        bot._generate.assert_not_awaited()
        bot._send.assert_not_awaited()
        return dict(passed=len(passed), cases=passed, sdk=importlib.metadata.version('maibot-plugin-sdk'),
                    host_sources=source_hashes, plugin_sha256=hashlib.sha256((root / 'plugin.py').read_bytes()).hexdigest())
    finally:
        await bot.on_unload()
        engine.dispose()
        close_handlers()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host-source', type=Path, required=True)
    parser.add_argument('--temp-dir', type=Path)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    host = args.host_source.resolve()
    report = args.report.resolve() if args.report else None
    with ExitStack() as cleanup:
        directory = cleanup.enter_context(TemporaryDirectory(prefix='draw-host-context-', dir=args.temp_dir))
        instance = Path(directory).resolve()
        old_cwd = Path.cwd()
        os.chdir(instance)
        cleanup.callback(os.chdir, old_cwd)
        os.environ.update(MAIBOT_INSTANCE_DIR=str(instance), MAIBOT_LOCALE='zh-CN',
                          MAIBOT_RUNTIME_CONTEXT=json.dumps(dict(bot_id='0' * 32,
                              http_endpoint=str(instance / 'synthetic.sock'), public_base_url='http://synthetic.invalid',
                              status_path=str(instance / 'worker-status.json'))))
        sys.path[:0] = [str(host / 'scripts'), str(host)]
        import neko_pytest_guard  # noqa: F401 -- deny every network call before Host imports
        result = asyncio.run(verify(host))
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if report:
        report.write_text(text + '\n', encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
