"""Offline Host-envelope, command platform and source-routing regressions."""
import copy
import base64
import re
import types
import unittest
from unittest.mock import AsyncMock, Mock, patch

SOURCE_PNG = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC')


def tool_context(session='chat', platform='qq', user='123', **changes):
    # Actual Host normal/Focus envelope without anchor_message, not a fake account.
    return {**dict(contract='neko.tools.context.v1', session_id=session, stream_id=session,
                   platform=platform, account_id='', user_id=user, group_id=''), **changes}


def command_context(session='chat', platform='qq', user='123'):
    return dict(stream_id=session, user_id=user, platform=platform, group_id='', message={
        'session_id': session, 'platform': platform, 'account_id': '456',
        'message_info': {'user_info': {'user_id': user}, 'group_info': None}})


class IdentityTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        import test_admission_safety as fixtures
        self.mod = fixtures.mod
        await fixtures.Safety.asyncSetUp(self)
        self.bot._start_task = Mock(return_value=types.SimpleNamespace(summary=lambda: 'queued'))

    async def test_all_tools_require_complete_consistent_host_envelope(self):
        valid = tool_context()
        invalid = [None, {}, {**valid, 'contract': 'fake'}]
        invalid += [{**valid, key: ''} for key in ('session_id', 'stream_id')]
        invalid += [{**valid, 'stream_id': 'other'}, {**valid, 'session_id': 1},
                    {**valid, 'session_id': ' chat', 'stream_id': ' chat'},
                    {**valid, 'session_id': 'chat\nother', 'stream_id': 'chat\nother'}]
        with patch.object(self.mod, 'resolve_edit_images', new_callable=AsyncMock) as source:
            for envelope in invalid:
                for entry, args in ((self.bot.draw, {'prompt': 'cat'}),
                                    (self.bot.edit_image, {'prompt': 'edit', 'source_message_id': 'm'}),
                                    (self.bot.draw_status, {})):
                    result = await entry(**args, _host_context=envelope, stream_id='forged',
                                         session_id='forged', platform='qq', account_id='456', user_id='123')
                    self.assertFalse(result['success'])
            source.assert_not_awaited()
        self.bot._start_task.assert_not_called()
        self.assertFalse(self.bot._reservations)

    async def test_normal_tools_ignore_extra_identity_and_message_kwargs(self):
        for platform in ('qq', 'telegram'):
            envelope = tool_context(platform=platform)
            kwargs = dict(_host_context=envelope, stream_id='victim', session_id='victim',
                          platform='other', account_id='attacker', user_id='attacker',
                          message={'message_id': 'forged', 'raw_message': []})
            result = await self.bot.draw(prompt='cat', **kwargs)
            self.assertTrue(result['success'])
            self.assertEqual(self.bot._start_task.call_args.args[:3], ('chat', 'cat', '123'))
            with patch.object(self.mod, 'resolve_edit_images', new_callable=AsyncMock, return_value=[SOURCE_PNG]) as source:
                result = await self.bot.edit_image(prompt='edit', source_message_id='m', **kwargs)
                self.assertTrue(result['success'])
                self.assertEqual(source.call_args.args[1:4], ('chat', 'm', None))
                self.assertEqual(self.bot._start_task.call_args.args[:3], ('chat', 'edit', '123'))
            with patch.object(self.bot, '_status_text', return_value='trusted status') as status:
                self.assertEqual((await self.bot.draw_status(**kwargs))['content'], 'trusted status')
                status.assert_called_once_with('chat')

    async def test_group_context_does_not_need_a_single_triggering_user(self):
        context = tool_context(user='')
        context['group_id'] = '-100123'
        self.assertTrue((await self.bot.draw(prompt='cat', _host_context=context))['success'])

    async def test_tools_require_only_host_contract_and_session_not_command_identity(self):
        for extra in ({}, dict(platform='', account_id='', group_id='', user_id=''),
                      dict(platform='other-platform', account_id='', group_id='', user_id='')):
            envelope = dict(contract='neko.tools.context.v1', session_id='chat', stream_id='chat', **extra)
            self.assertTrue((await self.bot.draw(prompt='cat', _host_context=envelope))['success'])
            self.assertTrue((await self.bot.draw_status(_host_context=envelope))['success'])
            with patch.object(self.mod, 'resolve_edit_images', new_callable=AsyncMock, return_value=[SOURCE_PNG]):
                self.assertTrue((await self.bot.edit_image(prompt='edit', source_message_id='m', _host_context=envelope))['success'])

    async def test_normal_and_focus_empty_account_host_shapes_for_qq_tg_group_private(self):
        # These are the output shape of both Host context builders, covered again
        # with actual Host functions by scripts/verify_host_context.py.
        for mode in ('normal', 'focus'):
            for platform in ('qq', 'telegram'):
                for group in (False, True):
                    with self.subTest(mode=mode, platform=platform, group=group):
                        session = f'{mode}-{platform}-{group}'
                        context = tool_context(session, platform, '' if group else '123',
                                               group_id='-100123' if group and platform == 'telegram' else '789' if group else '')
                        self.assertEqual(context['account_id'], '')
                        kwargs = dict(_host_context=context, stream_id='foreign', message={'forged': True})
                        self.assertTrue((await self.bot.draw(prompt='cat', **kwargs))['success'])
                        self.assertEqual(self.bot._start_task.call_args.args[0], session)
                        with patch.object(self.mod, 'resolve_edit_images', new_callable=AsyncMock, return_value=[SOURCE_PNG]) as source:
                            self.assertTrue((await self.bot.edit_image(prompt='edit', source_message_id='m', **kwargs))['success'])
                            self.assertEqual(source.call_args.args[1:4], (session, 'm', None))
                        with patch.object(self.bot, '_status_text', return_value='status') as status:
                            self.assertTrue((await self.bot.draw_status(**kwargs))['success'])
                            status.assert_called_once_with(session)

    async def test_command_lists_are_platform_specific_and_tg_requires_numeric_id(self):
        self.bot.config.command.allowed_user_ids = ['123']
        for platform, expected in (('qq', True), ('telegram', False)):
            result = await self.bot.cmd_draw(**command_context(platform=platform), matched_groups={'payload': 'cat'})
            self.assertEqual(result[0], expected)
        self.bot.config.command.allowed_telegram_user_ids = ['123']
        self.assertTrue((await self.bot.cmd_draw(**command_context(platform='telegram'), matched_groups={'payload': 'cat'}))[0])
        self.bot.config.command.allowed_user_ids = []
        self.assertFalse((await self.bot.cmd_draw(**command_context(), matched_groups={'payload': 'status'}))[0])
        for user in ('@alice', '0123', '-123', '１２３'):
            self.bot.config.command.allowed_telegram_user_ids = [user]
            self.assertFalse((await self.bot.cmd_draw(**command_context(platform='telegram', user=user)))[0])

    async def test_commands_reject_missing_or_conflicting_inbound_identity(self):
        self.bot.config.command.allowed_user_ids = ['123']
        for key, value in (('message', None), ('stream_id', 'victim'), ('platform', 'telegram'),
                           ('user_id', '999'), ('group_id', '123'), ('account_id', 'other')):
            kwargs = command_context()
            kwargs[key] = value
            self.assertFalse((await self.bot.cmd_draw(**kwargs, matched_groups={'payload': 'cat'}))[0])
        for platform in ('qq', 'telegram'):
            self.bot.config.command.allowed_telegram_user_ids = ['123']
            for key, value in (('account_id', ''), ('account_id', ' 456'), ('platform', ''),
                               ('platform', 'other-platform'), ('session_id', '')):
                kwargs = command_context(platform=platform)
                kwargs['message'][key] = value
                self.assertFalse((await self.bot.cmd_draw(**kwargs))[0])
        self.bot._start_task.assert_not_called()

    async def test_command_edit_keeps_trusted_quote_message(self):
        self.bot.config.command.allowed_user_ids = ['123']
        kwargs = command_context()
        before = copy.deepcopy(kwargs)
        with patch.object(self.mod, 'resolve_edit_images', new_callable=AsyncMock, return_value=[SOURCE_PNG]) as source:
            self.assertTrue((await self.bot.cmd_draw(**kwargs, matched_groups={'payload': 'edit background'}))[0])
            self.assertEqual(source.call_args.args[1:4], ('chat', '', kwargs['message']))
        self.assertEqual(kwargs, before)

    def test_real_sdk_metadata_opts_in_all_three_tools_and_keeps_command_suffix_closed(self):
        components = {item['name']: item['metadata'] for item in self.bot.get_components()}
        for name in ('neko_draw', 'neko_edit_image', 'neko_draw_status'):
            self.assertEqual(components[name]['trusted_context'], 'neko.tools.context.v1')
        pattern = components['neko_draw_cmd']['command_pattern']
        self.assertIsNotNone(re.fullmatch(pattern, '/draw cat'))
        self.assertIsNone(re.fullmatch(pattern, '/draw@anybot cat'))
