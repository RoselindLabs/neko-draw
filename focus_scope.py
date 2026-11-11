"""Host reply tool schemas are scope evidence; prompt text is not authority."""
import html
import re

HOST_ATTRIBUTES = frozenset({
    'msg_id', 'session_id', 'chat_name', 'quote', 'time', 'user', 'group_card', 'is_self_message',
})


def host_header(text):
    """Read only the offset-zero prefix line, even when Host merges body text."""
    if not isinstance(text, str) or '\n' not in text[:2059]:
        return None
    line = text[:2059].partition('\n')[0]
    prefix = re.fullmatch(r'<message ([^>\r\n]{1,2048})>', line)
    if not prefix or not re.fullmatch(r'[a-z_]+="[^"<>\r\n]*"(?: [a-z_]+="[^"<>\r\n]*")*', prefix[1]):
        return None
    pairs = re.findall(r'([a-z_]+)="([^"]*)"', prefix[1])
    attrs = dict(pairs)
    if (len(pairs) != len(attrs) or not attrs.keys() <= HOST_ATTRIBUTES
            or not attrs.get('msg_id') or not attrs.get('session_id')
            or ('is_self_message' in attrs and attrs['is_self_message'] != 'true')):
        return None
    return attrs


def planner_scopes(session_id, definitions):
    if isinstance(session_id, str) and session_id.strip():
        return (session_id,)
    if not isinstance(definitions, list) or len(definitions) > 512:
        return ()
    replies = [tool.get('function') for tool in definitions if isinstance(tool, dict)
               and tool.get('type') == 'function' and isinstance(tool.get('function'), dict)
               and tool['function'].get('name') == 'reply']
    if len(replies) != 1:
        return ()
    schema = replies[0].get('parameters')
    if not isinstance(schema, dict):
        return ()
    required, properties = schema.get('required'), schema.get('properties')
    if not isinstance(required, list) or 'session_id' not in required or not isinstance(properties, dict):
        return ()
    prop = properties.get('session_id')
    scopes = prop.get('enum') if isinstance(prop, dict) and prop.get('type') == 'string' else None
    if not isinstance(scopes, list) or not 0 < len(scopes) <= 32:
        return ()
    if any(not isinstance(s, str) or not 0 < len(s) <= 128 or s.strip() != s
           or any(ord(c) < 32 for c in s) for s in scopes):
        return ()
    return tuple(dict.fromkeys(scopes))


def items_for_scope(items, scope):
    result = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict) or item.get('item_type') != 'UserMessageItem':
            continue
        parts = item.get('parts')
        first = parts[0] if isinstance(parts, list) and parts else None
        text = first.get('text') if isinstance(first, dict) and first.get('type') == 'text' else None
        attrs = host_header(text)
        if attrs is None:
            continue
        if html.unescape(attrs.get('session_id', '')) == scope and attrs.get('is_self_message') != 'true':
            result.append(item)
    return result
