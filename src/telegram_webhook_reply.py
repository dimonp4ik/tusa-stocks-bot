"""Use Telegram's single Bot API method returned with a webhook response."""
import re

from flask import g, has_request_context, jsonify


_PLAIN_MARKDOWN = re.compile(r'(?<!\w)(`|\*|_)([^\n]+?)\1(?!\w)')


def begin(chat_id, callback_id=None):
    g.telegram_reply = dict(chat_id=chat_id, callback_id=callback_id, operation=None)


def queue(method, arguments, *, priority=1):
    if not has_request_context():
        return False
    reply = getattr(g, 'telegram_reply', None)
    if not reply:
        return False
    if method == 'answerCallbackQuery':
        if not reply['callback_id'] or arguments.get('callback_query_id') != reply['callback_id']:
            return False
    elif (method not in ('sendMessage', 'editMessageText') or not reply['chat_id']
          or arguments.get('chat_id') != reply['chat_id']):
        return False
    operation = reply['operation']
    if operation is None or priority > operation[0]:
        payload = dict(arguments, method=method)
        # There is no subsequent API response for retrying a formatting error.
        if payload.pop('parse_mode', None) and payload.get('text'):
            payload['text'] = _PLAIN_MARKDOWN.sub(r'\2', payload['text'])
        reply['operation'] = (priority, payload)
    return True


def response():
    reply = getattr(g, 'telegram_reply', None)
    if reply and reply['operation']:
        return jsonify(reply['operation'][1])
    return 'ok', 200
