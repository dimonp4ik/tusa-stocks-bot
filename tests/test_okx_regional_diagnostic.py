from unittest.mock import patch
from src import okx_trader as okx


def test_explicit_region_probe_cannot_submit_or_choose_an_arbitrary_host():
    creds=dict(api_key='TEST',api_secret='TEST',passphrase='TEST')
    with patch.object(okx.requests,'post') as post,patch.object(okx.requests,'get') as get:
        assert not okx._request(creds,'POST','/api/v5/trade/order',
            base_url='https://eea.okx.com')[0]
        assert not okx._request(creds,'GET','/api/v5/account/config',
            base_url='https://example.com')[0]
    post.assert_not_called();get.assert_not_called()


def test_region_probe_uses_only_official_eu_configuration_endpoint():
    creds=dict(api_key='TEST',api_secret='TEST',passphrase='TEST')
    with patch.object(okx.requests,'get') as get:
        get.return_value.json.return_value=dict(code='0',data=[dict(uid='123',posMode='net_mode',perm='read_only')])
        good,config=okx.get_account_configuration(creds,base_url='https://eea.okx.com')
    assert good and config['account_uid']=='123'
    assert get.call_args.args[0]=='https://eea.okx.com/api/v5/account/config'
