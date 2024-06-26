import requests
import random
import json
from hashlib import md5

# Set your own appid/appkey.
appid = '20240506002043059'
appkey = 'nX4lLn5YAM4l9X2pPe8S'

# For list of language codes, please refer to `https://api.fanyi.baidu.com/doc/21`
from_lang = 'yue'
to_lang =  'zh'

endpoint = 'http://api.fanyi.baidu.com'
path = '/api/trans/vip/translate'
url = endpoint + path

query = '香港原本係一個人煙稀少嘅漁港,但係自從英國人嚟咗之後'

# Generate salt and sign
def make_md5(s, encoding='utf-8'):
    return md5(s.encode(encoding)).hexdigest()

salt = random.randint(32768, 65536)
sign = make_md5(appid + query + str(salt) + appkey)

# Build request
headers = {'Content-Type': 'application/x-www-form-urlencoded'}
payload = {'appid': appid, 'q': query, 'from': from_lang, 'to': to_lang, 'salt': salt, 'sign': sign}

# Send request
r = requests.post(url, params=payload, headers=headers)
result = r.json()

# Show response
print(json.dumps(result, indent=4, ensure_ascii=False))
