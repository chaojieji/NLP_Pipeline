#coding=utf-8
import http.client
import hashlib
from urllib import parse
import random

# TODO 未完善的模块


appid = '20181025000225424'
secretKey = '*****'
httpClient = None
myurl = '/api/trans/vip/translate'
q = '苹果是一家很棒的公司\n我爱北京天安门'
fromLang = 'zh'
toLang = 'en'
salt = random.randint(32768, 65536)
sign = appid+q+str(salt)+secretKey
m1 = hashlib.md5()
m1.update(sign.encode(encoding='utf-8'))
sign = m1.hexdigest()
myurl = myurl+'?appid='+appid+'&q='+parse.quote(q)+'&from='+fromLang\
        +'&to='+toLang+'&salt='+str(salt)+'&sign='+sign

file = open('result.txt','w')
try:
    httpClient = http.client.HTTPConnection('api.fanyi.baidu.com')
    httpClient.request('GET', myurl)
    response = httpClient.getresponse()
    str = response.read().decode('utf-8')
    str = eval(str)
    for line in str['trans_result']:
        file.write(line['dst']+'\n')
except Exception as e:
    print(e)
finally:
    if httpClient:
        httpClient.close()
file.close()

