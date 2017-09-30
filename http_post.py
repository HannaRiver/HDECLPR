import urllib
import json
import urllib.request
import clpr as clpr


def http_post(path):
    url = '127.0.0.1'
    values = clpr.clpr(path)[0]
    jdata = json.dumps(values)
    req = urllib.request.Request(url, jdata) # 生成页面请求的完整数据
    response = urllib.request.urlopen(req) # 发送页面请求
    return response.read()