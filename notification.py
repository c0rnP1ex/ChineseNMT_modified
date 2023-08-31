import requests
import json

class Notification:

    def __init__(self, content, summary):
        # wxpusher
        if content == 'start':
            self.appToken = "AT_LWF80qfeNue1VWUGlw57EMVzXYIiiikQ"
        elif content == 'finish':
            self.appToken = "AT_PQ10o59q64hJvRbSz0ZMNsPcGCcjUT8k"
        else:
            self.appToken = "AT_J477WO9VnBGriU57Yz6x0XXfQ9TSy1DH"
        self.headers = {'content-type': "application/json"}
        self.body = {
        "appToken":self.appToken,
        "content": content,
        "summary": summary,
        "contentType":1,
        "topicIds":[],
        "uids":["UID_dDXjgk2VdGi42TyiTLvHZ1vzD8r9"]
        }

        ret = requests.post('http://wxpusher.zjiecode.com/api/send/message', data=json.dumps(self.body), headers=self.headers)
        pass


if __name__ == "__main__":
    Notification(content='阴暗地爬行', summary='分词模型训练开始了喵')
    Notification(content='光明地爬行', summary='分词模型训练开始了喵')