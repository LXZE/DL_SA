import requests as req
import time

url = 'https://notify-api.line.me/api/notify'
header = {'Authorization': 'Bearer PFFa34NPGcICWYWMDlLTF6K6f48n17K9ZRwZQVREmY0'}
data = {'message': f'test notification at time {time.ctime()}'}

res = req.post(url, data=data, headers=header)
print(res)