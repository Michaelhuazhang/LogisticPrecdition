# 经纬度转换为城市
# 使用百度地图API将经纬度数据转化为城市名并保存起来，用于后续的天气和油价数据爬取做准备。
import pymssql
import requests
from bs4 import BeautifulSoup
import json
import time

# 浏览器请求头
headers = {
    'User-Agent': 'Mozilla/5.0 (Linux; Android 5.0; SM-G900P Build/LRX21T) AppleWebKit/537.36 '
                  '(KHTML, like Gecko) Chrome/67.0.3396.62 Mobile Safari/537.36'}
# 百度API开发者AK
ak = ''

server = '127.0.0.1'
# 账户密码
user = ''
password = ''
database = 'Rider'
conn = pymssql.connect(server, user, password, database)
ids = []
long = []
lati = []

db_name = '[Logistics_Missing]'
start_num = 0

cursor = conn.cursor()
# 查询操作
cursor.execute('SELECT [ID],[longStart],[latiStart] FROM [Rider].[dbo].' + db_name + ' ORDER BY [ID]')
row = cursor.fetchone()
while row:
    # print("long=%s, lati=%s" % (row[0], row[1]))
    ids.append(row[0])
    long.append(row[1])
    lati.append(row[2])
    row = cursor.fetchone()

# print(len(long))
lenn = len(long)
# lenn = 120
with requests.Session() as s:
    for i in range(start_num, lenn):
        print('查询中--' + str(ids[i]))
        url1 = 'http://api.map.baidu.com/geocoder/v2/?ak=' + ak + '&location=' \
              + lati[i] + ',' + long[i] + '&output=json&pois=0'
        wb_data1 = s.get(url1, headers=headers)
        # print(wb_data1)
        # wb_data = requests.get(url, headers=headers)
        soup1 = BeautifulSoup(wb_data1.text, 'html.parser')
        # print(soup1)
        json_soup1 = json.loads(soup1.get_text())
        comp1 = json_soup1['result']['addressComponent']
        city1 = comp1['province']+'/'+comp1['city']
        sql = "update Rider.dbo." + db_name + " Set city = '%s' Where ID = %s" % (city1, ids[i])
        print(sql)
        cursor.execute(sql)
        if (i+1)%200 == 0:
            conn.commit()
            print('已提交' + str(ids[i]) + '条数据！')
            print('休息2秒！')
            time.sleep(2)

conn.commit()
print('已提交所有数据！')
conn.close()

