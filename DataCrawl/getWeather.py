import pymssql
import requests
from bs4 import BeautifulSoup
import json
import time
from pypinyin import lazy_pinyin
from selenium import webdriver


# 浏览器请求头
headers = {
    'User-Agent': 'Mozilla/5.0 (Linux; Android 5.0; SM-G900P Build/LRX21T) AppleWebKit/537.36 '
                  '(KHTML, like Gecko) Chrome/67.0.3396.62 Mobile Safari/537.36'}

driver = webdriver.PhantomJS()

server = '127.0.0.1'
user = 'sa'
password = 'amon@1991'
database = 'Rider'
conn = pymssql.connect(server, user, password, database)
ids = []
citys = []
dts = []

db_name = '[Logistics_Test_new]'
# start_num = 19200
# 19700
start_num = 0

cursor = conn.cursor()
# 查询操作
cursor.execute("SELECT [id],[DT],[city] FROM [Rider].[dbo]." + db_name + " Where weather='' ORDER BY [ID]")
row = cursor.fetchone()
while row:
    ids.append(row[0])
    dts.append(row[1])
    citys.append(row[2])
    row = cursor.fetchone()

length = len(citys)
# print(length)
# length = 120

for i in range(start_num, length):
    print('查询中--' + str(ids[i]))
    if citys[i] is None:
        continue
    if citys[i] == '山西省/长治市':
        city_name = 'changzhi'
    elif citys[i] == '辽宁省/朝阳市':
        city_name = 'chaoyang'
    elif citys[i] == '河南省/洛阳市':
        city_name = 'lvyang'
    elif citys[i] == '安徽省/六安市':
        city_name = 'liuan'
    elif citys[i] == '安徽省/蚌埠市':
        city_name = 'bangbu'
    elif '市' in citys[i].split('/')[1]:
        city_pinyin = lazy_pinyin(citys[i].split('/')[1][:-1])
        # 获取城市的拼音
        city_name = ''.join(city_pinyin)
    elif '地区' in citys[i].split('/')[1]:
        city_name = ''.join(lazy_pinyin(citys[i].split('/')[1][:-2]))
    elif '自治' in citys[i].split('/')[1]:
        city_name = ''.join(lazy_pinyin(citys[i].split('/')[1][:2]))
    else:
        city_name = ''.join(lazy_pinyin(citys[i].split('/')[1]))
    dts_sp = dts[i].split('-')
    # dt = ''.join(dts[i].split('/'))
    for j in range(len(dts_sp)):
        if len(dts_sp[j]) < 2:
            dts_sp[j] = '0' + dts_sp[j]
    dt = ''.join(dts_sp)
    print(dt)
    print(citys[i])
    url1 = 'http://www.tianqihoubao.com/lishi/' + city_name + '/' + dt + '.html'
    print(url1)
    driver.get(url1)
    elem = driver.find_elements_by_tag_name("tr")
    # print(soup_tr)
    # print(len(soup_tr))
    if len(elem) != 4:
        weather = ''
        temperature = ''
        wind = ''
    else:
        wea_list = elem[1].text.strip().split()
        temp_list = elem[2].text.strip().split()
        wind_list = elem[3].text.strip().split()
        weather = '/'.join(wea_list[1:])
        temperature = '/'.join(temp_list[1:])
        wind = '/'.join(wind_list[1:])

    print('写入中---' + str(ids[i]))
    # print(weather, temperature, wind)
    sql = "update Rider.dbo."+ db_name+" Set weather = '%s', temperature = '%s', wind = '%s' Where ID = %s"\
              % (weather, temperature, wind, str(ids[i]))
    print(sql)
    cursor.execute(sql)
    if (i) % 100 == 0:
        conn.commit()
        print('已提交' + str(ids[i]) + '条数据！')
        print('休息2秒钟~')
        time.sleep(2)
conn.commit()
print('已提交所有数据！')
conn.close()
