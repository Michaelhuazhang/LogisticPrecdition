import sqlutil
from selenium import webdriver
import urllib
import time

conn = sqlutil.conn
cursor = conn.cursor()

sql = 'Select DISTINCT city FROM Logistics_Train_new'
provinces = []
cursor.execute(sql)
row = cursor.fetchone()
while row:
    # print(row)
    city_j = row[0].split('/')[0]
    if '省' in city_j:
        city = city_j.split('省')[0]
    elif '市' in city_j:
        city = city_j.split('市')[0]
    elif '内蒙古' in city_j:
        city = '内蒙古'
    else:
        city = city_j[:2]
    provinces.append(city)
    row = cursor.fetchone()

provinces = list(set(provinces))
print(len(provinces))
print(provinces)

driver = webdriver.PhantomJS()

if __name__ == '__main__':
    cnt = 1
    provinces = ['山西']
    for prov in provinces:
        print(cnt)
        cnt += 1
        url_prov = urllib.parse.quote(prov)
        # print(url_city)
        url = 'http://data.eastmoney.com/cjsj/oil_city.aspx?city=' + url_prov
        print(url)
        driver.get(url)
        while True:
            pages = driver.find_elements_by_xpath("//div[@id='PageCont']//a")
            datas = driver.find_elements_by_xpath("//table[@id='dt_2']//tbody//tr")
            for i in range(len(datas)):
                tds = datas[i].find_elements_by_tag_name("td")
                try:
                    dt = tds[0].text
                    oil = tds[-2].text
                    print(prov, dt, oil)
                    cursor.execute("Insert INTO Oil_Price(province, dt, oil) VALUES ('%s', '%s', '%s')" % (prov, dt, oil))
                except:
                    print('tds未获取！')
                    continue
            if len(pages) == 0 or pages[-2].get_attribute('class') == 'nolink':
                break
            pages[-2].click()
            time.sleep(3)
        conn.commit()
