# Scraper that get data from Hemnet for sold apparmnets, input is url

#How to run script,
#pwd:/Users/joeriksson/IdeaProjects/hemnet/sold_items/sold_items/spiders
#run: scrapy crawl sold_items

import scrapy
from pydispatch import dispatcher
import time
import json
from scrapy import signals
import re
import scrapy


class villa_hemnetSpider(scrapy.Spider):
    name = "villa"
    #Hous price in avesta 
    #start_urls = ["https://www.hemnet.se/salda/bostader?location_ids%5B%5D=17874&item_types%5B%5D=villa&item_types%5B%5D=radhus"]
    
    #house prices in stockholm
    start_urls =["https://www.hemnet.se/salda/bostader?location_ids%5B%5D=925970&location_ids%5B%5D=925968&living_area_min=45&living_area_max=100&sold_age=3m"] 
    counter = 0
    results = {}

    def __init__(self):
        dispatcher.connect(self.spider_closed, signals.spider_closed)
    
    def parse(self, response):
        # to get hous price 
        for ad in response.css("ul.sold-results > li.sold-results__normal-hit > a.hcl-card::attr('href')"): 
        #for ad in response.css("ul.sold-results > li.sold-results__normal-hit > a.hcl-card::attr('href')"):
            print(ad.get())
            time.sleep(1.5)
            #print(ad.text())
            yield scrapy.Request(url=ad.get(), callback=self.parseInnerPage)

        nextPage = response.css("a.next_page::attr(href)").get()
        print(nextPage)
        if nextPage is not None:
            time.sleep(0.5)
            yield response.follow(nextPage, self.parse)
            
    def parseInnerPage(self, response):
        streetName = response.xpath('//h1//text()').extract()
        streetName = streetName[2:]
        streetName = ''.join(streetName)
        #print(streetName)

        #summary_info = summary_info[0:1]
        sold_date = response.xpath('//p//text()').extract()[6]
        #print(sold_date)
        summary_info = response.xpath('//p//text()').extract()
        summary_info = summary_info[2:8]
        summary_info = ''.join(summary_info)
        summary_info = summary_info.replace(u"\n","")
        #ext_sub = response.css('div.sold-property qa-sold-property > div.property-map > div.data-initial-data').get()

        #get area code
        #summary_info = response.xpath('//p[@class="sold-property__metadata"]//text()').extract()
        print('steat_name', streetName)
        print('summary',summary_info)

        area=re.findall('(?<=-)  [^<=,]*', summary_info)
        area = ''.join(area)
        print('NEW',area)
        
        #print("TESTAR", summary_info)
        brok_name = (response.css('div.broker-card__info > p.broker-card__text > strong::text').extract_first())
        broaker_company = (response.css('div.broker-card__info > a.hcl-link::text ').extract_first())
        print("broaker", broaker_company)
        price=(response.css("div.sold-property__price > span.sold-property__price-value::text").get())
        price= price.replace("kr","")
        price= price.replace(u"\xa0","")
        
        print('Price', price)
        
        datadict_first= {}
        second_dict = {}
        for atts,info in zip(response.css('div.sold-property__details > dl.sold-property__price-stats > dt'),
                             response.css('div.sold-property__details > dl.sold-property__price-stats > dd')):
            #print(atts.css('dt.sold-property__attribute::text').get())
            label=atts.css('dt.sold-property__attribute::text').get()
            # Extract the name of first box
            label = label.replace(u"\n", "")
            label = label.replace(u"\n", "")
            label = label.replace(u"\t", "")
            label = label.replace(u'\xa0', '')
            # Exctrakt the value from first boxj
            value=info.css('dd.sold-property__attribute-value::text').get()
            value = value.replace(u"\n", "")
            value = value.replace(u"\t", "")
            value = value.replace(u'\xa0', '')
            value = value.replace('kr/mån', '')
            value = value.replace('kr/år', '')
            value = value.replace('kr/m²', '')
            value = value.replace('m²', '')
            value = value.replace('kr', '')
            datadict_first[label]=value
        print(datadict_first)

        for atts,info in zip(response.css('div.sold-property__details > dl.sold-property__attributes > dt'),
                              response.css('div.sold-property__details > dl.sold-property__attributes > dd')):
             #print(atts.css('dt.sold-property__attribute::text').get())
             label=atts.css('dt.sold-property__attribute::text').get()
             # Extract the name of first box
             label = label.replace(u"\n", "")
             label = label.replace(u"\n", "")
             label = label.replace(u"\t", "")
             label = label.replace(u'\xa0', '')
             # Exctrakt the value from first boxj
             value=info.css('dd.sold-property__attribute-value::text').get()
             value = value.replace(u"\n", "")
             value = value.replace(u"\t", "")
             value = value.replace(u'\xa0', '')
             value = value.replace('kr/mån', '')
             value = value.replace('kr/år', '')
             value = value.replace('kr/m²', '')
             value = value.replace('m²', '')
             value = value.replace('kr', '')
             second_dict[label]=value
        print(second_dict)

        #Store the reuslt in dictonatnry
        self.results[self.counter] = {
            "streetName": streetName,
            "price": price,
            "info": summary_info,
            "area": area,
            "broker": broaker_company,
            "Name_brocker": brok_name ,
            "date_sold": sold_date,
            "firs_att": datadict_first,
            "second_att":second_dict,
            #"test": info_test,
        }
        self.counter = self.counter + 1

    def spider_closed(self, spider):
        with open('results.json','w') as fp:
            json.dump(self.results, fp)       