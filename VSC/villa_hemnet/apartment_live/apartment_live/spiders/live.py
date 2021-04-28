#How to run script,
#pwd:/Users/joeriksson/IdeaProjects/hemnet/sold_items/sold_items/spiders
#run: scrapy crawl live_items

import scrapy
import time
import json
#import cfscrape
#from fake_useragent import UserAgent
#from .decoder import Decoder
from scrapy import signals
from pydispatch import dispatcher


class QuotesSpider(scrapy.Spider):
    name = "live"
    # *** Change this url for your prefered search from hemnet.se ***
    #start_urls = ['https://www.hemnet.se/bostader?location_ids%5B%5D=925968&location_ids%5B%5D=925970&location_ids%5B%5D=898472&item_types%5B%5D=bostadsratt&rooms_min=2&rooms_max=4&living_area_min=45&living_area_max=65&price_min=3000000&price_max=6000000&keywords=balkong']
    start_urls = ['https://www.hemnet.se/bostader?location_ids%5B%5D=475414']
    globalIndex = 0
    results = {}

    def __init__(self):
        dispatcher.connect(self.spider_closed, signals.spider_closed)


    def parse(self, response):
        for ad in response.css("ul.normal-results > li.js-normal-list-item > a::attr('href')"):
            adUrl = ad.get()
            #ua = UserAgent(cache=False)
            #token, agent = cfscrape.get_tokens(adUrl, ua['google chrome'])
            #yield scrapy.Request(url=adUrl, cookies=token, headers={'User-Agent': agent}, callback=self.parseAd)
            time.sleep(1.5)
            yield scrapy.Request(url=adUrl, callback=self.parseAd)


        nextPage = response.css("a.next_page::attr('href')").get()
        if nextPage is not None:
            time.sleep(2)
            yield response.follow(nextPage, callback=self.parse)

    def parseAd(self, response):
        hemnetUrl = response.request.url
        address = response.css("div.property-info__primary-container > div.property-info__address-container > div.property-address > h1.qa-property-heading::text").get()
        area = response.css("div.property-info__primary-container > div.property-info__address-container > div.property-address > div.property-address__area-container > span.property-address__area::text").get()

        price = response.css("div.property-info__primary-container > div.property-info__price-container > p.qa-property-price::text").get()
        if price != None:
            price = price.replace('kr', '')
            price = price.replace(u'\xa0', '')

        attrData = {}
        for attr in response.css("div.property-info__attributes-and-description > div.property-attributes > div.property-attributes-table > dl.property-attributes-table__area > div.property-attributes-table__row"):
            attrLabel = attr.css("dt.property-attributes-table__label::text").get()
            attrValue = attr.css("dd.property-attributes-table__value::text").get()
            if attrLabel != None:
                if attrLabel == "Förening":
                    attrValue = attr.css("dd.property-attributes-table__value > div.property-attributes-table__housing-cooperative-name > span.property-attributes-table__value::text").get()

                attrValue = attrValue.replace(u'\xa0', '')
                attrValue = attrValue.replace(u'\n', '')
                attrValue = attrValue.replace(u'\t', '')
                attrValue = attrValue.replace('kr/mån', '')
                attrValue = attrValue.replace('kr/år', '')
                attrValue = attrValue.replace('kr/m²', '')
                attrValue = attrValue.replace('m²', '')
                attrValue = attrValue.strip()

                attrData[attrLabel] = attrValue

        description = ""
        for descr in response.css("div.property-description--long > p"):
            descrTxt = descr.css("p::text").get()
            if descrTxt != None and descrTxt != "":
                description = description + "\n" + descrTxt

        agencyUrl = response.css("a.property-description-broker-button::attr('href')").get()


        showings = {}
        i = 0
        for showing in response.css("ul.listing-showings__list > li"):
            showingTime = showing.css("div.listing-showings__showing-info > span.listing-showings__showing-time::text").get()
            if showingTime != None:
                showingTime = showingTime.replace(u'\xa0', '')
                showingTime = showingTime.replace(u'\n', '')
                showingTime = showingTime.replace(u'\t', '')
                showingTime = showingTime.strip()

            showingDesc = showing.css("div.listing-showings__showing-description::text").get()
            if showingTime != None:
                showingDesc = showingDesc.replace(u'\xa0', '')
                showingDesc = showingDesc.replace(u'\n', '')
                showingDesc = showingDesc.replace(u'\t', '')
                showingDesc = showingDesc.strip()

            showings[i] = {
                "time": showingTime,
                "description": showingDesc
            }
            i = i + 1

        self.results[self.globalIndex] = {
            "hemnetUrl": hemnetUrl,
            "address": address,
            "area": area,
            "price": price,
            "attributes": attrData,
            "description": description,
            "agencyUrl": agencyUrl,
            "showings": showings,
        }
        self.globalIndex = self.globalIndex + 1

    def spider_closed(self, spider):
        with open('live.json', 'w') as fp:
            json.dump(self.results, fp)

