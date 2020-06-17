import scrapy
from w3lib.html import remove_tags
from w3lib.html import replace_escape_chars
import lxml
import datetime
import re
import json


def clear_text(text: str):
    text = text.replace('\n', '')
    text = re.sub('\s+', ' ', text)
    text = text.replace(u'\xa0', u' ')
    return text


def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


class DOMAINS:
    DZIENNIK = 'dziennik'
    GAZETA = 'gazeta'
    INTERIA = 'interia'
    PAP = 'pap'
    RADIO_ZET = 'radio_zet'
    RMF = 'rmf'
    TVN24 = 'tvn24'
    TVN24bis = 'tvn24bis'
    TVP_INFO = 'tvp_info'
    POLSKIE_RADIO = 'polskie_radio'
    WPROST = 'wprost'
    POLSAT_NEWS = 'polsat_news'
    NIEZALEZNA = 'niezalezna'
    WPOLITYCE = 'w_polityce'
    DO_RZECZY = 'do_rzeczy'
    TOK_FM = 'tok_fm'
    ONET = 'onet'


DOMAIN_URLS = {
    # DOMAINS.DZIENNIK: ['https://wiadomosci.dziennik.pl/polityka,',  # Polityka
    #                    'https://wiadomosci.dziennik.pl/wydarzenia,',  # Polska
    #                    'https://wiadomosci.dziennik.pl/swiat,',  # Swiat
    #                    'https://wiadomosci.dziennik.pl/media,',  # Media
    #                    'https://gospodarka.dziennik.pl/news,'],  # Gospodarka
    DOMAINS.DZIENNIK: ['https://www.dziennik.pl/archiwum/'],
    DOMAINS.GAZETA: ['http://wiadomosci.gazeta.pl/wiadomosci/0,114871.html?str=',  # Najnowsze
                     'http://wiadomosci.gazeta.pl/wiadomosci/0,114884.html?str=',  # Polityka
                     'http://wiadomosci.gazeta.pl/wiadomosci/0,114883.html?str=',  # Polska
                     'http://wiadomosci.gazeta.pl/wiadomosci/0,114881.html?str='],  # Swiat
    DOMAINS.INTERIA: ['https://fakty.interia.pl/wiadomosci-lokalne,nPack,',  # Wiadomosci lokalne
                      'https://fakty.interia.pl/polska,nPack,',  # Polska
                      'https://fakty.interia.pl/swiat,nPack,'],  # Swiat
    DOMAINS.PAP: ['https://www.pap.pl/kraj?page=',  # Polska
                  'https://www.pap.pl/swiat?page=',  # Swiat
                  'https://www.pap.pl/gospodarka?page='],  # Ekonomia
    DOMAINS.RADIO_ZET: ['https://wiadomosci.radiozet.pl/Polska/(offset)/',  # Polska
                        'https://wiadomosci.radiozet.pl/Swiat/(offset)/',  # Swiat
                        'https://biznes.radiozet.pl/Newsy/(offset)/'],  # Swiat
    DOMAINS.RMF: ['https://www.rmf24.pl/fakty,nPack,'],
    DOMAINS.TVN24: ['https://tvn24.pl/najnowsze/'],
    DOMAINS.TVN24bis: ['https://tvn24bis.pl/najnowsze,72/'],
    DOMAINS.TVP_INFO: ['https://www.tvp.info/191866/polska?page=',
                       'https://www.tvp.info/191867/swiat?page='],
    DOMAINS.POLSKIE_RADIO: ['https://www.polskieradio24.pl/POLSKA/Tag295/Strona',
                            'https://www.polskieradio24.pl/SWIAT/Tag6638/Strona'],
    DOMAINS.WPROST: ['https://biznes.wprost.pl/gospodarka/',
                     'https://www.wprost.pl/kraj/',
                     'https://www.wprost.pl/swiat/'],
    DOMAINS.POLSAT_NEWS: ['https://www.polsatnews.pl/wyszukiwarka/?text=polska&type=event&page=',
                          'https://www.polsatnews.pl/wyszukiwarka/?text=swiat&type=event&page=',
                          'https://www.polsatnews.pl/wyszukiwarka/?text=polityka&page=',
                          'https://www.polsatnews.pl/wyszukiwarka/?text=Biznes&type=event&page='],
    DOMAINS.NIEZALEZNA: ['https://niezalezna.pl/dzial/polska/',
                         'https://niezalezna.pl/dzial/swiat/',
                         'https://niezalezna.pl/dzial/gospodarka/'],
    DOMAINS.WPOLITYCE: ['https://wpolityce.pl/api/polityka/publikacje?page=',
                        'https://wpolityce.pl/api/swiat/publikacje?page=',
                        'https://wpolityce.pl/api/gospodarka/publikacje?page='],
    DOMAINS.DO_RZECZY: ['https://dorzeczy.pl/kraj/',
                        'https://dorzeczy.pl/swiat/',
                        'https://dorzeczy.pl/ekonomia/',
                        'https://dorzeczy.pl/szukaj/polityka/',
                        'https://dorzeczy.pl/szukaj/polska/'],
    # DOMAINS.TOK_FM: ['http://www.tokfm.pl/Tokfm/0,103085.html?str=',
    #                  'http://www.tokfm.pl/Tokfm/0,103086.html?str=',
    #                  'http://www.tokfm.pl/Tokfm/0,103087.html?str=',
    #                  'http://www.tokfm.pl/Tokfm/0,103090.html?str='],
    DOMAINS.TOK_FM: ['http://www.tokfm.pl/Tokfm/3660000,0,,1,,'],
    DOMAINS.ONET: ['https://wiadomosci.onet.pl/archiwum/']
}
# DOMAINS.GAZETA_PRAWNA: ['https://www.gazetaprawna.pl/wiadomosci,',
#                         'https://podatki.gazetaprawna.pl/dzial/wiadomosci,',
#                         'https://praca.gazetaprawna.pl/dzial/wiadomosci,']

DOMAIN_ENDINGS = {
    DOMAINS.DZIENNIK: [''],
    DOMAINS.GAZETA: ['',  # Najnowsze
                     '_19834947',  # Polityka
                     '',  # Polska
                     ''],  # Swiat
    DOMAINS.INTERIA: ['',  # Wiadomosci lokalne
                      '',  # Polska
                      ''],  # Swiat
    DOMAINS.PAP: ['',  # Polska
                  '',  # Swiat
                  ''],  # Ekonomia
    DOMAINS.RADIO_ZET: ['',  # Polska
                        '',  # Swiat
                        ''],  # Swiat
    DOMAINS.RMF: [''],
    DOMAINS.TVN24: [''],
    DOMAINS.TVN24bis: [''],
    DOMAINS.TVP_INFO: ['',
                       ''],
    DOMAINS.POLSKIE_RADIO: ['',
                            ''],
    DOMAINS.WPROST: ['',
                     '',
                     ''],
    DOMAINS.POLSAT_NEWS: ['',
                          '',
                          '',
                          ''],
    DOMAINS.NIEZALEZNA: ['',
                         '',
                         ''],
    DOMAINS.WPOLITYCE: ['',
                        '',
                        ''],
    DOMAINS.DO_RZECZY: ['',
                        '',
                        '',
                        '',
                        ''],
    # DOMAINS.TOK_FM: ['_22958567',
    #                  '_22958563',
    #                  '_22958568',
    #                  '_22960069'],
    DOMAINS.TOK_FM: [''],
    DOMAINS.ONET: ['']
}

ALLOWED_DOMAINS = {
    DOMAINS.DZIENNIK: ['wiadomosci.dziennik.pl',
                       'gospodarka.dziennik.pl'
                       ],
    DOMAINS.GAZETA: None,
    DOMAINS.INTERIA: None,
    DOMAINS.PAP: None,
    DOMAINS.RADIO_ZET: None,
    DOMAINS.RMF: None,
    DOMAINS.TVN24: ['tvn24.pl'],
    DOMAINS.TVN24bis: ['tvn24bis.pl'],
    DOMAINS.TVP_INFO: None,
    DOMAINS.POLSKIE_RADIO: None,
    DOMAINS.WPROST: None,
    DOMAINS.POLSAT_NEWS: None,
    DOMAINS.NIEZALEZNA: None,
    DOMAINS.WPOLITYCE: None,
    DOMAINS.DO_RZECZY: None,
    DOMAINS.TOK_FM: ['tokfm.pl'],
    DOMAINS.ONET: ['wiadomosci.onet.pl']
    # DOMAINS.ONET: None
}

ARTICLES_LINKS = {
    # DOMAINS.DZIENNIK: '.itarticle a::attr("href")',
    DOMAINS.DZIENNIK: 'section.stream .dayInArchive a::attr("href")',
    DOMAINS.GAZETA: '.entry .article a ::attr("href")',
    DOMAINS.INTERIA: '.brief-list-item .tile-magazine-title-url ::attr("href")',
    DOMAINS.PAP: 'div.newsList div.imageWrapper a::attr("href")',
    DOMAINS.RADIO_ZET: 'div.list-element__image a::attr("href")',
    DOMAINS.RMF: '.article .thumbnail:not(.thumbnail.sponsored) .image ::attr("href")',
    DOMAINS.TVN24: '.wide-column .teaser-wrapper .link__content a.default-teaser__link::attr("href")',
    DOMAINS.TVN24bis: 'article div.photo-container a ::attr("href")',
    DOMAINS.POLSKIE_RADIO: '.article a.main-link ::attr("href")',
    # DOMAINS.WPROST: '.section-main-list .box-list-item .news-data a.news-open ::attr("href")',
    DOMAINS.WPROST: '.box-list-item .news-data::attr("data-href")',
    DOMAINS.POLSAT_NEWS: '#searchwrap article.news a::attr("href")',
    DOMAINS.NIEZALEZNA: 'div#content.mainpage div.columnLeft a.uitemUnderline::attr("href")',
    DOMAINS.DO_RZECZY: '#main-list .box-list-item div.news-data::attr("data-href")',
    # DOMAINS.TOK_FM: 'div.body li.entry a::attr("href")',
    DOMAINS.TOK_FM: '#results li a ::attr("href")',
    DOMAINS.ONET: 'section.stream .dayInArchive a::attr("href")'
}


class PageSpider(scrapy.Spider):
    name = "my_spider"

    def __init__(self, domain: str, ranges_start: str, ranges_end: str):
        '''Initialize spider for given domain'''
        self.domain = domain
        ranges_start = ranges_start.split(',')
        ranges_start = [int(i) for i in ranges_start]
        ranges_end = ranges_end.split(',')
        ranges_end = [int(i) for i in ranges_end]
        self.start_urls = []
        domain_urls = DOMAIN_URLS[domain]
        domain_endings = DOMAIN_ENDINGS[domain]
        if domain in [DOMAINS.ONET, DOMAINS.DZIENNIK]:
            this_day = datetime.datetime.now()
            start_date = this_day - datetime.timedelta(days=ranges_end[0])
            end_date = this_day - datetime.timedelta(days=ranges_start[0])
            delta = end_date - start_date

            date_list = [start_date + datetime.timedelta(days=x) for x in range(delta.days + 1)]
            date_list = [date.strftime("%Y-%m-%d") for date in date_list]
            self.start_urls += [domain_urls[0] + date for date in date_list]
        elif domain == DOMAINS.TOK_FM:
            this_day = datetime.datetime.now()
            start_date = this_day - datetime.timedelta(days=ranges_end[0])
            end_date = this_day - datetime.timedelta(days=ranges_start[0])
            delta = end_date - start_date

            date_list = [start_date + datetime.timedelta(days=x) for x in range(delta.days + 1)]
            date_list = [date.strftime("%Y-%m-%d") for date in date_list]
            self.start_urls += [domain_urls[0] + date + '.html' for date in date_list]
        else:
            for url_num in range(0, len(domain_urls)):
                self.start_urls += [domain_urls[url_num] + str(i) + domain_endings[url_num] for i in
                                    range(ranges_start[url_num], ranges_end[url_num])]

        self.allowed_domains = ALLOWED_DOMAINS[domain]

    # def make_requests_from_url(self, url):
    #     return scrapy.Request(url, dont_filter=True, meta={
    #         'dont_redirect': True,
    #         'handle_httpstatus_list': [301, 302]
    #     })

    def parse(self, response):
        if self.domain == DOMAINS.DZIENNIK:
            domain_parser = self.parse_dziennik
        elif self.domain == DOMAINS.GAZETA:
            domain_parser = self.parse_gazeta
        elif self.domain == DOMAINS.INTERIA:
            domain_parser = self.parse_interia
        elif self.domain == DOMAINS.PAP:
            domain_parser = self.parse_pap
        elif self.domain == DOMAINS.RADIO_ZET:
            domain_parser = self.parse_radiozet
        elif self.domain == DOMAINS.RMF:
            domain_parser = self.parse_rmf
        elif self.domain == DOMAINS.TVN24:
            domain_parser = self.parse_tvn24
        elif self.domain == DOMAINS.TVN24bis:
            domain_parser = self.parse_tvn24bis
        elif self.domain == DOMAINS.TVP_INFO:
            domain_parser = self.parse_tvp_info
        elif self.domain == DOMAINS.POLSKIE_RADIO:
            domain_parser = self.parse_polskie_radio
        elif self.domain == DOMAINS.WPROST:
            domain_parser = self.parse_wprost
        elif self.domain == DOMAINS.POLSAT_NEWS:
            domain_parser = self.parse_polsat_news
        elif self.domain == DOMAINS.NIEZALEZNA:
            domain_parser = self.parse_niezalezna
        elif self.domain == DOMAINS.WPOLITYCE:
            domain_parser = self.parse_wpolityce
        elif self.domain == DOMAINS.DO_RZECZY:
            domain_parser = self.parse_do_rzeczy
        elif self.domain == DOMAINS.TOK_FM:
            domain_parser = self.parse_tok_fm
        elif self.domain == DOMAINS.ONET:
            domain_parser = self.parse_onet
        else:
            print("Wrong domain: " + self.domain)

        if self.domain == DOMAINS.TVP_INFO:
            pattern = re.compile(r"window.__directoryData = ({.*?});", re.MULTILINE | re.DOTALL)
            data = response.xpath('//script[contains(., "window")]/text()')
            data = data.re(pattern)[0]
            data = json.loads(data)
            items = data['items']
            links = [item['url'] for item in items]
        elif self.domain == DOMAINS.WPOLITYCE:
            data = json.loads(response.body_as_unicode())
            links = [item.get('url', '') for item in data]
        elif self.domain == DOMAINS.ONET:
            links = response.css(ARTICLES_LINKS[self.domain]).extract()
            links = [link for link in links
                     if not any([garbage in link for garbage in
                                 ['losowanie-lotto', 'prognoza-pogody', '/pogoda/']])]
        else:
            links = response.css(ARTICLES_LINKS[self.domain]).extract()

        for article_url in links:
            article_url = re.sub('\s+', '', article_url)
            yield response.follow(article_url, callback=domain_parser)
    # https://docs.scrapy.org/en/latest/topics/dynamic-content.html

    def parse_dziennik(self, response):
        '''Parser for dziennik.pl'''
        url = response.url
        art_id = url.split('artykuly/')[1]
        art_id = art_id.split(',')[0]

        date = response.xpath("//meta[@property='article:published_time']/@content").extract()[0]
        date = date.split(' ')
        time = date[1]
        date = date[0]

        title = response.xpath("//meta[@property='og:title']/@content").extract()

        lead = response.css("article .lead::text").extract()
        lead = ' '.join(lead)
        lead = remove_tags(lead)

        text = response.css('article .detail p.hyphenate').extract()

        # W R usunac akapity ze zdjeciami oraz wpisami z twittera - https://t.co/ lub pic.twitter.com/               
        text = ' || '.join(text)
        text = remove_tags(text)

        # Joining lead with text
        text = ' || '.join([lead, text])
        text = clear_text(text)

        source = response.css(".articleFooter span[itemprop='name']::text").extract()
        tags = response.css(".relatedTopics .relatedTopic a::attr('title')").extract()
        yield {'id': art_id,
               'url': url,
               'date': date,
               'time': time,
               'title': ''.join(title),
               'lead': lead,
               'text': text,
               'source': ', '.join(source),
               'tags': ', '.join(tags)
               }

    def parse_gazeta(self, response):
        '''Parser for gazeta.pl'''
        url = response.url
        art_id = url.split(',')[2]

        date = response.css('.article_date time::attr("datetime")').extract_first()
        date = date.split(' ')
        time = date[1]
        date = date[0]

        title = response.css("h1#article_title::text").extract()
        title = ' '.join(title)
        title = replace_escape_chars(title)

        lead = response.css("#gazeta_article_lead").extract()
        lead = ' '.join(lead)
        lead = remove_tags(lead)

        text = response.css('p.art_paragraph').extract()
        text = ' || '.join(text)
        text = remove_tags(text)

        # Joining lead with text
        text = ' || '.join([lead, text])
        text = clear_text(text)

        autor = response.css(".article_author::text").extract()
        tags = response.css(".tags_list  .tags_item a::text").extract()

        yield {'id': art_id,
               'url': url,
               'date': date,
               'time': time,
               'title': ''.join(title),
               'lead': lead,
               'text': text,
               'autor': ', '.join(autor),
               'tags': ', '.join(tags)}

    def parse_interia(self, response):
        '''Parser for Interia'''
        url = response.url
        art_id = url.split('nId,')[1]

        date = response.css('.article-date ::attr("content")').extract_first()
        date = date.split('T')
        time = date[1]
        date = date[0]

        title = response.css("h1.article-title::text").extract()
        title = ' '.join(title)
        title = replace_escape_chars(title)

        lead = response.css(".article-body .article-lead::text").extract()
        lead = ' '.join(lead)
        lead = remove_tags(lead)

        # art_path = '//div[@class = "article-container"]/' \
        #            'div[not(*/@class = "embed")]/' \
        #            'p[not(/aside[@class = "embed embed-photo embed-center"])]'
        # text = response.xpath(art_path)

        exclude_selectors = (
            'not(self::*[contains(@class, "advert")])'
            ' and not(self::*[starts-with(text(), "ZOBACZ RÓWNIEŻ:")])'
            ' and not(self::*[starts-with(text(), "SPRAWDŹ:")])'
            ' and not(descendant-or-self::*[contains(@class, "sub")])'
            ' and not(descendant-or-self::*[contains(@class, "embed")])'
            ' and not(ancestor-or-self::*[contains(@class, "embed")])'
            ' and not(descendant-or-self::*[contains(@class, "aside")])'
            ' and not(ancestor-or-self::*[contains(@class, "aside")])'
            ' and not(descendant-or-self::*[contains(@class, "aside")])'
            ' and not(descendant-or-self::u)'
            ' and (self::p[not(contains(@dir, "ltr"))])'
        )
        selector = '//div[@class = "article-container"]/' \
                   'div[not(*/@class = "embed")]/' \
                   '*[%s]' % exclude_selectors
        text = response.xpath(selector)

        text = text.extract()
        text = ' || '.join(text)
        text = remove_tags(text)
        text = clear_text(text)

        source = response.css(".article-footer .article-source ::attr('content')").extract()
        yield {'id': art_id,
               'url': url,
               'date': date,
               'time': time,
               'title': ''.join(title),
               'lead': lead,
               'text': text,
               'source': ', '.join(source)
               }

    def parse_pap(self, response):
        url = response.url
        art_id = url.split('%2C')[1]

        date = response.css('article div.moreInfo').extract_first()
        date = date.split('</svg>')[1]
        date = date.split(', ')
        time = date[1][0:5]
        date = date[0]

        title = response.css("h1.title ::text").extract()
        title = ' '.join(title)
        title = replace_escape_chars(title)

        lead = response.css("article div.field.field--name-field-lead ::text").extract()
        lead = ' '.join(lead)

        text = response.css('article div.field.field--name-body p ::text').extract()

        # Czyszczenie tekstu
        text.pop()

        for i in range(0, len(text)):
            text[i] = remove_tags(text[i]).strip()

        if len(text[-1]) < 100:
            if 'arch.' in text[-1]:
                text.pop()
                if len(text[-1]) < 100:
                    autor = text[-1]
                    text.pop()
            else:
                autor = text[-1]
                text.pop()
        else:
            autor = ''

        if re.search('^(A|a)utor.*:', text[-1]) != None or len(text[-1]) < 100:
            text.pop()
            if len(text[-1]) < 100:
                text.pop()

        text[-1] = re.sub('(\(PAP\)|\(PAP Biznes\))', '', text[-1])

        text = ' || '.join(text)

        # Joining lead with text
        text = ' || '.join([lead, text])
        text = clear_text(text)
        tags = response.css("article div.field.field--name-field-tags  .tagsList .field--item a::text").extract()

        yield {'id': art_id,
               'url': url,
               'date': date,
               'time': time,
               'title': ''.join(title),
               'lead': lead,
               'text': text,
               'autor': autor,
               'tags': ', '.join(tags)}

    def parse_radiozet(self, response):
        url = response.url

        date = response.css('article .info-header__date--published__date::text').extract_first()
        # date = date.split(' ')
        # date = date[0]
        date = date.replace('.', '-')
        time = response.css('article .info-header__date--published__time::text').extract_first()

        title = response.css("article header .full__title.full__article__title::text").extract()
        title = ' '.join(title)
        title = replace_escape_chars(title)

        lead = response.css(".full__article__lead ::text").extract()
        lead = ' '.join(lead)
        lead = remove_tags(lead)
        lead = re.sub('\s+', ' ', lead)
        lead = re.sub(' \n', '', lead)

        exclude_selectors = (
            'not(ancestor::*[contains(@class, "advert")])'
            ' and not(ancestor::*[contains(@class, "embed__article")])'
            ' and not(ancestor::*[contains(@class, "SandboxRoot")])'
            ' and not(ancestor::*[contains(@class, "twitter-tweet")])'
            ' and not(ancestor::div[contains(@class, "cnnStoryElementBox")])'
            ' and not(descendant::*[starts-with(text(), "ZOBACZ TAKŻE:")])'
            ' and not(self::*[contains(@dir, "ltr")])'
        )

        # text = response.css('div.full__article__body p:not([class^="embed__article"])').extract()
        selector = '//div[contains(@class, "full__article__body")]//p[%s]' % exclude_selectors
        text = response.xpath(selector)
        text = text.extract()

        # W R usunac akapity ze zdjeciami oraz wpisami z twittera - https://t.co/ lub pic.twitter.com/ 
        source = text[-1]
        text.pop(-1)
        text.pop(0)
        text = ' || '.join(text)
        text = remove_tags(text)
        source = remove_tags(source)

        # Joining lead with text
        text = ' || '.join([lead, text])
        text = clear_text(text)

        tags = response.css('div.full__article__tags__list a::attr("title")').extract()
        yield {'url': url,
               'date': date,
               'time': time,
               'title': ''.join(title),
               'lead': lead,
               'text': text,
               'source': source,
               'tags': ', '.join(tags)
               }

    def parse_rmf(self, response):
        url = response.url
        art_id = url.split('nId,')[1]

        date = response.css('.article-date ::attr("content")').extract_first()
        date = date.split('T')
        time = date[1]
        date = date[0]

        title = response.css(".article-header .article-title::text").extract()
        title = ' '.join(title)
        title = replace_escape_chars(title)
        lead = response.css(".article-body .article-lead::text").extract()

        exclude_selectors = (
            'not(self::*[contains(@class, "advert")])'
            # ' and not(ancestor-or-self::*[contains(@class, "embed__article")])'
            # ' and not(ancestor::*[contains(@class, "SandboxRoot")])'
            # ' and not(ancestor::*[contains(@class, "twitter-tweet")])'
            ' and not(self::*[starts-with(text(), "ZOBACZ RÓWNIEŻ:")])'
            ' and not(self::*[starts-with(text(), "SPRAWDŹ:")])'
            ' and not(descendant-or-self::*[contains(@class, "sub")])'
            # # ' and not(contains(descendant-or-self, "b"))'
            ' and not(ancestor-or-self::*[contains(@class, "embed")])'
            ' and not(ancestor-or-self::*[contains(@class, "aside")])'
            ' and not(descendant-or-self::*[contains(@class, "aside")])'
            # ' and not(ancestor-or-self::*[contains(@class, "twitter-widget")])'
            # ' and not(self::*[contains(@class, "Tweet-text")])'
            ' and not(descendant-or-self::u)'
            ' and (self::p[not(contains(@dir, "ltr"))])'
        )
        # art_path = '//div[@class = "article-container"]/div[@class = "article-body"]/div[@class = "articleContent"][not(*/@class = "embed")]/p[not(contains(descendant-or-self, "u") or contains(descendant-or-self, "sub") or contains(descendant-or-self, "b") or contains(ancestor-or-self, "aside")  or contains(descendant-or-self, "aside") or contains(ancestor-or-self, "twitter-widget") or contains(@class, "Tweet-text"))]'
        selector = ('//div[@class = "article-container"]/'
                    'div[@class = "article-body"]/'
                    'div[@class = "articleContent"][not(*/@class = "embed")]//*[%s]') % exclude_selectors
        text = response.xpath(selector)
        text = text.extract()

        text = ' || '.join(text)
        text = text.split("<br><br>")
        text = ' || '.join(text)
        text = remove_tags(text)

        # Joining lead with text
        lead = ' '.join(lead)
        text = ' || '.join([lead, text])
        text = clear_text(text)
        autor = response.css(".article-author-name::text").extract()
        source = response.css(".article-footer .article-source ::attr('content')").extract()
        tags = response.css(".elementTagsList a::text").extract()
        yield {'id': art_id,
               'url': url,
               'date': date,
               'time': time,
               'title': ''.join(title),
               'lead': lead,
               'text': text,
               'autor': ', '.join(autor),
               'source': ', '.join(source),
               'tags': ', '.join(tags)}

    def parse_tvn24(self, response):
        url = response.xpath("//link[@rel='canonical']/@href").extract_first()
        art_id = url.split('-')[-1]

        title = response.xpath("//meta[@property='og:title']/@content").extract_first()

        lead = response.xpath("//meta[@name='description']/@content").extract_first()

        date = response.css('.article-story-header .article-top-bar time::attr("datetime")').extract_first()
        date = date.split('T')
        time = date[1]
        date = date[0]


        exclude_selectors = (
            ''
            'not(self::*[contains(@class, "advert")])'
            ' and not(self::*[contains(@class, "SandboxRoot")])'
            ' and not(self::*[contains(@class, "tweet")])'
            ' and not(self::*[contains(@class, "emb")])'
            ' and not(self::*[contains(@class, "app-ad")])'
            ' and not(self::*[contains(@class, "Reklama")])'
            ' and not(self::*[contains(@class, "comments")])'
            ' and not(descendant-or-self::aside)'
            ' and not(descendant-or-self::strong)'
            ' and not(self::figure)'
            ' and not(descendant-or-self::figcaption)'
            ' and not(ancestor::figcaption)'
            ' and not(descendant-or-self::*[starts-with(text(), "CZYTAJ WIĘCEJ")])'
            ' and not(descendant-or-self::*[starts-with(text(), "ZOBACZ")])'
            ' and not(descendant-or-self::blockquote)'
            ' and not(self::*[contains(@class, "innerArticleModule")])'
            # ' and not(ancestor::twitter-tweet)'
            ' and (self::p[contains(@class, "paragraph") and not(contains(@dir, "ltr"))])'
            # ' and (self::p[contains(@class, "hyphenate") and not(contains(@dir, "ltr"))])'
        )

        selector_text = '//div[contains(@class, "article-story-content__elements")]/' \
                        'div[contains(@class, "article-element--paragraph")]/' \
                        '*[%s]' % exclude_selectors
        text = response.xpath(selector_text).extract()

        if ('Autor:' in text[-1]) and ('Źródło:' in text[-1]):
            source = remove_tags(text[-1]).split(' / ')
            autor = source[0].replace('Autor: ', '')
            source = source[1].replace('Źródło: ', '')
            text = text[:-1]
        else:
            autor = ''
            source = ''

        text = ' || '.join(text)
        text = remove_tags(text)
        text = replace_escape_chars(text)

        # Joining lead with text
        text = ' || '.join([lead, text])
        text = clear_text(text)


        yield {'id': art_id,
               'url': url,
               'date': date,
               'time': time,
               'title': title,
               'lead': lead,
               'text': text,
               'autor': autor,
               'source': source
               }

    def parse_tvn24bis(self, response):
        url = response.url
        art_id = url.split(',')[-1].split('.')[0]

        date = response.css('article.detail header time::attr("datetime")').extract_first()
        date = date.split(' ')
        time = date[1][0:4]
        date = date[0]

        title = response.css("article.detail header h1 ::text").extract_first()
        title = replace_escape_chars(title).strip()

        lead = response.css("div.content p.lead ::text").extract_first()
        lead = replace_escape_chars(lead).strip()

        text = response.xpath(
            '//div[@class="content"]/p[not(contains(@clas, "rules") or contains(@clas, "footer"))]/text()').extract()

        text = ' || '.join(text)
        text = remove_tags(text)
        text = replace_escape_chars(text)
        text = clear_text(text)

        autor = response.css("div.content div.footer ::text").extract()[1].split('/')
        if len(autor) > 1:
            source = autor[1]
            source = source.strip().replace('Źródło: ', '')
            autor = autor[0].strip().replace('Autor: ', '')
        else:
            source = ''
            autor = autor[0].strip().replace('Autor: ', '')

        yield {'id': art_id,
               'url': url,
               'date': date,
               'time': time,
               'title': ''.join(title),
               'lead': lead,
               'text': text,
               'autor': autor,
               'source': source
               }

    def parse_tvp_info(self, response):
        '''Parser for TVP INFO'''
        url = response.url
        art_id = url.split('/')[3]

        date = response.css('.info-article .layout-article .date ::text').extract_first()
        date = date.split(',')
        time = date[1]
        date = date[0]
        date = date.replace('.', '-')

        title = response.xpath("//meta[@property='og:title']/@content").extract_first()

        lead = response.css(".article-layout p.am-article__heading ::text").extract()
        lead = ' '.join(lead)
        lead = remove_tags(lead)

        exclude_selectors = (
            ''
            'not(self::*[contains(@class, "advert")])'
            ' and not(self::*[contains(@class, "embed__article")])'
            ' and not(self::*[contains(@class, "SandboxRoot")])'
            ' and not(self::*[contains(@class, "twitter-tweet")])'
            ' and not(self::*[contains(@class, "am-article__image")])'
            ' and not(self::*[contains(@class, "facebook-paragraph")])'
            ' and not(self::*[contains(@class, "am-article__source")])'
            ' and not(self::*[contains(@class, "article-tags")])'
            ' and not(self::*[contains(@class, "Tweet")])'
            ' and not(self::*[contains(@class, "embed")])'
            ' and not(self::*[contains(@class, "social-article")])'
            ' and not(self::*[contains(@class, "video-module")])'
            ' and not(self::a)'
        )

        selector_text = '//div[contains(@class, "article-layout")]/*[%s]//text()' % exclude_selectors
        text = response.xpath(selector_text).extract()
        text = ' || '.join(text)
        text = clear_text(text)

        autor = response.css(".info-article__header .info-article__date .name ::text").extract()
        tags = response.css(".article-tags .article-tags__tag::text").extract()
        tags = ', '.join(tags)
        tags = clear_text(tags)

        source = response.css(".am-article__source .am-article__tag ::text").extract()

        yield {'id': art_id,
               'url': url,
               'date': date,
               'time': time,
               'title': ''.join(title),
               'lead': lead,
               'text': text,
               'autor': ', '.join(autor),
               'tags': tags,
               'source': ', '.join(source)}

    def parse_polsat_news(self, response):
        '''Parser for Polsat News'''
        url = response.xpath("//link[@rel='canonical']/@href").extract_first()
        art_id = response.css('.container__col--main article ::attr("data-id")').extract_first()

        date = response.css('.news__header .news__info .news__time ::attr("datetime")').extract_first()
        date = date.split(' ')
        time = date[1]
        date = date[0]

        title = response.css(".container__col--main .news__header .news__title ::text").extract_first()

        lead = response.css("article.news .news__content .news__preview ::text").extract()
        lead = ' '.join(lead)
        lead = remove_tags(lead)

        exclude_selectors = (
            ''
            'not(self::*[contains(@class, "advert")])'
            ' and not(self::*[contains(@class, "embed__article")])'
            ' and not(self::*[contains(@class, "SandboxRoot")])'
            ' and not(self::*[contains(@class, "twitter-tweet")])'
            ' and not(self::*[contains(@class, "am-article__image")])'
            ' and not(self::*[contains(@class, "facebook-paragraph")])'
            ' and not(self::*[contains(@class, "am-article__source")])'
            ' and not(self::*[contains(@class, "article-tags")])'
            ' and not(self::*[contains(@class, "tweet")])'
            ' and not(self::*[contains(@class, "embed")])'
            ' and not(self::*[contains(@class, "social-article")])'
            ' and not(self::*[contains(@class, "video-module")])'
            ' and not(self::*[contains(@class, "related")])'
            ' and not(self::*[contains(@class, "news__rndvod")])'
            ' and not(self::*[contains(@class, "news__vodevents")])'
            ' and not(self::*[contains(@class, "photos-container")])'
            ' and not(self::*[contains(@class, "app-ad")])'
            ' and not(self::*[contains(@class, "news__comments")])'
            ' and not(self::*[contains(@class, "news__author")])'
            ' and not(self::*[contains(@class, "videoPlayer")])'
            ' and not(self::a)'
            ' and not(self::span)'
            ' and not(self::strong)'
            ' and not(self::blockquote)'
            # ' and not(ancestor::twitter-tweet)'
            ' and (self::p[not(contains(@dir, "ltr"))])'
        )

        selector_text = '//div[contains(@class, "news__description")]//*[%s]/text()' % exclude_selectors
        text = response.xpath(selector_text).extract()
        text = ' || '.join(text)
        text = clear_text(text)
        text = ' || '.join([lead, text])

        autor = response.css(".news__description .news__author ::text").extract()
        tags = response.css(".tags .tag::text").extract()
        tags = ', '.join(tags)
        tags = clear_text(tags)

        source = response.css(".news__description .news__author ::text").extract()

        yield {'id': art_id,
               'url': url,
               'date': date,
               'time': time,
               'title': ''.join(title),
               'lead': lead,
               'text': text,
               'autor': ', '.join(autor),
               'tags': tags,
               'source': ', '.join(source)}

    def parse_polskie_radio(self, response):
        '''Parser for Polskie Radio'''
        url = response.xpath("//link[@rel='canonical']/@href").extract_first()
        art_id = url.split('/')[6]
        art_id = art_id.split(',')[0]

        date = response.css('.art-body article .article-time .time ::text').extract_first()
        date = clear_text(date)
        date = date.split(' ')
        time = date[2]
        date = date[1]
        date = date.replace('.', '-')

        title = response.xpath("//meta[@property='og:title']/@content").extract_first()

        lead = response.xpath("//meta[@name='description']/@content").extract_first()

        exclude_selectors = (
            ''
            'not(self::*[contains(@class, "advert")])'
            ' and not(self::*[contains(@class, "embed__article")])'
            ' and not(self::*[contains(@class, "SandboxRoot")])'
            ' and not(self::*[contains(@class, "twitter-tweet")])'
            ' and not(self::*[contains(@class, "am-article__image")])'
            ' and not(self::*[contains(@class, "facebook-paragraph")])'
            ' and not(self::*[contains(@class, "am-article__source")])'
            ' and not(self::*[contains(@class, "article-tags")])'
            ' and not(self::*[contains(@class, "tweet")])'
            ' and not(self::*[contains(@class, "emb")])'
            ' and not(self::*[contains(@class, "social-article")])'
            ' and not(self::*[contains(@class, "video-module")])'
            ' and not(self::*[contains(@class, "related")])'
            ' and not(self::*[contains(@class, "imgdesc")])'
            ' and not(self::*[contains(@class, "app-ad")])'
            ' and not(self::*[contains(@class, "comments")])'
            ' and not(self::a)'
            ' and not(self::span)'
            ' and not(self::b)'
            ' and not(self::blockquote)'
            # ' and not(ancestor::twitter-tweet)'
            ' and (self::p[not(contains(@dir, "ltr"))])'
        )

        selector_text = '//div[contains(@class, "content")]//*[%s]/text()' % exclude_selectors
        text = response.xpath(selector_text).extract()
        text = ' || '.join(text)
        text = clear_text(text)
        text = ' || '.join([lead, text])

        autor = ''
        # tags = response.xpath("//meta[@name='keywords']/@content").extract_first()
        tags = response.css(".tags a::text").extract()
        tags = ', '.join(tags)
        tags = clear_text(tags)
        source = ''

        yield {'id': art_id,
               'url': url,
               'date': date,
               'time': time,
               'title': title,
               'lead': lead,
               'text': text,
               'autor': autor,
               'tags': tags,
               'source': source}

    def parse_wprost(self, response):
        '''Parser for Wprost'''
        url = response.xpath("//link[@rel='canonical']/@href").extract_first()
        art_id = url.split('/')[4]
        if not isfloat(art_id):
            art_id = url.split('/')[5]

        date = response.css('.art-details-datetime time::attr("datetime")').extract_first()
        date = date.split('T')
        time = date[1]
        date = date[0]

        title = response.xpath("//meta[@property='og:title']/@content").extract_first()

        lead = response.xpath("//meta[@name='description']/@content").extract_first()

        exclude_selectors = (
            ''
            'not(self::*[contains(@class, "advert")])'
            ' and not(self::*[contains(@class, "embed__article")])'
            ' and not(self::*[contains(@class, "SandboxRoot")])'
            ' and not(self::*[contains(@class, "twitter-tweet")])'
            ' and not(self::*[contains(@class, "am-article__image")])'
            ' and not(self::*[contains(@class, "aside")])'
            ' and not(self::*[contains(@class, "am-article__source")])'
            ' and not(self::*[contains(@class, "article-tags")])'
            ' and not(self::*[contains(@class, "tweet")])'
            ' and not(self::*[contains(@class, "emb")])'
            ' and not(self::*[contains(@class, "relations")])'
            ' and not(self::*[contains(@class, "video-module")])'
            ' and not(self::*[contains(@class, "related")])'
            ' and not(self::*[contains(@class, "imgdesc")])'
            ' and not(self::*[contains(@class, "teads-adCall")])'
            ' and not(self::*[contains(@class, "comments")])'
            ' and not(self::a)'
            ' and not(self::span)'
            ' and not(self::b)'
            ' and not(self::blockquote)'
            # ' and not(ancestor::twitter-tweet)'
            ' and (self::p[not(contains(@dir, "ltr"))])'
        )

        selector_text = '//div[contains(@class, "art-text-inner")]//*[%s]/text()' % exclude_selectors
        text = response.xpath(selector_text).extract()
        text = ' || '.join(text)
        text = clear_text(text)
        text = ' || '.join([lead, text])

        autor = ''
        tags = ''
        source = ''

        yield {'id': art_id,
               'url': url,
               'date': date,
               'time': time,
               'title': title,
               'lead': lead,
               'text': text,
               'autor': autor,
               'tags': tags,
               'source': source}

    def parse_do_rzeczy(self, response):
        '''Parser for Do Rzeczy'''
        url = response.xpath("//link[@rel='canonical']/@href").extract_first()
        art_id = url.split('/')[-2]

        date = response.css('article.article header .art-details-datetime time::attr("datetime")').extract_first()
        date = date.split('T')
        time = date[1]
        date = date[0]

        title = response.xpath("//meta[@property='og:title']/@content").extract_first()

        lead = response.xpath("//meta[@name='description']/@content").extract_first()

        exclude_selectors = (
            ''
            'not(self::*[contains(@class, "advert")])'
            ' and not(self::*[contains(@class, "embed__article")])'
            ' and not(self::*[contains(@class, "SandboxRoot")])'
            ' and not(self::*[contains(@class, "twitter-tweet")])'
            ' and not(self::*[contains(@class, "am-article__image")])'
            ' and not(self::*[contains(@class, "facebook-paragraph")])'
            ' and not(self::*[contains(@class, "am-article__source")])'
            ' and not(self::*[contains(@class, "article-tags")])'
            ' and not(self::*[contains(@class, "tweet")])'
            ' and not(self::*[contains(@class, "emb")])'
            ' and not(self::*[contains(@class, "social-article")])'
            ' and not(self::*[contains(@class, "video-module")])'
            ' and not(self::*[contains(@class, "related")])'
            ' and not(self::*[contains(@class, "imgdesc")])'
            ' and not(self::*[contains(@class, "app-ad")])'
            ' and not(self::*[contains(@class, "comments")])'
            ' and not(self::a)'
            ' and not(self::span)'
            ' and not(self::b)'
            ' and not(self::blockquote)'
            # ' and not(ancestor::twitter-tweet)'
            ' and (self::p[not(contains(@dir, "ltr"))])'
        )

        selector_text = '//div[contains(@class, "art-text-inner")]//*[%s]/text()' % exclude_selectors
        text = response.xpath(selector_text).extract()
        text = ' || '.join(text)
        text = clear_text(text)
        text = ' || '.join([lead, text])

        autor = ''
        tags = response.xpath("//meta[@name='keywords']/@content").extract_first()
        # tags = response.css(".tags a::text").extract()
        source = ''

        yield {'id': art_id,
               'url': url,
               'date': date,
               'time': time,
               'title': title,
               'lead': lead,
               'text': text,
               'autor': autor,
               'tags': tags,
               'source': source}

    def parse_niezalezna(self, response):
        '''Parser for Niezalezna'''
        url = response.xpath("//link[@rel='canonical']/@href").extract_first()
        art_id = url.split('/')[-1]
        art_id = art_id.split('-')[0]

        date = response.css('.articleContent .newsTime::text').extract_first()
        date = date.split(', godz. ')
        time = date[1]
        date = date[0]

        title = response.xpath("//meta[@property='og:title']/@content").extract_first()

        lead = response.xpath("//meta[@name='description']/@content").extract_first()

        exclude_selectors = (
            ''
            'not(self::*[contains(@class, "advert")])'
            ' and not(self::*[contains(@class, "embed__article")])'
            ' and not(self::*[contains(@class, "SandboxRoot")])'
            ' and not(self::*[contains(@class, "twitter-tweet")])'
            ' and not(self::*[contains(@class, "am-article__image")])'
            ' and not(self::*[contains(@class, "facebook-paragraph")])'
            ' and not(self::*[contains(@class, "am-article__source")])'
            ' and not(self::*[contains(@class, "article-tags")])'
            ' and not(self::*[contains(@class, "tweet")])'
            ' and not(self::*[contains(@class, "emb")])'
            ' and not(self::*[contains(@class, "social-article")])'
            ' and not(self::*[contains(@class, "video-module")])'
            ' and not(self::*[contains(@class, "related")])'
            ' and not(self::*[contains(@class, "imgdesc")])'
            ' and not(self::*[contains(@class, "app-ad")])'
            ' and not(self::*[contains(@class, "comments")])'
            ' and not(self::a)'
            ' and not(self::span)'
            ' and not(self::b)'
            ' and not(self::blockquote)'
            # ' and not(ancestor::twitter-tweet)'
            ' and (self::p[not(contains(@dir, "ltr"))])'
        )

        selector_text = '//div[@id="article1"]' \
                        '//div[contains(@class, "articleBody")]//*[%s]/text()' % exclude_selectors
        text = response.xpath(selector_text).extract()
        text = ' || '.join(text)
        text = clear_text(text)
        text = ' || '.join([lead, text])

        autor = ''
        tags = response.xpath("//meta[@name='keywords']/@content").extract_first()
        # tags = response.css(".tags a::text").extract()
        source = ''

        yield {'id': art_id,
               'url': url,
               'date': date,
               'time': time,
               'title': title,
               'lead': lead,
               'text': text,
               'autor': autor,
               'tags': tags,
               'source': source}

    def parse_tok_fm(self, response):
        '''Parser for TOK FM'''
        url = response.xpath("//link[@rel='canonical']/@href").extract_first()
        art_id = response.css('div.main_content article::attr("data-id")').extract_first()

        date = response.css('.author_and_date .article_date time::attr("datetime")').extract_first()
        date = date.split(' ')
        time = date[1]
        date = date[0]

        title = response.xpath("//meta[@property='og:title']/@content").extract_first()

        lead = response.xpath("//meta[@property='og:description']/@content").extract_first()

        text = response.css('p.art_paragraph').extract()
        text = ' || '.join(text)
        text = remove_tags(text)

        # Joining lead with text
        text = ' || '.join([lead, text])
        text = clear_text(text)

        autor = ''
        tags = response.xpath("//meta[@name='Keywords']/@content").extract_first()
        # tags = response.css(".tags a::text").extract()
        source = ''

        yield {'id': art_id,
               'url': url,
               'date': date,
               'time': time,
               'title': title,
               'lead': lead,
               'text': text,
               'autor': autor,
               'tags': tags,
               'source': source}

    def parse_wpolityce(self, response):
        '''Parser for wPolityce'''
        url = response.xpath("//link[@rel='canonical']/@href").extract_first()
        art_id = url.split('/')[-1]
        art_id = art_id.split('-')[0]

        date = response.css('header .article-meta time::attr("title")').extract_first()
        date = date.split(' ')
        time = date[1]
        date = date[0]

        title = response.xpath("//meta[@property='og:title']/@content").extract_first()

        lead = response.xpath("//meta[@property='og:description']/@content").extract_first()

        exclude_selectors = (
            ''
            'not(self::*[contains(@class, "advert")])'
            ' and not(self::*[contains(@class, "embed__article")])'
            ' and not(self::*[contains(@class, "SandboxRoot")])'
            ' and not(self::*[contains(@class, "twitter-tweet")])'
            ' and not(self::*[contains(@class, "am-article__image")])'
            ' and not(self::*[contains(@class, "facebook-paragraph")])'
            ' and not(self::*[contains(@class, "am-article__source")])'
            ' and not(self::*[contains(@class, "article-tags")])'
            ' and not(self::*[contains(@class, "tweet")])'
            ' and not(self::*[contains(@class, "emb")])'
            ' and not(self::*[contains(@class, "social-article")])'
            ' and not(self::*[contains(@class, "video-module")])'
            ' and not(self::*[contains(@class, "related")])'
            ' and not(self::*[contains(@class, "imgdesc")])'
            ' and not(self::*[contains(@class, "app-ad")])'
            ' and not(self::*[contains(@class, "comments")])'
            ' and not(self::a)'
            ' and not(self::span)'
            ' and not(self::b)'
            ' and not(self::blockquote)'
            # ' and not(ancestor::twitter-tweet)'
            ' and (self::p[not(contains(@dir, "ltr"))])'
        )

        selector_text = '//div[contains(@class, "intext-links")]//*[%s]//text()' % exclude_selectors
        text = response.xpath(selector_text).extract()
        text = ' || '.join(text)
        text = clear_text(text)
        # text = ' || '.join([lead, text])

        autor = ''
        tags = ''
        # tags = response.css(".tags a::text").extract()
        source = ''

        yield {'id': art_id,
               'url': url,
               'date': date,
               'time': time,
               'title': title,
               'lead': lead,
               'text': text,
               'autor': autor,
               'tags': tags,
               'source': source}

    def parse_onet(self, response):
        '''Parser for Onet'''
        article_url = response.css('.content .showMore a ::attr("href")').extract_first()
        if article_url:
            yield response.follow(article_url, callback=self.parse_onet)
        else:
            url = response.xpath("//link[@rel='canonical']/@href").extract_first()
            art_id = url.split('/')[-1]
            # art_id = url

            date = response.xpath("//meta[@property='article:published_time']/@content").extract_first()
            date = date.split(' ')
            time = date[1]
            date = date[0]
            # time = ''

            title = response.xpath("//meta[@property='og:title']/@content").extract_first()

            lead = response.xpath("//meta[@name='description']/@content").extract_first()

            # exclude_selectors = (
            #     ''
            #     'not(self::*[contains(@class, "advert")])'
            #     ' and not(self::*[contains(@class, "embed__article")])'
            #     ' and not(self::*[contains(@class, "SandboxRoot")])'
            #     ' and not(self::*[contains(@class, "twitter-tweet")])'
            #     ' and not(self::*[contains(@class, "am-article__image")])'
            #     ' and not(self::*[contains(@class, "facebook-paragraph")])'
            #     ' and not(self::*[contains(@class, "am-article__source")])'
            #     ' and not(self::*[contains(@class, "article-tags")])'
            #     ' and not(self::*[contains(@class, "tweet")])'
            #     ' and not(self::*[contains(@class, "emb")])'
            #     ' and not(self::*[contains(@class, "social-article")])'
            #     ' and not(self::*[contains(@class, "video-module")])'
            #     ' and not(self::*[contains(@class, "related")])'
            #     ' and not(self::*[contains(@class, "imgdesc")])'
            #     ' and not(self::*[contains(@class, "app-ad")])'
            #     ' and not(self::*[contains(@class, "comments")])'
            #     ' and not(self::a)'
            #     ' and not(self::span)'
            #     ' and not(self::b)'
            #     ' and not(self::blockquote)'
            #     # ' and not(ancestor::twitter-tweet)'
            #     # ' and (self::p[not(contains(@dir, "ltr"))])'
            #     # ' and (self::p[contains(@class, "hyphenate")])'
            #     ' and (self::p[contains(@class, "hyphenate") and not(contains(@dir, "ltr"))])'
            # )
            #
            # selector_text = '//div[contains(@class, "articleBody")]//*[%s]' % exclude_selectors
            # text = response.xpath(selector_text).extract()
            text = response.css('article .detail p.hyphenate').extract()
            text = ' || '.join(text)
            text = remove_tags(text)
            text = clear_text(text)
            text = ' || '.join([lead, text])

            autor = ''
            tags = ''
            source = response.css('.authorsSources .sourceOrganization ::text').extract()
            source = ', '.join(source)

            yield {'id': art_id,
                   'url': url,
                   'date': date,
                   'time': time,
                   'title': title,
                   'lead': lead,
                   'text': text,
                   'autor': autor,
                   'tags': tags,
                   'source': source}