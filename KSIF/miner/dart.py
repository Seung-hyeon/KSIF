# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 16:31:43 2016

@author: hyunsik

요청변수 설명
-------------
auth 발급받은 인증키(40자리) (필수)
crp_cd 공시대상회사의 종목코드(상장사)(6자리숫자) 또는 고유번호(기타법인)(8자리숫자)
end_dt 검색종료 접수일자(YYYYMMDD) : 없으면 당일
start_dt 검색시작 접수일자(YYYYMMDD) : 없으면 end_dt
crp_cd가 없는 경우 검색기간은 3개월로 제한
fin_rpt 최종보고서만 검색여부(Y or N) 기본값 : N (정정이 있는 경우 최종정정만 검색)
dsp_tp
정기공시(A), 주요사항보고(B), 발행공시(C), 지분공시(D), 기타공시(E), 외부감사관련(F),
펀드공시(G), 자산유동화(H), 거래소공시(I), 공정위공시(J)
bsn_tp
정기공시(5개), 주요사항보고(3개), 발행공시(11개), 지분공시(4개), 기타공시(9개),
외부감사관련(4개), 펀드공시(3개), 자산유동화(6개), 거래소공시(6개),
공정위공시(5개) (*상세 유형 참조)
sort 접수일자(date), 회사명(crp), 보고서명(rpt) 기본값 : date
series 오름차순(asc), 내림차순(desc) 기본값 : desc
page_no 페이지 번호(1~n) 기본값 : 1
page_set 페이지당 건수(1~100) 기본값 : 10, 최대값 : 100
callback 콜백함수명(JSONP용)

dsp_tp bsn_tp 설명
------------------
A
정기공시
A001 사업보고서
A002 반기보고서
A003 분기보고서
A004 등록법인결산서류(자본시장법이전)
A005 소액공모법인결산서류
B
주요사항보고
B001 주요사항보고서
B002 주요경영사항신고(자본시장법 이전)
B003 최대주주등과의거래신고(자본시장법 이전)
C
발행공시
C001 증권신고(지분증권)
C002 증권신고(채무증권)
C003 증권신고(파생결합증권)
C004 증권신고(합병등)
C005 증권신고(기타)
C006 소액공모(지분증권)
C007 소액공모(채무증권)
C008 소액공모(파생결합증권)
C009 소액공모(합병등)
C010 소액공모(기타)
C011 호가중개시스템을통한소액매출
D
지분공시
D001 주식등의대량보유상황보고서
D002 임원ㆍ주요주주특정증권등소유상황보고서
D003 의결권대리행사권유
D004 공개매수
E
기타공시
E001 자기주식취득/처분
E002 신탁계약체결/해지
E003 합병등종료보고서
E004 주식매수선택권부여에관한신고
E005 사외이사에관한신고
E006 주주총회소집공고
E007 시장조성/안정조작
E008 합병등신고서(자본시장법 이전)
E009 금융위등록/취소(자본시장법 이전)
F
외부감사관련
F001 감사보고서
F002 연결감사보고서
F003 결합감사보고서
F004 회계법인사업보고서
G
펀드공시
G001 증권신고(집합투자증권-신탁형)
G002 증권신고(집합투자증권-회사형)
G003 증권신고(집합투자증권-합병)
H
자산유동화
H001 자산유동화계획/양도등록
H002 사업/반기/분기보고서
H003 증권신고(유동화증권등)
H004 채권유동화계획/양도등록
H005 수시보고
H006 주요사항보고서
I
거래소공시
I001 수시공시
I002 공정공시
I003 시장조치/안내
I004 지분공시
I005 증권투자회사
I006 채권공시
J
공정위공시
J001 대규모내부거래관련
J002 대규모내부거래관련(구)
J004 기업집단현황공시
J005 비상장회사중요사항공시
J006 기타공정위공시
"""

__author__ = 'Hyunski Jung'

import requests
import lxml.html
import re
import urllib


from KSIF.core.utils import date_to_str, to_list

MY_KEY = "593e52ea9510c62b8369a303a362dce77ba19cd8"  # DART API 인증키
URL_DART = "http://dart.fss.or.kr/api/search.json?"  # DART URL
URL_DART_RPT = "http://dart.fss.or.kr/dsaf001/main.do?rcpNo="  # DART report URL
URL_DART_XBRL = "http://dart.fss.or.kr/pdf/download/ifrs.do?lang=ko&rcp_no="  # 영어로 바꾸고 싶으면 lang=ko를 lang=en으로
URL_DART_EXCEL = "http://dart.fss.or.kr/pdf/download/excel.do?lang=ko&rcp_no="


class DART_api:
    def __init__(self, crp_cds, dsp_tps=None, bsn_tps=['A001'],
                 auth=MY_KEY, start='19990101', end=None, fin_rpt="'y'",
                 page_set=100):
        self.crp_cds, self.dsp_tps, self.bsn_tps = to_list(crp_cds, dsp_tps, bsn_tps)
        self.auth = auth
        self.start = start
        self.end = end
        self.page_set = page_set
        self.fin_rpt = fin_rpt
        self.response = {crp_cd: {bsn_tp: requests.get(url(MY_KEY, crp_cd, self.start, self.end,
                                                           bsn_tp, self.fin_rpt, self.page_set))
                                  for bsn_tp in self.bsn_tps} for crp_cd in self.crp_cds}
        self.raw_result = {crp_cd: {bsn_tp: self.response[crp_cd][bsn_tp].json()
                                    for bsn_tp in self.bsn_tps} for crp_cd in self.crp_cds}

        self.result = {crp_cd: {bsn_tp: [self.raw_result[crp_cd][bsn_tp]['list'][i]['rcp_no']
                                         for i in range(len(self.raw_result[crp_cd][bsn_tp]['list']))]
                                for bsn_tp in self.bsn_tps} for crp_cd in self.crp_cds}

    def save(self, path="", type="excel"):
        for crp_cd in self.crp_cds:
            for bsn_tp in self.bsn_tps:
                for i in range(len(self.result[crp_cd][bsn_tp])):
                    rpt_no = self.result[crp_cd][bsn_tp][i]
                    text = requests.get(URL_DART_RPT + str(rpt_no)).text
                    tree = lxml.html.fromstring(text)
                    onclick = tree.xpath('//*[@id="north"]/div[2]/ul/li[1]/a')[0].attrib['onclick']
                    pattern = re.compile("^openPdfDownload\('\d+',\s*'(\d+)'\)")
                    dcm = pattern.search(onclick).group(1)
                    filename = path + "\\" + str(crp_cd) + "_" + str(bsn_tp) + "_" + str(rpt_no)
                    # You can Choose XBRL or EXCEL Type
                    if type == 'excel':
                        url = URL_DART_EXCEL + str(rpt_no) + "&dcm_no=" + dcm
                        filename += '.xls'
                    elif type == 'xbrl':
                        url = URL_DART_XBRL + str(rpt_no) + "&dcm_no=" + dcm
                        # XBRL로 다운받으면 ZIP 파일로 받아짐
                        filename+= ".zip"
                    print(url)
                    urllib.request.urlretrieve(url, filename)


def url(key, crp_cd, start, end, bsn_tp, fin_rpt, page_set):
    if end is None:
        end_url = ""
    else:
        end_url = "&end_dt=%s" % date_to_str(end, form="%Y%m%d")
    return (URL_DART + "auth=%s" % key +
            "&crp_cd=%s" % crp_cd.replace('A', '')
            + "&fin_rpt = %s" % fin_rpt
            + "&start_dt=%s" % date_to_str(start, form="%Y%m%d")
            + end_url + "&bsn_tp=%s" % bsn_tp
            + "&page_set%s" % page_set)


def test():
    crp_cds = ['A005930', 'A012330', 'A096770', 'A010950']
    bsn_tps = ['A001', 'A002', 'A003']  # A001 A002 A003 사업 반기 분기
    result = DART_api(crp_cds, bsn_tps=bsn_tps)
    result.save(r"C:\Users\ysh\Google 드라이브\package\KSIF\data")


if __name__ == '__main__':
    test()

