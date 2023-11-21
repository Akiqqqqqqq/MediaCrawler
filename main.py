import argparse
import asyncio
import json
import logging
import os
import shutil
import time
from collections import defaultdict
from datetime import datetime, timedelta

import requests
import schedule
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

import config
import db
from base import proxy_account_pool
from media_platform.douyin import DouYinCrawler
from media_platform.xhs import XiaoHongShuCrawler
from vcdb import VCDB

PLATFORM = "xhs"  
LOGIN_TYPE = "qrcode"
MAX_LIMIT = 16384
GPT_3 = 'gpt-3.5-turbo'
GPT_4 = 'gpt-4-1106-preview'
MODEL=GPT_4
NEWS_FEED_PATH = '/api/v1/newsfeed'

run_at = '05:00'
bot_url='http://127.0.0.1:8010'
db_inited = False
logger = logging.getLogger(__name__)


EXTRACT_TOPIC_PROMPT = '''
假设你是一个专门分析用户感兴趣话题的AI。请分析以下用户记忆内容，确定其中体现的主要兴趣话题topic，并为在网上搜索相关信息提供准确的关键字keywords。请以 JSON 格式回应，包括一个话题和相应的搜索关键字。记忆内容如下：

{}

你的分析结果应该采用以下格式：
{{
  "topic": "话题",
  "keywords": "关键字"
}}
请记住，关键字keyword不要超过3个。关键字之间用空格分开, 不要使用逗号作为分隔符。同时keywords的类型是str。如果分析发现用户没有感兴趣话题，请返回空json: {{}}
'''

class CrawlerFactory:
    @staticmethod
    def create_crawler(platform: str):
        if platform == "xhs":
            return XiaoHongShuCrawler()
        elif platform == "dy":
            return DouYinCrawler()
        else:
            raise ValueError("Invalid Media Platform Currently only supported xhs or dy ...")

async def crawl(platform: str, login_type: str):
    start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logger.info(f'begin today\'s crawling , starttime:{start_time}')

    # 1. query new memories produced in 1 day from milvus
    try:
        recs = query_milvus()
    except Exception:
        logger.error(f"milvus: retry to max limit, abandoned : {e}")
        return
    
    # 2. group memories into <user_id, [m1, m2, ...]>
    grouped_user = defaultdict(list)
    for rec in recs:
        u = rec['user_id']
        content_str = rec['content_str']
        time_str = rec['time']
        time_obj = datetime.strptime(time_str, '%Y-%m-%d %Hh')
        grouped_user[u].append({'content_str': content_str, 'time_obj': time_obj})

    for u, recs in grouped_user.items():
        recs.sort(key=lambda x: x['time_obj'])  # sort by memory time
        all_memories = ', '.join(rec['content_str'] for rec in recs)  # concate memories
        grouped_user[u] = all_memories
    
    # 3. ask LLM to extract a <topic, keywords> from users' memories, and return <keywords, [u1, u2, ...]>
    _ , grouped_key = query_llm(grouped_user)
    if len(grouped_key) == 0:
        logger.warning('LLM returns nothing')
        return

    # 4. begin to crawl xhs
    # init account pool
    logger.info('init account...')
    account_pool = proxy_account_pool.create_account_pool()

    global db_inited
    if not db_inited and config.IS_SAVED_DATABASED:
        logger.info('init db...')
        await db.init_db()
        db_inited = True

    logger.info('creating crawler')
    crawler = CrawlerFactory.create_crawler(platform=platform)
    crawler.init_config(
        platform=platform,
        login_type=login_type,
        account_pool=account_pool
    )

    # for testing
    # grouped_key = defaultdict(list)
    # grouped_key['英语 学习 小红书'] = '001'
    # grouped_key['上海小日子咖啡店 建筑摄影 法租界'] = '002'
    # grouped_key['科幻电影 三体 黑镜'] = '003'
    keywords = []
    for k in grouped_key:
        keywords.append(k)

    logger.info(f'\n{len(keywords)} items to crawl')

    # notes = await crawler.start(keys_str)
    # retry with exp backoff
    try:
        notes = await start_crawler(crawler, keywords)
    except Exception as e:
        logger.error(f"retry to max limit, abandoned : {e}")
        return
    logger.info(f'total num: {len(notes)}')

    # 5. refill <user, [n1, n2, ...]>
    result_data = defaultdict(list)
    for n in notes:
        # TODO: filter out 2 weeks ago's notes
        k = n['keyword']
        u = grouped_key[k]
        result_data[u].append(n)
    
    for u, notes in result_data.items():
        logger.info(f'user:{u}, num of notes: {len(notes)}')
        notes.sort(key=lambda x: int(x['liked_count']), reverse=True)  # sort by liked count
    
    best_result = {}
    for u, notes in result_data.items():
        best_note = notes[0]
        best_result[u] = best_note
        title = best_note['title']
        liked_count = best_note['liked_count']
        logger.info(f'user:{u}, best notes: {title}, liked_count: {liked_count}')

    # 6. to noiz bot
    logger.info(f'{len(best_result)} notes to send')
    send_data_to_api(best_result)
    
    end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logger.info(f'finish today\'s crawling ,endtime:{end_time}')


@retry(stop=stop_after_attempt(10), 
       wait=wait_exponential(multiplier=10, max=300),
       before=lambda retry_state: print(f"try send to bot on {retry_state.attempt_number} times..."),
       after=lambda retry_state: print(f"failed to send to bot on {retry_state.attempt_number} times, error: {retry_state.outcome.exception()}" if retry_state.outcome.failed else "send to bot successfully!"))
def send_data_to_api(data_dict):
    items = [{
        'user_id': user_id,
        'content': {
            'platform': 'xhs',
            'note': note
        }
    } for user_id, note in data_dict.items()]

    payload = {
        'items': items,
        'count': len(items)
    }

    response = requests.post(bot_url + NEWS_FEED_PATH, json=payload)
    if response.status_code == 200:
        print("send successfully")
    else:
        print(f"sent failed, error code:{response.status_code}, response: {response.text}")


def clean_browser_data():
    directory = os.path.join(os.getcwd(), "browser_data")
    if os.path.exists(directory):
        # 删除整个目录
        shutil.rmtree(directory)
        print(f"removed all under {directory}")

def before_retry(retry_state):
    print(f"try crawling on {retry_state.attempt_number} times...")
    if retry_state.attempt_number >= 3:
        clean_browser_data()

# 在第3次重试后，等待时间达到3600秒
@retry(stop=stop_after_attempt(6), 
       wait=wait_exponential(multiplier=900, max=3600),
       before=before_retry,
       after=lambda retry_state: print(f"failed to crawl on {retry_state.attempt_number} times, error: {retry_state.outcome.exception()}" if retry_state.outcome.failed else "crawled successfully!"))
async def start_crawler(crawler: XiaoHongShuCrawler, keywords):
    return await crawler.start(keywords)

@retry(stop=stop_after_attempt(10), 
       wait=wait_exponential(multiplier=10, max=120),
       before=lambda retry_state: print(f"try query milvus on {retry_state.attempt_number} times..."),
       after=lambda retry_state: print(f"failed to query milvus on {retry_state.attempt_number} times, error: {retry_state.outcome.exception()}" if retry_state.outcome.failed else "query milvus successfully!"))
def query_milvus():
    now = datetime.now()
    one_day_ago = now - timedelta(days=3)
    formatted_time = one_day_ago.strftime('%Y-%m-%d %Hh')
    vcdb = VCDB()
    return vcdb.query(expr=f"time > '{formatted_time}'", limit=MAX_LIMIT, ouput_fields=['user_id', 'content_str', 'time'])


def query_llm(grouped_user:dict):
    user_topic = {}
    grouped_key = defaultdict(list)
    for u, m in grouped_user.items():
        try:
            resp = get_completion(EXTRACT_TOPIC_PROMPT.format(m), True)
        except Exception as e:
            print(e)
            continue
        try:
            topic_data = json.loads(resp)
        except Exception as e:
            print(e)
            continue
        if not topic_data or len(topic_data) == 0 or not ('keywords' in topic_data and 'topic' in topic_data):
            continue
        user_topic[u] = topic_data
        grouped_key[topic_data['keywords']].append(u)
        t = topic_data['topic']
        k = topic_data['keywords']
        print(f'user:{u}, topic:{t}, keywords:{k}')
    return user_topic, grouped_key


@retry(stop=stop_after_attempt(5),
       wait=wait_exponential(multiplier=5, max=60),
       before=lambda retry_state: print(f"try query LLM on {retry_state.attempt_number} times..."),
       after=lambda retry_state: print(f"failed to query LLM on {retry_state.attempt_number} times, error: {retry_state.outcome.exception()}" if retry_state.outcome.failed else "query LLM successfully!"))
def get_completion(prompt: str, return_json=False) -> str:
    order = [{"role": "user", "content": prompt}]
    response = OpenAI(api_key=os.environ['OPENAI_API_KEY']).chat.completions.create(
        model=MODEL,
        response_format={"type": "json_object"} if return_json else None,
        messages=order,
    )
    content = response.choices[0].message.content
    if return_json:
        content = content.lstrip("```json").rstrip(
            "```")  # GPT4-preview return extra characters
    return content

# if __name__ == '__main__':
#     try:
#         # asyncio.run(main())
#         asyncio.get_event_loop().run_until_complete(crawl(PLATFORM, LOGIN_TYPE))
#     except KeyboardInterrupt:
#         sys.exit()

    # now = datetime.now()
    # one_day_ago = now - timedelta(days=3)
    # formatted_time = one_day_ago.strftime('%Y-%m-%d %Hh')

    # vcdb = VCDB()
    # recs = vcdb.query(expr=f"time > '{formatted_time}'", limit=MAX_LIMIT, ouput_fields=['user_id','content_str', 'time'])
    # print(recs)
    # memory = '''
    # 用户名字叫杰克，
    # 用户周末打算去盐田度假，希望住一个不错的酒店
    # 用户毕业于清华大学
    # 用户是北京人
    # 用户现在在香港生活
    # '''
    # print(EXTRACT_TOPIC_PROMPT.format(memory))
    # res = get_completion(EXTRACT_TOPIC_PROMPT.format(memory), False)
    # print(res)
    # ls = [{'name':'aki','num':'100'},{'name':'bob','num':'70'},{'name':'david','num':'50'},{'name':'frank','num':'65'}]
    # ls.sort(key=lambda x: int(x['num']))  # 有问题
    # print(ls)

def run_main_task():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(crawl(PLATFORM, LOGIN_TYPE))

if __name__ == '__main__':
    logger.info('Starting cron service...')

    parser = argparse.ArgumentParser(description='Run scheduled tasks.')

    parser.add_argument('-r', '--run_at', type=str,
                        default='05:00', help='Start time in HH:MM format')
    parser.add_argument('-u', '--bot_url', type=str,
                        default='http://127.0.0.1:8010', help='URL of the bot')
    parser.add_argument('-d', '--db_inited', action='store_true',
                        help='Flag to indicate if DB is initialized')

    args = parser.parse_args()
    run_at = args.run_at
    bot_url = args.bot_url
    db_inited = args.db_inited

    logger.info(f"Run at: {run_at}, Bot URL: {bot_url}, DB Initialized: {db_inited}")
    schedule.every().day.at(run_at).do(run_main_task)
    # schedule.every().minute.at(":00").do(run_main_task)
    while True:
        schedule.run_pending()
        time.sleep(1)
