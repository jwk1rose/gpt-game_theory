import argparse
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from api_key import api_keys
from llm import Agent

output_path = "../outputs/"
output = []
error = []


def parse_dialog_history(dialog_history):
    """Parse dialog history.
    """
    content = []
    count = 0
    for i in range(len(dialog_history)):
        if dialog_history[i]["role"] == "buyer":
            content.append({"buyer": f"{dialog_history[i]['content']}\n"})
            count += 1
        if dialog_history[i]["role"] == "moderator":
            content.append({"moderator": f"{dialog_history[i]['content']}\n"})
        if dialog_history[i]["role"] == "seller":
            content.append({"seller": f"{dialog_history[i]['content']}\n"})
            count += 1
    return content, int(count / 2)


def define_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_round', type=int, default=10,
                        help='number of rounds')
    parser.add_argument('--verbose', type=int, default=1, help="0: not logger.write, 1: logger.write")
    parser.add_argument('--output_path', type=str, default="./outputs/",
                        help='path to save the output')

    # parse and set arguments
    args = parser.parse_args()

    return args


def run_dr(buyer, seller, moderator,
           n_round=10, no_deal_thres=1):
    """Run single game.
    """

    buyer_run = buyer.memories[-1]["content"]
    start_involve_moderator = False
    deal = False
    deal_price = 0
    no_deal_cnt = 0
    dialog = []
    for _ in range(n_round):
        # 能否10美金的价格成交的回应
        seller_run, contain_money = seller.interact_with_model(buyer_run)
        dialog.append({"role": "seller", "content": seller_run})
        if (start_involve_moderator is False and not contain_money):
            start_involve_moderator = True

        if (start_involve_moderator):
            content, deal = moderator.interact_with_model(input=f"buyer:{buyer_run}\nseller:{seller_run}")
            dialog.append({"role": "moderator", "content": content})
            if deal:
                price_pattern = re.compile(r"\$(\d+\.?\d*)")
                matches = price_pattern.findall(buyer_run + seller_run)
                # 取最后一个匹配项（假设最后一个价格是最终成交价）
                deal_price = float(matches[-1])
                break
            else:
                no_deal_cnt += 1
                if (no_deal_cnt == no_deal_thres): break
        # 买家对卖家的回应
        buyer_run, contain_money = buyer.interact_with_model(seller_run)
        dialog.append({"role": "buyer", "content": buyer_run})

        if (start_involve_moderator is False and not contain_money):
            start_involve_moderator = True

        if (start_involve_moderator):
            content, deal = moderator.interact_with_model(input=f"seller:{seller_run}\nbuyer:{buyer_run}")
            dialog.append({"role": "moderator", "content": content})
            if deal:
                price_pattern = re.compile(r"\$(\d+\.?\d*)")
                matches = price_pattern.findall(seller_run + buyer_run)
                # 取最后一个匹配项（假设最后一个价格是最终成交价）
                deal_price = float(matches[-1])
                break

            else:
                no_deal_cnt += 1
                if (no_deal_cnt == no_deal_thres): break
    if deal:
        return deal_price, dialog
    else:
        return -1, dialog


lock = threading.Lock()  # 用于logger.write的锁


def run_single_experiment(i, args, start_time, run_n_prices_list):
    buyer = Agent(key=api_keys[i], type='buyer')
    seller = Agent(key=api_keys[i], type='seller')
    moderator = Agent(key=api_keys[i], type='moderator')
    global lock  # 使用全局锁
    try:
        price, dialog = run_dr(seller=seller, buyer=buyer, moderator=moderator)  # 确保这个函数返回一个可以解包的对象
        run_n_prices_list.append(price)

        text, count = parse_dialog_history(dialog)

        with lock:  # 获取锁来写日志和更新价格列表
            output.append({"round": i, "price": price, "dialog": text, "count": count,
                           "history": [{"buyer": buyer.parse_memories(buyer.memories)},
                                       {"seller": seller.parse_memories(seller.memories)},
                                       {"moderator": moderator.parse_memories(moderator.memories)}]})

    except Exception as e:
        with lock:
            error.append(f"An exception occurred in thread {i}: {e}.")


def run_dr_simple(args, n_threads=30, n_exp=30):
    run_n_prices_list = []
    start_time = time.time()
    out_file = f"{output_path}json/threads({n_threads})_n({n_exp})_{start_time}.json"
    try:
        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = {executor.submit(run_single_experiment, i, args, start_time, run_n_prices_list) for i in
                       range(n_exp)}

            for future in as_completed(futures):
                if future.exception() is not None:
                    with lock:
                        error.append(f"A thread raised an exception: {future.exception()}")

        mean_price = np.array(run_n_prices_list).mean()
        output.append({"mean_price": mean_price, "error": error})
    finally:
        # 无论是否发生异常，都会执行这一部分的代码
        with open(out_file, 'w') as f:
            json.dump(output, f, indent=4)


if __name__ == '__main__':
    run_dr_simple(args=None)
