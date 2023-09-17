import argparse
import re

import openai
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
import numpy as np
from llm import Agent
from api_key import api_keys


def parse_dialog_history(dialog_history):
    """Parse dialog history.
    """
    content = ''
    for i in range(len(dialog_history)):
        if dialog_history[i] == "buyer":
            content += f"buyer:{dialog_history[i]}\n"
        if dialog_history[i] == "moderator":
            content += f"moderator:{dialog_history[i]}\n"
        if dialog_history[i] == "seller":
            content += f"seller:{dialog_history[i]}\n"
    return content


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
    for i in range(n_round):
        # 能否10美金的价格成交的回应
        seller_run, contain_money = seller.interact_with_model(buyer_run)
        dialog.append({"seller": seller_run})
        if (start_involve_moderator is False and not contain_money):
            start_involve_moderator = True

        if (start_involve_moderator):
            content, deal = moderator.interact_with_model(input=f"buyer:{buyer_run}\nseller:{seller_run}")
            dialog.append({"moderator": content})
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
        dialog.append({"buyer": buyer_run})

        if (start_involve_moderator is False and not contain_money):
            start_involve_moderator = True

        if (start_involve_moderator):
            content, deal = moderator.interact_with_model(input=f"seller:{seller_run}\nbuyer:{buyer_run}")
            dialog.append({"moderator": content})
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


# lock = threading.Lock()  # 用于logger.write的锁


# def run_single_experiment(i, args, start_time, run_n_prices_list):
#     global lock  # 使用全局锁
#     retries = 0  # 初始化重试计数器
#
#     while retries < 2:
#         try:
#             final_price, log_str, history = run_dr(buyer, seller, critic, moderator)  # 确保这个函数返回一个可以解包的对象
#
#             with lock:  # 获取锁来写日志和更新价格列表
#                 logger.write("==== ver %s CASE %d, %.2f min ====" % (args.ver, i, compute_time(start_time)))
#                 logger.write(log_str)
#                 logger.write('PRICE: %s' % final_price)
#                 logger.write(
#                     f"-------------------------------buyer-------------------------\n   buyer_dialog:{parse_dialog_history(history['buyer'])}")
#                 logger.write(
#                     f"-------------------------------seller-------------------------\n   buyer_dialog:{parse_dialog_history(history['seller'])}")
#                 logger.write(
#                     f"-------------------------------critic-------------------------\n   buyer_dialog:{parse_dialog_history(history['critic'])}")
#                 if final_price != -1:
#                     run_n_prices_list.append(final_price)
#
#             break  # 如果成功执行，跳出循环
#
#         except Exception as e:
#             with lock:
#                 logger.write(f"An exception occurred in thread {i}: {e}. Retrying... ({retries + 1})")
#             retries += 1  # 更新重试计数器
#             time.sleep(1)  # 可选：等待一段时间再重试
#
#
# def run_dr_simple(args, n_round=10, who_is_first="seller"):
#     start_time = time.time()
#     run_n_prices_list = []
#
#     with ThreadPoolExecutor(max_workers=args.n_threads) as executor:
#         futures = {executor.submit(run_single_experiment, i, args, start_time, run_n_prices_list) for i in
#                    range(args.n_exp)}
#
#         for future in as_completed(futures):
#             if future.exception() is not None:
#                 with lock:
#                     logger.write(f"A thread raised an exception: {future.exception()}")
#
#     mean_price = np.array(run_n_prices_list).mean()
#     logger.write(f"Mean Price: {mean_price}")

if __name__ == '__main__':
    buyer = Agent(key=api_keys[0], type='buyer')
    seller = Agent(key=api_keys[0], type='seller')
    moderator = Agent(key=api_keys[0], type='moderator')
    price, dialog = run_dr(buyer, seller, moderator)
    print(parse_dialog_history(dialog))
