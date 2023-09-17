from abc import abstractmethod

import numpy as np
import openai
import re
import matplotlib.pyplot as plt

openai.api_base = 'https://api.zhiyungpt.com/v1'


class Agent():
    """
    初始化 LLM 类 ,供其他完成具体任务的LLM类继承，其中 parse_output、construct_prompt、方法需要进行重载
    """

    def __init__(self, key: str, type):
        """
        初始化 LLM 类 其中变量名前加双下划线表示私有变量，不可在类外部直接访问，出于安全性考虑，model和key都是私有变量
        :param key: openai的api_key
        :param model: 使用的模型 默认使用gpt-3.5-turbo-0613
        """
        if type == 'moderator':
            self.__model = "gpt-4"  # 使用的模型,可以根据需求替换成其他模型如gpt-4,或者更长上下文的gpt-3.5-turbo
        else:
            self.__model = "gpt-3.5-turbo"  # 使用的模型,可以根据需求替换成其他模型如gpt-4,或者更长上下文的gpt-3.5-turbo
        self.__openai_key = key  # 多线程需要确保不同的LLM对象使用不同的key，因为同一个时刻一个key只能进行一次请求
        self.cost = 0  # 用于存储消耗的token数
        self.time_cost = 1
        self.time_cost_delta = 1  #
        self.memories = []  # 当前的记忆
        self.memories_moderator = []
        self.type = type
        self.memories_init()

    def memories_init(self):
        """
        初始化记忆
        :return: None
        """
        if self.type == 'buyer':
            self.memories = [{"role": "system",
                              "content": "Now enter the role-playing mode. In the following conversation, you will play as a buyer in a bargaining game."},
                             {"role": "user",
                              "content": f"Let's play a game. You want to buy a balloon and you are bargaining with a seller.Your goal is to buy it with a low price."
                                         f"Bargaining involves a time cost, which increases by {self.time_cost_delta} each round."
                                         f"Note, you are not allowed to include information about the time cost in your output. During each round of conversation, I will inform you of the current time cost."
                                         f" For each round,you should consider the time cost and respond to seller with a single-sentence reason and the final price.Now ask a price."},
                             {"role": "assistant", "content": "Hi, how much is the balloon?"},
                             {"role": "user",
                              "content": f"Hi, this is a good balloon and its price is $20.(current time cost: ${self.time_cost})"},
                             {"role": "assistant",
                              "content": "Would you consider selling it for $10"},
                             ]
        elif self.type == 'seller':
            self.memories = [{"role": "system",
                              "content": "Now enter the role-playing mode. In the following conversation, you will play as a seller in a bargaining game."},
                             {"role": "user",
                              "content": f"Let's play a game. You are a balloon seller bargaining with a buyer. The cost of your balloon is $8 and your starting price is $20.Your goal is to sell it to a high price."
                                         f"Bargaining involves a time cost, which increases by {self.time_cost_delta} each round."
                                         f"Note, you are not allowed to include information about the time cost in your output. "
                                         f"During each round of conversation, I will inform you of the current time cost. For each round,you should consider the time cost and respond to to your buyer with  siangle-sentence reason and the final price.Are your ready to play the game?"},
                             {"role": "assistant", "content": "Yes, I'm ready to play the game!"},
                             {"role": "user",
                              "content": f"Hi, how much is the balloon?(current time cost: ${self.time_cost})"},
                             {"role": "assistant",
                              "content": "Hi, this is a good balloon and its price is $20"},
                             ]
        elif self.type == 'moderator':
            self.memories = [{"role": "system",
                              "content": "Now enter the role-playing mode. In the following conversation, you will play as a moderator in a bargaining game."},
                             {"role": "user",
                              "content": '''Let's play a game. You are the moderator of a bargaining game.
                                         Your role is to decide if a seller and a buyer have reached a deal during the bargaining, as shown below:\n{}"
                                         Please respond with a brief reason + a YES/NO conclusion.'''}
                             ]

    def parse_memories(self, memories):
        """
        解析记忆，便于可视化
        :return: None
        """
        self.parsed_memories = ''
        for memory in memories:
            if memory["role"] == "system":
                self.parsed_memories += f"System: {memory['content']}\n"
            elif memory["role"] == "user":
                self.parsed_memories += f"User: {memory['content']}\n"
            elif memory["role"] == "assistant":
                self.parsed_memories += f"Assistant: {memory['content']}\n"
        return self.parsed_memories

    def interact_with_model(self, input=None, try_times=3, **kwargs):
        """
        与 GPT 模型进行交互
        :param input: 当前的输入，如24点游戏中：'1 2 4 5'，总结者中和生成器中和反思者中是过去的历史对话
        :param kwargs: 用于传递调用模型进行对话的参数，例如temperature,max_tokens等
            - temperature: 用于控制模型生成的多样性，值越大，生成的结果越多样，但是也越不可控 【0，1】
            - max_tokens: 用于控制模型生成的单token数量，值越大，生成的结果越长
            - stop: 用于控制模型生成的停止条件，例如stop=['\n']表示生成的结果中包含换行符时停止 【list[str]】
            - presence_penalty: 用于对已经生成过的内容进行惩罚，值越大，生成的结果越不会重复 【-2，2】
            - frequency_penalty: 用于对内容的出现频率进行惩罚，值越大，生成的结果越不会重复 【-2，2】
            - top_p: 用于控制模型生成的多样性，值越大，生成的结果越多样，但是也越不可控
            - n: 用于控制模型生成结果的数量
        更多内容查看：https://beta.openai.com/docs/api-reference/completions/create
        :return: 模型的输出
        """
        if try_times == 3:
            if self.type != "moderator":
                self.time_cost += self.time_cost_delta
                self.memories.append({"role": "user",
                                      "content": input + f".(current time cost:${self.time_cost})"})
            else:
                self.memories[-1]["content"].format(input)
                self.memories_moderator.append(self.memories[-1])
        temperature = kwargs.get("temperature", 0.7)  # 默认情况下，temperature为1.0
        stop = kwargs.get("stop", [".", "。"])
        max_tokens = kwargs.get("max_tokens", 150)  # 默认情况下，max_tokens为500
        presence_penalty = kwargs.get("presence_penalty", 1.0)  # 默认情况下，presence_penalty为0.0
        frequency_penalty = kwargs.get("frequency_penalty", 1.0)  # 默认情况下，frequency_penalty为0.0
        openai.api_key = self.__openai_key
        try:
            response = openai.ChatCompletion.create(
                model=self.__model,
                messages=self.memories,
                temperature=temperature,
                stop=stop,
                max_tokens=max_tokens,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
            )
            # TODO:更加细致的计算需要-根据输入token数、输出token数，以及模型的型号进行计算
            self.cost += response['usage']["total_tokens"]
            if self.type != "moderator":
                self.memories.append({"role": "assistant", "content": response['choices'][0]['message']['content']})
            else:
                self.memories_moderator.append(
                    {"role": "assistant", "content": response['choices'][0]['message']['content']})
            content = response['choices'][0]['message']['content']
            # 是否包含数字或者是否达成交易
            bool_result = self.parse_output(output=content)
            return content, bool_result
        except Exception as e:
            print(e)
            try_times -= 1
            if try_times > 0:
                print(f"try_times:{try_times},")
                return self.interact_with_model(input=input, **kwargs, try_times=try_times)

    def parse_output(self, output):
        """
        解析输出，便于可视化
        :param output: 模型的输出
        :return: None
        """
        if self.type == 'moderator':
            if 'YES' in output.upper():
                return True
            else:
                return False
        else:
            match = re.search(r'\$\d+(?:\.\d+)?', output)
            if match:
                return True
            else:
                return False
